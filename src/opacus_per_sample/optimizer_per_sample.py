# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import logging
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from opt_einsum.contract import contract
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from conf.global_settings import SET_optim_fisher_diff_analysis
from opacus_per_sample.optimizer_obc_fisher_mask import (
    create_fisher_obc_mask,
    prune_blocked,
)

logger = logging.getLogger(__name__)


def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.

    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.

    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``

    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor

    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.

    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`

    Args:
        x: gradient tensor or a list of tensors

    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0

    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generated
            (see the notes)

    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).

        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )


class DPOptimizerPerSample(Optimizer):
    """
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add Gaussian noise.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have the ``grad_sample`` attribute.

    On a high level ``DPOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
    2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
    3) Aggregate clipped per sample gradients into ``p.grad``
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    5) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = DPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError("You must provide expected batch size of the loss reduction is mean")

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode

        self.param_groups = self.original_optimizer.param_groups
        self.defaults = self.original_optimizer.defaults
        self.state = self.original_optimizer.state
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        # self.compute_fisher_mask = False
        # self.use_w_tilde = False
        # self.use_fisher_mask_with_true_grads = False
        # self.add_hessian_clipping_and_noise = False
        # self.use_clipped_true_grads = False

        self.compute_fisher_mask = False
        self.use_w_tilde = False
        self.method_name = None

        for p in self.params:
            # summed grad has the clipped gradients sum
            p.summed_grad = None
            p.summed_true_grad = None
            p.summed_grad_sq = None
            p.noise = None

        for p in self.param_groups[1]["params"]:
            p.mask = None
            p.running_true_fisher_hessian = None
            p.running_clipped_true_fisher_hessian = None
            p.running_noisy_fisher_hessian = None
            p.running_combination_clipped_true_noisy_hessian = None
            p.running_true_grad = None
            p.running_clipped_true_grad = None
            p.running_squared_clipped_true_grad = None
            p.running_noisy_grad = None
            p.running_squared_noisy_grad = None
            p.running_combination_clipped_true_noisy_grad = None

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.

        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.

        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated

        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute

        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list

        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        if not hasattr(p, "grad_sample"):
            raise ValueError("Per sample gradient not found. Are you using GradSampleModule?")
        if p.grad_sample is None:
            raise ValueError("Per sample gradient is not initialized. Not updated in backward pass?")
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        return ret

    def signal_skip_step(self, do_skip=True):
        """
        Signals the optimizer to skip an optimization step and only perform clipping and
        per sample gradient accumulation.

        On every call of ``.step()`` optimizer will check the queue of skipped step
        signals. If non-empty and the latest flag is ``True``, optimizer will call
        ``self.clip_and_accumulate``, but won't proceed to adding noise and performing
        the actual optimization step.
        It also affects the behaviour of ``zero_grad()``. If the last step was skipped,
        optimizer will clear per sample gradients accumulated by
        ``self.clip_and_accumulate`` (``p.grad_sample``), but won't touch aggregated
        clipped gradients (``p.summed_grad``)

        Used by :class:`~opacus.utils.batch_memory_manager.BatchMemoryManager` to
        simulate large virtual batches with limited memory footprint.

        Args:
            do_skip: flag if next step should be skipped
        """
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self, pop_next=True):
        """
        Checks if next step should be skipped by the optimizer.
        This is for large Poisson batches that get split into smaller physical batches
        to fit on the device. Batches that do not correspond to the end of a Poisson
        batch or thus `skipped` as their gradient gets accumulated for one big step.
        """
        if self._step_skip_queue:
            if pop_next:
                return self._step_skip_queue.pop(0)
            else:
                return self._step_skip_queue[0]
        else:
            return False

    @property
    def params(self) -> List[nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self)

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        """
        Returns a flat list of per sample gradient tensors (one per parameter)
        """
        ret = []
        for p in self.params:
            ret.append(self._get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        """
        Returns number of batches currently accumulated and not yet processed.

        In other words ``accumulated_iterations`` tracks the number of forward/backward
        passed done in between two optimizer steps. The value would typically be 1,
        but there are possible exceptions.

        Used by privacy accountants to calculate real sampling rate.
        """
        vals = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError("Per sample gradient not found. Are you using GradSampleModule?")
            if isinstance(p.grad_sample, torch.Tensor):
                vals.append(1)
            elif isinstance(p.grad_sample, list):
                vals.append(len(p.grad_sample))
            else:
                raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        if len(set(vals)) > 1:
            raise ValueError("Number of accumulated steps is inconsistent across parameters")
        return vals[0]

    def attach_step_hook(self, fn: Callable[[DPOptimizerPerSample], None]):
        """
        Attaches a hook to be executed after gradient clipping/noising, but before the
        actual optimization step.

        Most commonly used for privacy accounting.

        Args:
            fn: hook function. Expected signature: ``foo(optim: DPOptimizer)``
        """

        self.step_hook = fn

    def flatten_normalize(self, vect: torch.tensor) -> torch.tensor:
        if vect.dim() > 1:
            vect = vect.flatten(start_dim=1)
        vect = vect / (self.expected_batch_size * self.accumulated_iterations)
        return vect

    def update_hessian_true_grads(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            if (self.method_name == "optim_fisher_diff_analysis") and (idx not in SET_optim_fisher_diff_analysis):
                continue
            print(f"Currently in update_hessian_true_grads updating parameter with index {idx}.")
            true_grad = p.summed_true_grad.flatten(start_dim=1) / (
                self.expected_batch_size * self.accumulated_iterations
            )
            try:
                running_fisher_hessian_approx = torch.einsum("lm,lp->lmp", true_grad, true_grad)
            except:
                print(
                    f"Encountered problem at idx={idx}: Cannot store fisher_hessian update in gpu so it is computed in cpu."
                )
                true_grad_cpu = true_grad.to("cpu")
                running_fisher_hessian_approx = torch.einsum("lm,lp->lmp", true_grad_cpu, true_grad_cpu)
            if p.running_true_fisher_hessian is None:
                p.running_true_fisher_hessian = running_fisher_hessian_approx.to("cpu")
                p.running_true_grad = true_grad
            else:
                p.running_true_fisher_hessian += running_fisher_hessian_approx.to("cpu")
                p.running_true_grad += true_grad

    def update_hessian_clipped_true_grads(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            if (self.method_name in ["optim_fisher_diff_analysis"]) and (idx not in SET_optim_fisher_diff_analysis):
                continue
            print(f"Currently updating parameter with index {idx}.")
            clipped_true_grad = p.summed_grad.flatten(start_dim=1) / (
                self.expected_batch_size * self.accumulated_iterations
            )
            try:
                running_fisher_hessian_approx = torch.einsum("lm,lp->lmp", clipped_true_grad, clipped_true_grad)
            except:
                print(
                    f"Encountered problem at idx={idx}: Cannot store fisher_hessian update in gpu so it is computed in cpu."
                )
                clipped_true_grad_cpu = clipped_true_grad.to("cpu")
                running_fisher_hessian_approx = torch.einsum(
                    "lm,lp->lmp", clipped_true_grad_cpu, clipped_true_grad_cpu
                )

            if self.method_name == "optim_noisy_precision":
                try:
                    hessian_noise_matrix = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm / self.expected_batch_size,
                        reference=running_fisher_hessian_approx,
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                except:
                    print("Noise addition cannot take place on gpu - memory overflow.")
                    running_fisher_hessian_approx = running_fisher_hessian_approx.to("cpu")
                    hessian_noise_matrix = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm / self.expected_batch_size,
                        reference=running_fisher_hessian_approx,
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                hessian_noise_matrix = hessian_noise_matrix.view_as(running_fisher_hessian_approx)
                # TODO verify dimensions.
                hessian_noise_matrix = (hessian_noise_matrix + hessian_noise_matrix.transpose(dim0=1, dim1=2)) / 2
                running_fisher_hessian_approx += hessian_noise_matrix
                # running_fisher_hessian_approx.diagonal(dim1=1, dim2=2).clamp_(min=1e-3)
                del hessian_noise_matrix
            elif self.method_name == "optim_fisher_diag_clipped_true_grads":
                diag_running_fisher_hessian = torch.zeros_like(running_fisher_hessian_approx)
                diag_running_fisher_hessian.diagonal(dim1=1, dim2=2).copy_(
                    running_fisher_hessian_approx.diagonal(dim1=1, dim2=2)
                )
                running_fisher_hessian_approx = diag_running_fisher_hessian
                del diag_running_fisher_hessian

            elif self.method_name == "optim_fisher_combination_clipped_true_noisy_grads":
                if p.running_combination_clipped_true_noisy_hessian is None:
                    p.running_combination_clipped_true_noisy_hessian = running_fisher_hessian_approx.to("cpu")
                    p.running_combination_clipped_true_noisy_grad = clipped_true_grad
                else:
                    p.running_combination_clipped_true_noisy_hessian += running_fisher_hessian_approx.to("cpu")
                    p.running_combination_clipped_true_noisy_grad += clipped_true_grad
                continue

            if p.running_clipped_true_fisher_hessian is None:
                p.running_clipped_true_fisher_hessian = running_fisher_hessian_approx.to("cpu")
                p.running_clipped_true_grad = clipped_true_grad
            else:
                p.running_clipped_true_fisher_hessian += running_fisher_hessian_approx.to("cpu")
                p.running_clipped_true_grad += clipped_true_grad

    def update_hessian_half_multiplier_noisy_grad(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter with index {idx}.")
            clipped_true_grad = p.summed_grad.flatten(start_dim=1)
            noise = p.noise.flatten(start_dim=1)
            half_multiplier_noisy_grad = (clipped_true_grad + 0.5 * noise) / (
                self.expected_batch_size * self.accumulated_iterations
            )
            try:
                running_fisher_hessian_approx = torch.einsum(
                    "lm,lp->lmp", half_multiplier_noisy_grad, half_multiplier_noisy_grad
                )
                if "extra_noise" in self.method_name:
                    normalized_noise = noise / (2 * self.expected_batch_size * self.accumulated_iterations)
                    running_fisher_hessian_approx += torch.einsum("lm,lp->lmp", normalized_noise, normalized_noise)
                elif "extra_independent_noise" in self.method_name:
                    independent_noise = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm,
                        reference=clipped_true_grad,
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                    normalized_independent_noise = independent_noise / (
                        2 * self.expected_batch_size * self.accumulated_iterations
                    )
                    running_fisher_hessian_approx += torch.einsum(
                        "lm,lp->lmp", normalized_independent_noise, normalized_independent_noise
                    )
                elif "extra_stability" in self.method_name:
                    stability_hessian = torch.zeros_like(running_fisher_hessian_approx)
                    expected_noise_var = (
                        self.noise_multiplier
                        * self.max_grad_norm
                        / (2 * self.expected_batch_size * self.accumulated_iterations)
                    ) ** 2
                    stability_hessian.diagonal(dim1=1, dim2=2).copy_(torch.tensor(expected_noise_var))
                    running_fisher_hessian_approx += stability_hessian
            except:
                print(
                    f"Encountered problem at idx={idx}: Cannot store fisher_hessian update in gpu so it is computed in cpu."
                )
                half_multiplier_noisy_grad_cpu = half_multiplier_noisy_grad.to("cpu")
                running_fisher_hessian_approx = torch.einsum(
                    "lm,lp->lmp", half_multiplier_noisy_grad_cpu, half_multiplier_noisy_grad_cpu
                )
                if "extra_noise" in self.method_name:
                    normalized_noise = (noise / (2 * self.expected_batch_size * self.accumulated_iterations)).to("cpu")
                    running_fisher_hessian_approx += torch.einsum("lm,lp->lmp", normalized_noise, normalized_noise)
                elif "extra_independent_noise" in self.method_name:
                    independent_noise = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm,
                        reference=clipped_true_grad,
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                    normalized_independent_noise = (
                        independent_noise / (2 * self.expected_batch_size * self.accumulated_iterations)
                    ).to("cpu")
                    running_fisher_hessian_approx += torch.einsum(
                        "lm,lp->lmp", normalized_independent_noise, normalized_independent_noise
                    )
                elif "extra_stability" in self.method_name:
                    stability_hessian = torch.zeros_like(
                        running_fisher_hessian_approx
                    )  # running_fisher_hessian_approx is already on cpu
                    expected_noise_var = (
                        self.noise_multiplier
                        * self.max_grad_norm
                        / (2 * self.expected_batch_size * self.accumulated_iterations)
                    ) ** 2
                    stability_hessian.diagonal(dim1=1, dim2=2).copy_(torch.tensor(expected_noise_var))
                    running_fisher_hessian_approx += stability_hessian

            if p.running_noisy_fisher_hessian is None:
                p.running_noisy_fisher_hessian = running_fisher_hessian_approx.to("cpu")
                p.running_noisy_grad = half_multiplier_noisy_grad
            else:
                p.running_noisy_fisher_hessian += running_fisher_hessian_approx.to("cpu")
                p.running_noisy_grad += half_multiplier_noisy_grad

    def update_hessian_noisy_grad(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter with index {idx}.")
            noisy_grad = p.grad.flatten(start_dim=1) / (self.expected_batch_size * self.accumulated_iterations)
            noise = p.noise.flatten(start_dim=1)
            try:
                running_fisher_hessian_approx = torch.einsum("lm,lp->lmp", noisy_grad, noisy_grad)
                if self.method_name == "optim_fisher_with_noisy_grads_extra_noise":
                    normalized_noise = noise / (self.expected_batch_size * self.accumulated_iterations)
                    running_fisher_hessian_approx += torch.einsum("lm,lp->lmp", normalized_noise, normalized_noise)
            except:
                print(
                    f"Encountered problem at idx={idx}: Cannot store fisher_hessian update in gpu so it is computed in cpu."
                )
                noisy_grad_cpu = noisy_grad.to("cpu")
                running_fisher_hessian_approx = torch.einsum("lm,lp->lmp", noisy_grad_cpu, noisy_grad_cpu)
                if self.method_name == "optim_fisher_with_noisy_grads_extra_noise":
                    normalized_noise = (noise / (self.expected_batch_size * self.accumulated_iterations)).to("cpu")
                    running_fisher_hessian_approx += torch.einsum("lm,lp->lmp", normalized_noise, normalized_noise)

            if self.method_name == "optim_fisher_diag_clipped_noisy_grads":
                diag_running_fisher_hessian = torch.zeros_like(running_fisher_hessian_approx)
                diag_running_fisher_hessian.diagonal(dim1=1, dim2=2).copy_(
                    running_fisher_hessian_approx.diagonal(dim1=1, dim2=2)
                )
                running_fisher_hessian_approx = diag_running_fisher_hessian
                del diag_running_fisher_hessian

            if self.method_name == "optim_fisher_combination_clipped_true_noisy_grads":
                if p.running_combination_clipped_true_noisy_hessian is None:
                    p.running_combination_clipped_true_noisy_hessian = running_fisher_hessian_approx.to("cpu")
                    p.running_combination_clipped_true_noisy_grad = noisy_grad
                else:
                    p.running_combination_clipped_true_noisy_hessian += running_fisher_hessian_approx.to("cpu")
                    p.running_combination_clipped_true_noisy_grad += noisy_grad
                continue

            if p.running_noisy_fisher_hessian is None:
                p.running_noisy_fisher_hessian = running_fisher_hessian_approx.to("cpu")
                p.running_noisy_grad = noisy_grad
            else:
                p.running_noisy_fisher_hessian += running_fisher_hessian_approx.to("cpu")
                p.running_noisy_grad += noisy_grad

    def update_hessian_seperate_heavy_tail_noise(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter with index {idx}.")
            clipped_true_grad = p.summed_grad.flatten(start_dim=1) / (
                self.expected_batch_size * self.accumulated_iterations
            )
            noisy_grad = p.grad.flatten(start_dim=1) / (self.expected_batch_size * self.accumulated_iterations)
            noise = p.noise.flatten(start_dim=1) / (self.expected_batch_size * self.accumulated_iterations)
            if self.method_name == "optim_fisher_seperate_independent_heavy_tail_noise":
                independent_noise = _generate_noise(
                    std=self.noise_multiplier * self.max_grad_norm,
                    reference=clipped_true_grad,
                    generator=self.generator,
                    secure_mode=self.secure_mode,
                )
                noise = independent_noise / (self.expected_batch_size * self.accumulated_iterations)
            try:
                running_fisher_hessian_approx = torch.einsum("lm,lp->lmp", clipped_true_grad, clipped_true_grad)
                running_fisher_hessian_approx += torch.einsum("lm,lp->lmp", noise, noise)
            except:
                print(
                    f"Encountered problem at idx={idx}: Cannot store fisher_hessian update in gpu so it is computed in cpu."
                )
                clipped_true_grad_cpu = clipped_true_grad.to("cpu")
                noise_cpu = noise.to("cpu")
                running_fisher_hessian_approx = torch.einsum(
                    "lm,lp->lmp", clipped_true_grad_cpu, clipped_true_grad_cpu
                )
                running_fisher_hessian_approx += torch.einsum("lm,lp->lmp", noise_cpu, noise_cpu)

            if p.running_noisy_fisher_hessian is None:
                p.running_noisy_fisher_hessian = running_fisher_hessian_approx.to("cpu")
                p.running_noisy_grad = noisy_grad
            else:
                p.running_noisy_fisher_hessian += running_fisher_hessian_approx.to("cpu")
                p.running_noisy_grad += noisy_grad

    def update_true_clipped_grad(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter in update_true_clipped_sq_gradwith index {idx}.")
            clipped_true_grad = self.flatten_normalize(p.summed_grad)
            if p.running_clipped_true_grad is None:
                p.running_clipped_true_grad = clipped_true_grad
            else:
                p.running_clipped_true_grad += clipped_true_grad

    def update_true_clipped_sq_grad(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter in update_true_clipped_sq_gradwith index {idx}.")
            clipped_true_grad = self.flatten_normalize(p.summed_grad)
            if p.running_clipped_true_grad is None:
                p.running_clipped_true_grad = clipped_true_grad
                p.running_squared_clipped_true_grad = clipped_true_grad**2
            else:
                p.running_clipped_true_grad += clipped_true_grad
                p.running_squared_clipped_true_grad += clipped_true_grad**2

    def update_noisy_sq_grad(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter with index {idx}.")
            noisy_grad = self.flatten_normalize(p.grad)
            if self.method_name == "optim_mp_w_noisy_grads":
                if p.running_noisy_grad is None:
                    p.running_noisy_grad = noisy_grad
                    p.running_squared_noisy_grad = noisy_grad**2
                else:
                    p.running_noisy_grad += noisy_grad
                    p.running_squared_noisy_grad += noisy_grad**2
            elif self.method_name == "optim_mp_w_noisy_grads_extra_noise":
                noise = p.noise.flatten(start_dim=1)
                normalized_noise = noise / (2 * self.expected_batch_size * self.accumulated_iterations)
                if p.running_noisy_grad is None:
                    p.running_noisy_grad = noisy_grad
                    p.running_squared_noisy_grad = noisy_grad**2 + normalized_noise**2
                else:
                    p.running_noisy_grad += noisy_grad
                    p.running_squared_noisy_grad += noisy_grad**2 + normalized_noise**2

    def update_noisy_grad_sq(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter with index {idx}.")
            noisy_grad = self.flatten_normalize(p.grad)
            noisy_grad_sq = self.flatten_normalize(p.summed_grad_sq)
            if p.running_noisy_grad is None:
                p.running_noisy_grad = noisy_grad
                p.running_squared_noisy_grad = noisy_grad_sq
            else:
                p.running_noisy_grad += noisy_grad
                p.running_squared_noisy_grad += noisy_grad_sq

    def update_noisy_grad(self):
        for idx, p in enumerate(self.param_groups[1]["params"]):
            print(f"Currently updating parameter with index {idx}.")
            noisy_grad = p.grad.flatten(start_dim=1) / (self.expected_batch_size * self.accumulated_iterations)
            if p.running_noisy_grad is None:
                p.running_noisy_grad = noisy_grad
            else:
                p.running_noisy_grad += noisy_grad

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """
        if self.method_name == "optim_clip_g_and_g2":
            print("Accessed clip_accumulate_clip_g_and_g2")
            len_g = self.grad_samples[0].shape[0]
            self.grad_samples_and_grad_sq = [torch.vstack([g, g**2]) for g in self.grad_samples]
            if len(self.grad_samples_and_grad_sq[0]) == 0:
                # Empty batch
                per_sample_clip_factor = torch.zeros((0,))
            else:
                # CHANGE g here.
                per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples_and_grad_sq]
                per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
                per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

            for p in self.params:
                _check_processed_flag(p.grad_sample)

                grad_sample = self._get_flat_grad_sample(p)
                grad_sample_sq = self._get_flat_grad_sample(p) ** 2
                # grad_and_grad_sq = contract("i,i...", per_sample_clip_factor, torch.vstack([grad_sample, grad_sample_sq]))
                grad = contract("i,i...", per_sample_clip_factor[:len_g], grad_sample)
                grad_sq = contract("i,i...", per_sample_clip_factor[len_g:], grad_sample_sq)

                if p.summed_grad is not None:
                    p.summed_grad += grad
                    p.summed_grad_sq += grad_sq
                else:
                    p.summed_grad = grad
                    p.summed_grad_sq = grad_sq

                _mark_as_processed(p.grad_sample)
            return

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
                if self.method_name in ["optim_fisher_with_true_grads", "optim_fisher_diff_analysis"]:
                    p.summed_true_grad += grad_sample.sum(dim=0)
            else:
                p.summed_grad = grad
                if self.method_name in ["optim_fisher_with_true_grads", "optim_fisher_diff_analysis"]:
                    p.summed_true_grad = grad_sample.sum(dim=0)

            _mark_as_processed(p.grad_sample)

    # def noise_project_clipped_fisher(self):
    #     for idx, p in enumerate(self.param_groups[1]["params"]):
    #         print(f"Currently Kayhan's idea noising the hessian of parameter with index {idx}.", flush=True)
    # hessian_noise = _generate_noise(
    #     std=self.noise_multiplier * self.max_grad_norm / self.expected_batch_size,
    #     reference=p.running_clipped_true_fisher_hessian.flatten(),
    #     generator=self.generator,
    #     secure_mode=self.secure_mode,
    # )
    # hessian_noise_matrix = hessian_noise.view_as(p.running_clipped_true_fisher_hessian)
    # # TODO verify dimensions.
    # hessian_noise_matrix = (hessian_noise_matrix + hessian_noise_matrix.transpose(dim0=1, dim1=2)) / 2
    # p.running_clipped_true_fisher_hessian += hessian_noise_matrix
    # p.running_clipped_true_fisher_hessian.diagonal(dim1=1, dim2=2).clamp_(min=1e-3)

    def get_optimization_method_mask(self, init_weights, sparsity=0.5, correction_coefficient=0.1, verbose=False):
        # Assumes param_groups[1] is the one corresponding to conv2d
        if not self.compute_fisher_mask:
            return
        print("Beginning Fisher pruning.")
        for idx, (p, init_weight) in tqdm(enumerate(zip(self.param_groups[1]["params"], init_weights))):
            W_original = p.data.clone() + init_weight
            W_original = W_original.flatten(start_dim=1)
            rows, columns = W_original.shape[0], W_original.shape[1]

            if self.method_name in [
                "optim_averaged_noisy_grads",
                "optim_averaged_clipped_grads",
                "optim_weights_noisy_grads",
                "optim_mp_w_clipped_grads",
                "optim_mp_w_noisy_grads",
                "optim_mp_w_noisy_grads_extra_noise",
                "optim_clip_g_and_g2",
            ]:
                if self.method_name == "optim_averaged_noisy_grads":
                    mp_entries = p.running_noisy_grad
                if self.method_name == "optim_averaged_clipped_grads":
                    mp_entries = p.running_clipped_true_grad
                elif self.method_name == "optim_weights_noisy_grads":
                    mp_entries = p.running_noisy_grad * W_original  # elementwise multiplication
                elif self.method_name == "optim_mp_w_clipped_grads":
                    correction_coefficient = correction_coefficient if self.use_w_tilde else 0
                    W_opt = (
                        W_original
                        - correction_coefficient * p.running_clipped_true_grad / p.running_squared_clipped_true_grad
                    )
                    mp_entries = W_opt * p.running_clipped_true_grad
                elif self.method_name == "optim_mp_w_noisy_grads":
                    correction_coefficient = correction_coefficient if self.use_w_tilde else 0
                    W_opt = W_original - correction_coefficient * p.running_noisy_grad / p.running_squared_noisy_grad
                    mp_entries = W_opt * p.running_noisy_grad
                elif self.method_name == "optim_mp_w_noisy_grads_extra_noise":
                    correction_coefficient = correction_coefficient if self.use_w_tilde else 0
                    W_opt = W_original - correction_coefficient * p.running_noisy_grad / (p.running_squared_noisy_grad)
                    mp_entries = W_opt * p.running_noisy_grad
                elif self.method_name == "optim_clip_g_and_g2":
                    correction_coefficient = correction_coefficient if self.use_w_tilde else 0
                    W_opt = W_original - correction_coefficient * p.running_noisy_grad / p.running_squared_noisy_grad
                    mp_entries = W_opt * p.running_noisy_grad
                idx_weights = torch.argsort(mp_entries.abs().flatten(), descending=False)
                idx_weights = idx_weights[: int(len(idx_weights) * (1 - sparsity))]
                layerwise_mask = torch.ones_like(mp_entries).flatten()
                layerwise_mask[idx_weights] = 0
                p.mask = layerwise_mask
                continue

            if self.method_name == "optim_fisher_with_true_grads":
                fisher_hessian = p.running_true_fisher_hessian
                gradient = p.running_true_grad if self.use_w_tilde else None
            elif self.method_name in ["optim_fisher_with_clipped_true_grads", "optim_fisher_diag_clipped_true_grads"]:
                fisher_hessian = p.running_clipped_true_fisher_hessian
                gradient = p.running_clipped_true_grad if self.use_w_tilde else None
            elif self.method_name in [
                "optim_fisher_with_noisy_grads",
                "optim_fisher_diag_clipped_noisy_grads",
                "optim_fisher_half_multiplier_noisy_grads",
                "optim_fisher_half_multiplier_noisy_grads_extra_noise",
                "optim_fisher_half_multiplier_noisy_grads_extra_independent_noise",
                "optim_fisher_half_multiplier_noisy_grads_extra_stability",
                "optim_fisher_with_noisy_grads_extra_noise",
            ]:
                fisher_hessian = p.running_noisy_fisher_hessian
                gradient = p.running_noisy_grad if self.use_w_tilde else None
            elif self.method_name == "optim_noisy_precision":
                fisher_hessian = p.running_clipped_true_fisher_hessian
                gradient = p.running_noisy_grad if self.use_w_tilde else None
            elif self.method_name == "optim_fisher_combination_clipped_true_noisy_grads":
                fisher_hessian = p.running_combination_clipped_true_noisy_hessian
                gradient = p.running_combination_clipped_true_noisy_grad if self.use_w_tilde else None

            elif self.method_name == "optim_fisher_diff_analysis":
                if idx not in SET_optim_fisher_diff_analysis:
                    continue
                print("-----------------------")
                print(f"The norm between the fisher matrix obtained thanks to true gradients and clipped gradients is")
                print(torch.norm(p.running_true_fisher_hessian - p.running_clipped_true_fisher_hessian))
                print("-----------------------")
                print(f"The norm between the fisher matrix obtained thanks to true gradients and clipping this matrix")
                clipped_true_matrix = p.running_true_fisher_hessian * max(
                    1.0, 1 / torch.norm(p.running_true_fisher_hessian)
                )
                print(torch.norm(p.running_true_fisher_hessian - clipped_true_matrix))
                print("-----------------------")
                continue

            Loss, Traces = create_fisher_obc_mask(
                fisher_hessian=fisher_hessian,
                W_original=W_original,
                device=p.device,
                parallel=32,
                lambda_stability=0.01,
                use_w_tilde=self.use_w_tilde,
                gradient=gradient,
                correction_coefficient=correction_coefficient,
                use_LDLT_correction=(self.method_name == "optim_noisy_precision"),
            )
            W_s = prune_blocked(Traces, Loss, rows, columns, device=p.device, sparsities=[1 - sparsity])[0]
            mask = (W_s != 0.0).float()
            p.mask = mask
            if verbose:
                print(f"W_original.shape = {W_original.shape}")
                print(f"columns = {columns}")
                print(f"We have for parameter p with dimensions {p.data.shape}:")
                print(f"W_original.shape = {W_original.shape}")
                print(f"Obtained Loss = {Loss.shape}")
                print(f"Traces[0].shape = {Traces[0].shape}")
                print("Finalized mask computation.")
                print(f"mask.shape = {mask.shape}")

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )

            if self.method_name is not None and "extra_noise" in self.method_name:
                p.noise = noise
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)
        del noise

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    def filter_grad(self):
        for p in self.param_groups[1]["params"]:
            if p.mask is not None:
                p.grad = p.grad * p.mask.view_as(p.grad)

    def clear_momentum_grad(self):
        for p in self.param_groups[1]["params"]:
            p.mask = None
            p.running_true_fisher_hessian = None
            p.running_clipped_true_fisher_hessian = None
            p.running_noisy_fisher_hessian = None
            p.running_combination_clipped_true_noisy_hessian = None
            p.running_true_grad = None
            p.running_clipped_true_grad = None
            p.running_squared_clipped_true_grad = None
            p.running_noisy_grad = None
            p.running_squared_noisy_grad = None
            p.running_combination_clipped_true_noisy_grad = None

            param_state = self.state[p]
            if "momentum_buffer" in param_state:
                del param_state["momentum_buffer"]
        self.compute_fisher_mask = False
        self.method_name = None

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None
                p.summed_true_grad = None
                p.summed_grad_sq = None

        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        if self.method_name in ["optim_fisher_with_true_grads", "optim_fisher_diff_analysis"]:
            self.update_hessian_true_grads()
        if self.method_name in [
            "optim_fisher_with_clipped_true_grads",
            "optim_noisy_precision",
            "optim_fisher_diag_clipped_true_grads",
            "optim_fisher_diff_analysis",
            "optim_fisher_combination_clipped_true_noisy_grads",
        ]:
            self.update_hessian_clipped_true_grads()
        if self.method_name == "optim_averaged_clipped_grads":
            self.update_true_clipped_grad()
        if self.method_name == "optim_mp_w_clipped_grads":
            self.update_true_clipped_sq_grad()

        self.add_noise()
        gc.collect()
        torch.cuda.empty_cache()

        if self.method_name in [
            "optim_fisher_half_multiplier_noisy_grads",
            "optim_fisher_half_multiplier_noisy_grads_extra_noise",
            "optim_fisher_half_multiplier_noisy_grads_extra_independent_noise",
            "optim_fisher_half_multiplier_noisy_grads_extra_stability",
        ]:
            self.update_hessian_half_multiplier_noisy_grad()
        if self.method_name in [
            "optim_fisher_with_noisy_grads",
            "optim_fisher_diag_clipped_noisy_grads",
            "optim_fisher_combination_clipped_true_noisy_grads",
            "optim_fisher_with_noisy_grads_extra_noise",
        ]:
            self.update_hessian_noisy_grad()
        elif self.method_name in [
            "optim_fisher_seperate_heavy_tail_noise",
            "optim_fisher_seperate_independent_heavy_tail_noise",
        ]:
            self.update_hessian_seperate_heavy_tail_noise()
        elif self.method_name in ["optim_averaged_noisy_grads", "optim_weights_noisy_grads", "optim_noisy_precision"]:
            self.update_noisy_grad()
        elif self.method_name in ["optim_mp_w_noisy_grads", "optim_mp_w_noisy_grads_extra_noise"]:
            self.update_noisy_sq_grad()
        elif self.method_name == "optim_clip_g_and_g2":
            self.update_noisy_grad_sq()

        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return None

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)
