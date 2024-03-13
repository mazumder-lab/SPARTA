import math
import time

import torch


def prune_blocked(Traces, Loss, rows, columns, device, sparsities):
    # Code from the OBC paper
    parallel = Traces[0].shape[1]
    blockcount = Traces[0].shape[0] - 1
    losses = Loss[:, 1:].reshape(-1)
    order = torch.argsort(losses)
    Ws = [torch.zeros((rows, columns), device=device) for _ in sparsities]
    losses = [0] * len(sparsities)
    for i in range(rows):
        if i % parallel == 0:
            Trace = Traces[i // parallel].to(device)
        for j, sparsity in enumerate(sparsities):
            count = int(math.ceil(rows * blockcount * sparsity))
            perrow = torch.sum(torch.div(order[:count], blockcount, rounding_mode="trunc") == i).item()
            losses[j] += torch.sum(Loss[i, : (perrow + 1)]).item()
            Ws[j][i, :] = Trace[perrow, i % parallel, :]
    for sparsity, loss in zip(sparsities, losses):
        print("%.4f error" % sparsity, loss, flush=True)
    return Ws


def prepare_pruning(i1, parallel, W_original, device, GTG, eTG=None):
    rows, columns = W_original.shape[0], W_original.shape[1]
    i2 = min(i1 + parallel, rows)
    count = i2 - i1
    w_old = W_original[i1:i2, :].double()
    mask = torch.zeros_like(w_old).bool()
    mat_hessian = GTG[i1:i2, :, :].to(device)
    grad_sum = eTG[i1:i2, :].to(device) if eTG is not None else None
    deads_W = mat_hessian[:, torch.eye(columns, device=device).bool()] == 0
    # deads_W = torch.diag(mat_hessian) == 0
    # diagonal_elements = mat_hessian.diagonal()
    # deads_W = (diagonal_elements == 0)
    w_old[deads_W] = 0
    mask[w_old == 0] = True
    return i2, count, w_old, mat_hessian, mask, grad_sum


def create_fisher_obc_mask(
    GTG,
    W_original,
    device,
    parallel=32,
    lambda_stability=0.01,
    use_w_tilde=False,
    eTG=None,
    correction_coefficient=0.1,
):
    tick = time.time()
    rows, columns = W_original.shape[0], W_original.shape[1]
    Loss = torch.zeros([rows, columns + 1], device=device)
    Traces = []

    for i1 in range(0, rows, parallel):
        i2, count, w_old, mat_hessian, mask, grad_sum = prepare_pruning(i1, parallel, W_original, device, GTG, eTG)
        rangecount = torch.arange(count, device=device)
        # Add for stability
        to_add = lambda_stability * torch.mean(torch.diagonal(mat_hessian, dim1=1, dim2=2), 1)
        to_add = torch.eye(columns, device=device)[None] * to_add[:, None, None]
        mat_hessian += to_add
        # Check for prunable rows in w_old -> setting corresponding hessians to I:
        idx_0_rows = torch.where(torch.max(torch.abs(w_old), 1).values == 0)[0]
        mat_hessian[idx_0_rows] += torch.eye(columns, device=device)[None]
        # Invert hessian
        while True:
            try:
                mat_hessian = torch.cholesky_inverse(torch.linalg.cholesky(mat_hessian))
                break
            except:
                print("Inside the mat_hessian except.")
                mat_hessian += to_add
        # Update w_old
        if use_w_tilde and grad_sum is not None and (correction_coefficient > 1e-9):
            w_old -= torch.einsum("bmn,bn->bm", mat_hessian, grad_sum) * correction_coefficient
        # Code from OBC
        start = int(torch.min(torch.sum((w_old == 0).float(), 1)).item()) + 1
        Trace = torch.zeros((columns + 1, count, columns), device=device)
        Trace[0, :, :] = w_old
        Trace[:start, :, :] = w_old
        for zeros in range(start, columns + 1):
            diag = torch.diagonal(mat_hessian, dim1=1, dim2=2)
            scores = (w_old**2) / diag
            scores[mask] = float("inf")
            j = torch.argmin(scores, 1)
            Loss[i1:i2, zeros] = scores[rangecount, j]
            row = mat_hessian[rangecount, j, :]
            d = diag[rangecount, j]
            w_old -= row * (w_old[rangecount, j] / d).unsqueeze(1)
            mask[rangecount, j] = True
            w_old[mask] = 0
            Trace[zeros, :, :] = w_old
            if zeros == columns:
                break
            row /= torch.sqrt(d).unsqueeze(1)
            mat_hessian -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
        Loss[i1:i2, :] /= 2
        Traces.append(Trace.cpu())
    if device not in ("cpu", torch.device("cpu")):
        torch.cuda.synchronize()
    print("time %.2f" % (time.time() - tick))
    return Loss, Traces
