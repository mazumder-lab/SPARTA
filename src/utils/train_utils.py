import random

import numpy as np
import torch
import argparse
import torch.nn.functional as F


def set_seed(seed):
    """
    Set the seed for reproducibility in Python's random, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to set.
    """
    random.seed(seed)  # Set seed for Python's standard random library
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch CPU operations

    if torch.cuda.is_available():
        # Set seed for PyTorch CUDA operations
        torch.cuda.manual_seed(seed)
        # Set seed for all GPUs (if using more than one)
        torch.cuda.manual_seed_all(seed)
        # Ensure CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        # Disable dynamic algorithm selection for convolution
        torch.backends.cudnn.benchmark = False


def count_parameters(model, all_param_flag=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or all_param_flag)


def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_test_stats(net, testloader, epoch_number, device, criterion, outF):
    print("Computing test stats")

    # [T.1] Switch the net to eval mode
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # [T.2] Cycle through all test batches
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets).mean()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:  # TODO fix
                print(
                    "Epoch: %d, Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        epoch_number,
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    )
                )
    acc = 100.0 * correct / total
    print("For epoch: {}, test loss: {} and accuracy: {}".format(epoch_number, test_loss / (batch_idx + 1), acc))
    outF.write("For epoch: {}, test loss: {} and accuracy: {}".format(epoch_number, test_loss / (batch_idx + 1), acc))
    outF.write("\n")
    outF.flush()

    return acc, test_loss / (batch_idx + 1)
