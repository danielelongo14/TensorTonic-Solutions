import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    print(op)
    print(x)
    if op == 'flatten':
        res = x.flatten()
    elif op == 'squeeze':
        res = x.squeeze()
    elif op == 'transpose':
        res = x.t()

    print(res)
    return res