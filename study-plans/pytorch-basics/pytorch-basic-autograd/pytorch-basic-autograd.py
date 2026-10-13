import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """
    print(values)
    values.requires_grad_()
    y = (values**3 + values * 2).sum()

    y.backward()
    print(values.grad)
    return values.grad
