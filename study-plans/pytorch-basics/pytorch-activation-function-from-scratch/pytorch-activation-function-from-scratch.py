import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    print(method)
    print(x)

    if method == 'relu':
        res = torch.where(x > 0, x, 0.0)

    if method == 'sigmoid':
        res = 1.0 / (1.0 + torch.exp(- x))

    if method == 'tanh':
        magnitude = torch.where(x >= 0, x, -x)
        decay = torch.exp(-2*magnitude)
        positive = (1 - decay) / (1 + decay)
        res = torch.where(x >= 0, positive, -positive)

    if method == 'leaky_relu':
        res = torch.clamp(x, min=0) + 0.01 * torch.clamp(x, max=0)

    return res