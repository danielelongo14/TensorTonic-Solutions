import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    # Cast degli input a float32 prima di eseguire qualsiasi operazione
    x = x.to(torch.float32)
    y = y.to(torch.float32)

    if op == 'add':
        return torch.add(x, y)
    elif op == 'multiply':
        return torch.mul(x, y)
    elif op == 'matmul':
        return torch.matmul(x, y)
    elif op == 'power':
        return torch.pow(x, y)
    elif op == 'max':
        return torch.maximum(x, y)
    else:
        raise ValueError(f"Unsupported operation: {op}")