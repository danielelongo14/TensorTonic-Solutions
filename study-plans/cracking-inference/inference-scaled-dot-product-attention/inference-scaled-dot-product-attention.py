import torch
from typing import Optional
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    d_k = query.shape[-1]

    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores += mask

    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ value
