import torch

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> tuple:
    """
    Returns (output, latent), tensors shaped (batch, seq, model width) and (batch, seq, latent width).
    """
    batch, seq, d_model = hidden_states.shape
    query = hidden_states @ w_q  # (B, S, d_model)

    c = hidden_states @ w_down  # (B, S, d_latent)

    key = c @ w_up_k    # (B, S, d_model)
    value = c @ w_up_v  # (B, S, d_model)

    dk = d_model // num_heads

    def split_heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(batch, seq, num_heads, dk).transpose(1, 2)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)

    scores = (query @ key.transpose(-2, -1)) / (dk ** 0.5)

    if causal:
        mask = torch.triu(
            torch.ones((seq, seq), device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attention = weights @ value

    out = attention.transpose(1, 2).contiguous().view(batch, seq, d_model)
    out = out @ w_o

    return out, c