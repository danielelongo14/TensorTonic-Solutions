import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("Num of query heads should be divisible by num kv heads")
    batch, seq, d_model = hidden_states.shape #B, S, d_m

    if d_model % num_query_heads != 0:
        raise ValueError("D model shoud be divisible bt num query heads")
    d_k = d_model // num_query_heads

    query = hidden_states @ w_q #(same shape)
    key = hidden_states @ w_k #(B, S, d_k)
    value = hidden_states @ w_v #(B, S, D) @ (D, dv) -> (B, S, d_V)

    query = query.view(batch, seq, num_query_heads, d_k).transpose(1, 2) # (B, S, QH, D_K) -> (B, QH, S, D_K)

    key = key.view(batch, seq, num_kv_heads, d_k).transpose(1,2)
    value = value.view(batch, seq, num_kv_heads, d_k).transpose(1,2)

    # Now expand k and v to align to the query group
    # Expansion: [K0, K1] -> [K0, K0, K0, K0, K1, K1, K1, K1]
    g = num_query_heads // num_kv_heads  
    key = key.repeat_interleave(g, dim=1) # ( B, num_query_heads, S, d_k)
    value = value.repeat_interleave(g, dim=1) # ( B, num_query_heads, S, d_k)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)

    if causal:
        mask = torch.triu(
            torch.ones((seq,seq), device=scores.device, dtype=torch.bool),
        diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attention = torch.matmul(weights, value)

    concat_att = attention.transpose(1,2).contiguous().view(batch, seq, d_model)
    return concat_att @ w_o

    
    
    
