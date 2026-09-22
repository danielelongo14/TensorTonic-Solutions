import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    batch, seq, d_model = hidden_states.shape
    d_k = d_model // num_query_heads

    query = hidden_states @ w_q #(BATCH, SEQ, DIM)
    key = hidden_states @ w_k #(BATCH, SEQ, d_k)
    value = hidden_states @ w_v #(B,S,D) -> (D, d_v) -> (B, S, d_v)

    query = query.view(batch, seq, num_query_heads, d_k).transpose(1,2) #(B, S, QH, d_k)

    key = key.unsqueeze(1) #(B,S, d_K) -> (B, 1, S, d_k)
    value = value.unsqueeze(1) # (B,S,d_v) -> (B, S, 1, d_v)
    scores = torch.matmul(query, key.transpose(-2,-1)) / (d_k ** 0.5)

    if causal:
        mask = torch.triu(
            torch.ones((seq, seq), device=scores.device,
            dtype=torch.bool),
            diagonal = 1
        )

        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attention = torch.matmul(weights, value)

    concat_att = attention.transpose(1, 2).contiguous().view(batch, seq, d_model)

    return concat_att @ w_o