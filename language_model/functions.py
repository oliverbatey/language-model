import torch
from einops import einsum, rearrange, repeat
from math import sqrt


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x / torch.sum(x, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output o
    """
    QK_t = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    logits = QK_t / sqrt(K.shape[-1])
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    scores = softmax(logits, dim=-1)
    return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")


