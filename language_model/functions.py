import torch
from einops import einsum
from math import sqrt


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax via max subtraction before exponentiation.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute the softmax distribution.

    Returns:
        Tensor of the same shape as x with values summing to 1 along dim.
    """
    x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x / torch.sum(x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute scaled dot-product attention over queries, keys, and values.

    Computes softmax(QK^T / sqrt(d_k)) V, applying an optional boolean mask
    to block certain key positions (e.g. future tokens in causal attention).
    Positions where mask is True are filled with -inf before softmax.

    Args:
        Q: Query tensor of shape (..., queries, d_k).
        K: Key tensor of shape (..., keys, d_k).
        V: Value tensor of shape (..., keys, d_v).
        mask: Boolean tensor of shape (..., queries, keys). True means mask out.

    Returns:
        Tensor of shape (..., queries, d_v).
    """
    QK_t = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    logits = QK_t / sqrt(K.shape[-1])
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    scores = softmax(logits, dim=-1)
    return einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
