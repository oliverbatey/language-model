import torch
from einops import einsum, rearrange
from math import sqrt
from language_model.functions import scaled_dot_product_attention


class Linear(torch.nn.Module):
    """Linear transformation y = xW^T with truncated normal weight initialisation."""

    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Initialise a linear layer with truncated normal weights.

        Args:
            d_in: Input feature dimension.
            d_out: Output feature dimension.
            device: Device to create parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        self._initialise_weights()

    @staticmethod
    def _initial_weight_std(d_in: int, d_out: int) -> float:
        """Compute the Glorot initialisation standard deviation."""
        return sqrt(2 / (d_in + d_out))

    def _initialise_weights(self):
        """Create the weight parameter using truncated normal initialisation."""
        W_std = self._initial_weight_std(self.d_in, self.d_out)
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(self.d_out, self.d_in, device=self.device, dtype=self.dtype),
                mean=0, std=W_std, a=-3 * W_std, b=3 * W_std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to x.

        Args:
            x: Input tensor of shape (..., d_in).

        Returns:
            Tensor of shape (..., d_out).
        """
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    """Learnable embedding table mapping token IDs to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """Initialise the embedding matrix with truncated normal weights.

        Args:
            num_embeddings: Vocabulary size.
            embedding_dim: Dimension of each embedding vector.
            device: Device to create parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0.0, std=1.0, a=-3.0, b=3.0,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for a batch of token IDs.

        Args:
            token_ids: Integer tensor of arbitrary shape (...).

        Returns:
            Float tensor of shape (..., embedding_dim).
        """
        return self.embedding_matrix[token_ids]


class RMSNorm(torch.nn.Module):
    """Root mean square layer normalisation with a learnable per-dimension scale."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Initialise RMSNorm with ones for the gain parameter.

        Args:
            d_model: Dimension of the input features.
            eps: Small constant added for numerical stability.
            device: Device to create parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _rms(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the root mean square of a along the last dimension.

        Args:
            a: Input tensor of shape (..., d_model).

        Returns:
            Tensor of shape (..., 1) containing the RMS values.
        """
        return torch.sqrt((a ** 2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x by its RMS and apply the learnable scale.

        Casts to float32 for the normalisation computation, then restores
        the original dtype.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalised tensor of shape (..., d_model).
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        return (x / self._rms(x) * self.gain).to(in_dtype)


class SwiGLU(torch.nn.Module):
    """SwiGLU feed-forward network: W2(SiLU(W1(x)) * W3(x))."""

    def __init__(self, d_in: int, d_ff: int, device=None, dtype=None):
        """Initialise the three linear projections of SwiGLU.

        Args:
            d_in: Input and output feature dimension.
            d_ff: Inner (expanded) feature dimension.
            device: Device to create parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.W1 = Linear(d_in=d_in, d_out=d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_in=d_ff, d_out=d_in, device=device, dtype=dtype)
        self.W3 = Linear(d_in=d_in, d_out=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: project, gate with SiLU, then project back.

        Args:
            x: Input tensor of shape (..., d_in).

        Returns:
            Tensor of shape (..., d_in).
        """
        gate = self.W1(x)
        return self.W2(gate * torch.sigmoid(gate) * self.W3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    """Rotary positional embeddings (RoPE) with precomputed sin/cos buffers."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        """Precompute sin/cos rotation values for all positions up to max_seq_len.

        Args:
            theta: Base for exponential frequency decay (Θ in the paper).
            d_k: Dimension of query/key vectors. Must be even.
            max_seq_len: Maximum sequence length to support.
            device: Device for buffers.
            dtype: Data type for computation.
        """
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # θ_i = θ^(-2i/d_k) for each dimension pair i; rotation angle for position m is m * θ_i
        positions = torch.arange(max_seq_len, device=device, dtype=dtype or torch.float32)
        freqs = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=dtype or torch.float32) / d_k)
        angles = positions[:, None] * freqs[None, :]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to query or key tensor.

        Each consecutive pair of dimensions (2i, 2i+1) is rotated by the angle
        m * θ_i, where m is the token position.

        Args:
            x: Input of shape (..., seq_len, d_k).
            token_positions: Integer tensor of shape (..., seq_len) with token positions.

        Returns:
            Rotated tensor of the same shape as x.
        """
        cos_seq = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin_seq = self.sin[token_positions]

        x = rearrange(x, "... seq (pair d_pair) -> ... seq pair d_pair", d_pair=2)
        x0, x1 = x[..., 0], x[..., 1]

        x0_rot = x0 * cos_seq - x1 * sin_seq
        x1_rot = x0 * sin_seq + x1 * cos_seq

        return rearrange([x0_rot, x1_rot], "d_pair ... seq pair -> ... seq (pair d_pair)", d_pair=2)


class MultiHeadSelfAttention(torch.nn.Module):
    """Causal multi-head self-attention with optional RoPE positional encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
    ):
        """Initialise projection matrices and optionally a RoPE module.

        Q, K, V projections are fused into a single matrix W_qkv for efficiency.
        RoPE is only constructed when both max_seq_len and theta are provided.

        Args:
            d_model: Model (and output) feature dimension.
            num_heads: Number of attention heads. d_model must be divisible by num_heads.
            max_seq_len: Maximum sequence length for RoPE precomputation.
            theta: RoPE base frequency parameter.
            device: Device to create parameters on.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_qkv = Linear(d_in=d_model, d_out=3 * d_model, device=device)
        self.W_o = Linear(d_in=d_model, d_out=d_model, device=device)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention to the input sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        seq_len = x.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        y = self.W_qkv(x)
        Q = y[..., :self.d_model]
        K = y[..., self.d_model:2 * self.d_model]
        V = y[..., 2 * self.d_model:]

        Q = rearrange(Q, "batch seq (num_heads d_k) -> batch num_heads seq d_k", num_heads=self.num_heads)
        K = rearrange(K, "batch seq (num_heads d_k) -> batch num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(V, "batch seq (num_heads d_k) -> batch num_heads seq d_k", num_heads=self.num_heads)

        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(x=Q, token_positions=token_positions)
            K = self.rope(x=K, token_positions=token_positions)

        scores = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)
        multihead = rearrange(scores, "batch num_heads seq d_k -> batch seq (num_heads d_k)")
        return self.W_o(multihead)


class Transformer(torch.nn.Module):
    """Pre-norm transformer block: RMSNorm → MHA → residual, RMSNorm → SwiGLU → residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        eps: float | None = None,
        device: torch.device | None = None,
    ):
        """Initialise the transformer block sub-layers.

        Args:
            d_model: Feature dimension throughout the block.
            num_heads: Number of attention heads.
            d_ff: Inner dimension of the SwiGLU feed-forward network.
            max_seq_len: Maximum sequence length for RoPE (optional).
            theta: RoPE base frequency (optional).
            eps: RMSNorm epsilon for numerical stability (defaults to 1e-5).
            device: Device to create parameters on.
        """
        super().__init__()
        norm_kwargs = {"d_model": d_model, "device": device}
        if eps is not None:
            norm_kwargs["eps"] = eps

        self.norm_1 = RMSNorm(**norm_kwargs)
        self.norm_2 = RMSNorm(**norm_kwargs)
        self.mha = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device
        )
        self.swiglu = SwiGLU(d_in=d_model, d_ff=d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention then pre-norm feed-forward, both with residuals.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        x = x + self.mha(self.norm_1(x))
        x = x + self.swiglu(self.norm_2(x))
        return x
