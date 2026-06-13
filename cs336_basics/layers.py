import torch
from einops import einsum, rearrange, repeat
from math import sqrt

class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super(Linear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device  # The device to store layer parameters on
        self.dtype = dtype
        self._initialise_weights()
    
    @staticmethod
    def _initial_weight_std(d_in: int, d_out: int) -> float:
        return sqrt(2/(d_in + d_out))
    
    def _initialise_weights(self):
        W_std = self._initial_weight_std(self.d_in, self.d_out)
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(self.d_out, self.d_in), mean=0, std=W_std, a=-3*W_std, b=3*W_std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, sequence_length, d_model)
        # W.shape = (d_out, d_in)
        # We need sum over the third dimension of x which means we need d_model==d_in
        Wt = rearrange(self.W, "d_model d_out -> d_out d_model")
        return einsum(x, Wt, "batch_size sequence_length d_model, d_model d_out -> batch_size sequence_length d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_matrix = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), mean=0.0, std=1.0, a=-3.0, b=3.0)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # embedding_matrix.shape = (vocab_size, d_model)
        # token_ids.shape = (batch_size, sequence_length)
        # output.shape = (batch_size, sequence_length, d_model)
        return self.embedding_matrix[token_ids]
        

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = torch.nn.Parameter(torch.ones(d_model))
    
    def _rms(self, a: torch.Tensor) -> float:
        return torch.sqrt((1 / self.d_model) * einsum(a, a, "... d_model, ... d_model -> ...") + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Pytorch would broadcast rms automatically, but I've included it for explicit clarity.
        rms = repeat(self._rms(x), "... -> ... d_model", d_model=self.d_model)
        return torch.multiply(torch.div(x, rms), self.gain).to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_in: int, d_ff: int):
        super(SwiGLU, self).__init__()
        self.d_in = d_in
        self.d_ff = d_ff
        self.W1 = Linear(d_in=d_in, d_out=d_ff)
        self.W2 = Linear(d_in=d_ff, d_out=d_in)
        self.W3 = Linear(d_in=d_in, d_out=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = self.W1(x) 
        return self.W2(
            x_trans * torch.sigmoid(x_trans) * self.W3(x)
        )
    

class RotaryPositionalEmbedding(torch.nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        """
        Initialize RoPE module with precomputed sin/cos buffers.

        Args:
            theta: Base for exponential frequency decay (Θ in the paper)
            d_k: Dimension of query/key vectors
            max_seq_len: Maximum sequence length to support
            device: Device for buffers
            dtype: Data type for computation
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute θ_i = θ^(-2i/d_k) for each dimension pair i
        # For position m and pair i, the rotation angle is m * θ_i
        positions = torch.arange(max_seq_len, device=device, dtype=dtype or torch.float32)
        freqs = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=dtype or torch.float32) / d_k)
        angles = positions[:, None] * freqs[None, :]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE rotation to input.

        Args:
            x: Input of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)

        Returns:
            Rotated tensor of shape (..., seq_len, d_k)
        """
        # Get precomputed cos/sin for this batch's positions
        cos_seq = self.cos[token_positions]
        sin_seq = self.sin[token_positions]

        # Separate into pairs: (..., seq_len, d_k//2, 2)
        x = rearrange(x, "... seq (pair d_pair) -> ... seq pair d_pair", d_pair=2)
        x0, x1 = x[..., 0], x[..., 1]

        # Apply 2D rotation to each pair
        x0_rot = x0 * cos_seq - x1 * sin_seq
        x1_rot = x0 * sin_seq + x1 * cos_seq

        # Recombine pairs back into d_k
        x_rot = rearrange([x0_rot, x1_rot], "d_pair ... seq pair -> ... seq (pair d_pair)", d_pair=2)

        return x_rot



