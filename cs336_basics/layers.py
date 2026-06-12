import torch
from einops import einsum, rearrange, repeat
from math import sqrt

class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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




