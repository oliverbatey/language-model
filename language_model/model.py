import torch
from language_model.layers import Embedding, Linear, RMSNorm, Transformer


class TransformerLM(torch.nn.Module):
    """Transformer language model: embedding → N transformer blocks → RMSNorm → linear head."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        eps: float | None = None,
        device: torch.device | None = None,
    ):
        """Initialise all sub-layers of the transformer language model.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            context_length: Maximum sequence length (used for RoPE precomputation).
            num_layers: Number of stacked transformer blocks.
            d_model: Feature dimension throughout the model.
            num_heads: Number of attention heads per block.
            d_ff: Inner dimension of each SwiGLU feed-forward network.
            theta: RoPE base frequency parameter.
            eps: RMSNorm epsilon for numerical stability (defaults to 1e-5).
            device: Device to create all parameters on.
        """
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.transformer_blocks = torch.nn.ModuleList(
            Transformer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                eps=eps,
                device=device,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model=d_model, device=device)
        self.linear = Linear(d_in=d_model, d_out=vocab_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return unnormalised logits over the vocabulary.

        Args:
            x: Integer tensor of token IDs with shape (batch, seq_len).

        Returns:
            Float tensor of shape (batch, seq_len, vocab_size) with unnormalised logits.
        """
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.linear(self.norm(x))
