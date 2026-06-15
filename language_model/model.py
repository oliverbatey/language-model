import torch
from language_model.layers import(
    Embedding,
    Linear,
    RMSNorm,
    Transformer
)

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,  # Number of tranformer blocks
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None, 
        eps: float | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.transformer_blocks = torch.nn.ModuleList()
        self.norm = RMSNorm(d_model=d_model, device=device)
        self.linear = Linear(d_in=d_model, d_out=vocab_size)

        for _ in range(num_layers):
            self.transformer_blocks.append(
                Transformer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    eps=eps,
                    device=device
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return self.linear(self.norm(x))