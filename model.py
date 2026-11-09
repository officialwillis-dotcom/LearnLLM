import torch
from torch import nn
from torch.nn import functional as F


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normalized = self.ln_1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.feed_forward(self.ln_2(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *(TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer))
        )
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.language_model_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, index: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = index.shape
        if length > self.block_size:
            raise ValueError(f"Input length {length} exceeds block size {self.block_size}")
        positions = torch.arange(length, device=index.device)
        x = self.token_embedding(index) + self.position_embedding(positions)
        logits = self.language_model_head(
            self.final_layer_norm(self.blocks(x))
        )
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, index: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = index[:, -self.block_size :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, next_token), dim=1)
        return index
