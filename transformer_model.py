"""Implementation of a Transformer model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from student_checks import IncompleteTodoError


class TransformerBlock(nn.Module):
    """Implementation of a Transformer block, which consists of a multi-head
    self-attention layer and a feed-forward network, each preceded by layer
    normalization and followed by a residual connection. A Transformer block is
    also known as an attention layer."""

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        mlp_multiplier: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        # TODO: Implement the multi-head self-attention layer.
        # Hint: Use nn.MultiheadAttention with batch_first=True.
        raise IncompleteTodoError() # delete this line when you implement the TODO
        self.attn = None  # Replace with actual attention layer.
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            # TODO: Fill in the dimensions of the linear layers.
            # Each occurrence of None should be replaced.
            nn.Linear(None, None),
            nn.ReLU(),
            nn.Linear(None, None),
        )
        raise IncompleteTodoError() # delete this line when you implement the TODO above
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size,
                       dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer block."""
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        attn_input = self.ln1(x)
        attn_out, _ = self.attn(attn_input, attn_input,
                                attn_input, attn_mask=mask)
        # TODO: Explain in your own words why we don't just
        # set `x = attn_out` here. What is the technical term
        # for this kind of connection?
        raise IncompleteTodoError() # delete this line when you implement the TODO
        x = x + attn_out

        ff_input = self.ln2(x)
        # TODO: Replace None with an expression that updates x with
        # the output of the feed forward network.
        raise IncompleteTodoError() # delete this line when you implement the TODO
        x = x + None  # Replace with actual feed-forward output.
        return x


class TinyTransformerLM(nn.Module):
    """Implementation of a Tiny Transformer Language Model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
        n_embd: int,
        mlp_multiplier: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=n_embd,
                    n_heads=n_heads,
                    block_size=block_size,
                    mlp_multiplier=mlp_multiplier,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        # TODO: Fill in the dimensions of the language model unembedding.
        # Each occurrence of None should be replaced.
        raise IncompleteTodoError() # delete this line when you implement the TODO
        self.lm_unembedding = nn.Linear(None, None)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the Tiny Transformer Language Model."""
        batch_size, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(positions)[None, :, :]
        # TODO: Implement the forward pass for the attention blocks
        # and final layer normalization. About three lines of code
        # will be needed here.
        raise IncompleteTodoError() # delete this line when you implement the TODO

        logits = self.lm_unembedding(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, -1),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generates new tokens autoregressively given a starting sequence of token IDs.
        Args:
            idx: Tensor of shape (batch_size, current_seq_len) containing the starting token IDs.
            max_new_tokens: The maximum number of new tokens to generate.
        Returns:
            Tensor of shape (batch_size, current_seq_len + max_new_tokens) containing the 
            generated token IDs.
        """
        # TODO: The shapes of the tensors in this method are hard to follow.
        # Write out the exact content of every local variable for the first
        # iteration of the for loop when the input idx has shape (1, 1) and
        # contains the token ID 5 (corresponding to "The").
        # Include your results in comments below.
        raise IncompleteTodoError() # delete this line when you implement the TODO
        for _ in range(max_new_tokens):
            # shape (batch_size, block_size),
            #    or (batch_size, current_seq_len) if current_seq_len < block_size
            ctx_tokens = idx[:, -self.block_size:]
            # shape (batch_size, seq_len, vocab_size)
            logits, _ = self(ctx_tokens)
            # shape (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]
            # shape (batch_size, 1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # shape (batch_size, current_seq_len + 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
