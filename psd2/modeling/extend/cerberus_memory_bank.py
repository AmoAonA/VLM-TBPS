"""
FIFO Memory Bank for the Cerberus semantic-ID branch.

Inspired by MACA (SIGIR 2024): stores historical embeddings + prototype
indices per attribute group.  At training time the stored entries are
concatenated to the prototype matrix so that the classification head sees
more negatives — exactly the "memory buffer" idea from MACA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CerberusMemoryBank(nn.Module):
    """Per-group FIFO queue of (embedding, prototype_index) pairs."""

    def __init__(
        self,
        queue_size: int = 256,
        embed_dim: int = 512,
        group_names=None,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.embed_dim = embed_dim
        self.group_names = tuple(group_names or ("gender", "hair", "top", "pants", "shoes"))

        for group in self.group_names:
            self.register_buffer(
                f"queue_{group}",
                torch.zeros(queue_size, embed_dim),
            )
            self.register_buffer(
                f"indices_{group}",
                torch.full((queue_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                f"ptr_{group}",
                torch.zeros(1, dtype=torch.long),
            )
            self.register_buffer(
                f"count_{group}",
                torch.zeros(1, dtype=torch.long),
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        group: str,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
    ):
        """Enqueue a batch of (embedding, index) pairs for *group*.

        Only entries with ``indices >= 0`` are stored.  All tensors are
        detached — no gradient flows through the queue.
        """
        valid = indices >= 0
        if not valid.any():
            return

        embs = embeddings[valid].detach()
        idxs = indices[valid].detach()
        n = embs.shape[0]

        queue = getattr(self, f"queue_{group}")
        idx_buf = getattr(self, f"indices_{group}")
        ptr = getattr(self, f"ptr_{group}")
        count = getattr(self, f"count_{group}")

        if n >= self.queue_size:
            embs = embs[-self.queue_size:]
            idxs = idxs[-self.queue_size:]
            n = self.queue_size

        start = int(ptr.item())
        end = start + n
        if end <= self.queue_size:
            queue[start:end] = embs
            idx_buf[start:end] = idxs
        else:
            first = self.queue_size - start
            queue[start:] = embs[:first]
            idx_buf[start:] = idxs[:first]
            rest = n - first
            queue[:rest] = embs[first:]
            idx_buf[:rest] = idxs[first:]

        ptr.fill_((start + n) % self.queue_size)
        count.fill_(min(int(count.item()) + n, self.queue_size))

    def get_memory(self, group: str):
        """Return ``(embeddings [K, D], indices [K])`` for valid entries."""
        count = int(getattr(self, f"count_{group}").item())
        if count == 0:
            device = getattr(self, f"queue_{group}").device
            return (
                torch.zeros(0, self.embed_dim, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        queue_buf = getattr(self, f"queue_{group}")
        indices_buf = getattr(self, f"indices_{group}")
        if count < self.queue_size:
            queue = queue_buf[:count]
            indices = indices_buf[:count]
        else:
            ptr = int(getattr(self, f"ptr_{group}").item())
            queue = torch.cat([queue_buf[ptr:], queue_buf[:ptr]], dim=0)
            indices = torch.cat([indices_buf[ptr:], indices_buf[:ptr]], dim=0)
        return queue, indices
