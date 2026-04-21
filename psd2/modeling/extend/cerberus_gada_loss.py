"""
Global-Attribute Distribution Alignment (GADA) loss for the Cerberus branch.

Inspired by ACCA (IEEE TMM 2026) L_gada: forces the global visual feature
to be consistent with the per-group attribute predictions from Cerberus.

The global feature is projected into each attribute subspace, and a KL
divergence is computed between the global prediction and the
confidence-weighted group prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CerberusGADALoss(nn.Module):
    """KL alignment between a global projection and Cerberus group logits."""

    def __init__(
        self,
        global_dim: int = 1024,
        semantic_dim: int = 512,
        num_groups: int = None,
        group_names=None,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.group_names = tuple(group_names or ("gender", "hair", "top", "pants", "shoes"))
        self.num_groups = len(self.group_names) if num_groups is None else num_groups

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, semantic_dim * self.num_groups),
            nn.BatchNorm1d(semantic_dim * self.num_groups),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.global_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        visual_embs: torch.Tensor,
        cerberus_embedding_dict: dict,
        prototype_bank,
        logit_scale: float = 30.0,
    ) -> dict:
        """
        Args:
            visual_embs: [N, global_dim] — global visual features.
            cerberus_embedding_dict: per-group embeddings from the Cerberus
                branch, e.g. ``{"gender_embedding": [N,512], ...}``.
            prototype_bank: ``CerberusPrototypeBank`` instance (carries the
                learnable prototype parameters).
            logit_scale: temperature for the cosine classifier.

        Returns:
            ``{"loss_gada": scalar}``
        """
        N = visual_embs.shape[0]
        if N == 0:
            device = visual_embs.device
            return {"loss_gada": torch.tensor(0.0, device=device)}

        # 1. project global feature → per-group sub-vectors
        projected = self.global_proj(visual_embs)  # [N, D*G]
        group_vectors = projected.view(N, self.num_groups, self.semantic_dim)

        total_kl = torch.tensor(0.0, device=visual_embs.device)
        n_valid = 0

        for gi, group_name in enumerate(self.group_names):
            group_emb = cerberus_embedding_dict[f"{group_name}_embedding"]
            prototypes = prototype_bank.get_group_prototypes(group_name)
            proto_norm = F.normalize(prototypes, dim=-1)

            # group logits → softmax → confidence weight
            group_logits = (
                F.normalize(group_emb, dim=-1) @ proto_norm.t() * logit_scale
            )
            group_probs = F.softmax(group_logits, dim=-1)  # [N, C]
            confidence = group_probs.max(dim=-1).values.detach()  # [N]

            # global projection logits → log softmax
            global_logits = (
                F.normalize(group_vectors[:, gi], dim=-1)
                @ proto_norm.t()
                * logit_scale
            )
            global_log_probs = F.log_softmax(global_logits, dim=-1)

            # weighted KL: sum over classes, weighted by confidence per sample
            kl_per_sample = F.kl_div(
                global_log_probs, group_probs.detach(), reduction="none"
            ).sum(dim=-1)  # [N]
            weighted_kl = (kl_per_sample * confidence).mean()
            total_kl = total_kl + weighted_kl
            n_valid += 1

        if n_valid > 0:
            total_kl = total_kl / n_valid

        return {"loss_gada": total_kl}
