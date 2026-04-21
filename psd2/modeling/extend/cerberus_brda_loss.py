"""
Bias-Reduction Dynamic-smoothing Alignment (BRDA) loss for Cerberus.

Inspired by ACCA (IEEE TMM 2026) L_brda: uses cosine similarity between
the embedding and its target prototype to dynamically compute a label
smoothing factor gamma.  High similarity → sharper one-hot; low similarity
→ more uniform smoothing.

Designed to *replace* the vanilla cross-entropy ``loss_cerberus_cls_{group}``
produced by ``CerberusSemanticIDBranch.compute_losses()``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CerberusBRDALoss(nn.Module):
    """Dynamic label-smoothing CE for each Cerberus attribute group."""

    def __init__(
        self,
        smoothing_floor: float = 0.05,
        smoothing_cap: float = 0.6,
        group_names=None,
    ):
        super().__init__()
        self.smoothing_floor = smoothing_floor
        self.smoothing_cap = smoothing_cap
        self.group_names = tuple(group_names or ("gender", "hair", "top", "pants", "shoes"))

    def forward(
        self,
        cerberus_embedding_dict: dict,
        attributes_list: list,
        valid_mask_dict: dict,
        prototype_bank,
        logit_scale: float = 30.0,
    ) -> dict:
        """
        Args:
            cerberus_embedding_dict: ``{"gender_embedding": [N,512], ...}``
            attributes_list: list[dict] — per-sample attribute dicts.
            valid_mask_dict: ``{"gender_valid": [N], ...}`` bool masks.
            prototype_bank: ``CerberusPrototypeBank``.
            logit_scale: cosine classifier temperature.

        Returns:
            ``{"loss_brda_cls_{group}": scalar}`` for each group.
        """
        losses = {}
        device = next(iter(cerberus_embedding_dict.values())).device
        zero = torch.tensor(0.0, device=device)

        for group_name in self.group_names:
            emb = cerberus_embedding_dict[f"{group_name}_embedding"]
            prototypes = prototype_bank.get_group_prototypes(group_name)
            group_valid = None
            if valid_mask_dict is not None:
                gv = valid_mask_dict.get(f"{group_name}_valid")
                if gv is not None:
                    group_valid = gv.detach().cpu().tolist()

            valid_indices = []
            target_indices = []
            for idx, attr_dict in enumerate(attributes_list):
                target = prototype_bank.get_group_index(group_name, attr_dict)
                if target >= 0 and (group_valid is None or group_valid[idx]):
                    valid_indices.append(idx)
                    target_indices.append(target)

            if not valid_indices:
                losses[f"loss_brda_cls_{group_name}"] = zero
                continue

            vi = torch.tensor(valid_indices, device=device)
            ti = torch.tensor(target_indices, device=device)
            valid_emb = F.normalize(emb[vi], dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            target_proto = proto_norm[ti]

            # dynamic gamma from cosine similarity
            cos_sim = F.cosine_similarity(valid_emb, target_proto, dim=-1)
            gamma = (cos_sim + 1.0) / 2.0  # [0, 1]
            gamma = gamma.clamp(self.smoothing_floor, self.smoothing_cap)

            # smoothed labels
            num_classes = prototypes.shape[0]
            one_hot = F.one_hot(ti, num_classes).float()  # [K, C]
            uniform = torch.full_like(one_hot, 1.0 / num_classes)
            gamma_col = gamma.unsqueeze(1)  # [K, 1]
            smoothed = gamma_col * one_hot + (1.0 - gamma_col) * uniform

            # soft cross-entropy
            logits = valid_emb @ proto_norm.t() * logit_scale
            log_probs = F.log_softmax(logits, dim=-1)
            soft_ce = -(smoothed * log_probs).sum(dim=-1).mean()

            losses[f"loss_brda_cls_{group_name}"] = soft_ce

        return losses
