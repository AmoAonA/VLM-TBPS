"""
Cerberus-inspired semantic ID branch for person search.

This is an approximation for person search rather than a faithful reproduction:
- official code for the paper is not available
- person search introduces detector/proposal noise that is absent in standard re-ID
"""

import json
import itertools
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CerberusPrototypeBank(nn.Module):
    carry_fallback_schema = {
        "bag": {"0": "no", "1": "yes", "2": "other/unknown"},
        "umbrella": {"0": "no", "1": "yes", "2": "other/unknown"},
    }

    def __init__(
        self,
        schema_path: str,
        semantic_dim: int = 512,
        use_carry_group: bool = False,
    ):
        super().__init__()
        with open(schema_path, "r", encoding="utf-8") as handle:
            self.schema = json.load(handle)

        self.semantic_dim = semantic_dim
        self.schema_by_attribute = {
            schema_item["attribute"]: schema_item for schema_item in self.schema
        }
        if use_carry_group:
            for attr_name, mapping in self.carry_fallback_schema.items():
                if attr_name not in self.schema_by_attribute:
                    schema_item = {"attribute": attr_name, "mapping": dict(mapping)}
                    self.schema.append(schema_item)
                    self.schema_by_attribute[attr_name] = schema_item
        self.attribute_groups = {
            "gender": ["gender"],
            "hair": ["hair_color", "hair_length"],
            "top": ["top_color", "top_style"],
            "pants": ["pants_color", "pants_style"],
            "shoes": ["shoes_color", "shoes_style"],
        }
        if use_carry_group:
            carry_attrs = [attr_name for attr_name in ("bag", "umbrella")]
            if carry_attrs:
                self.attribute_groups["carry"] = carry_attrs
        self.group_names = tuple(self.attribute_groups.keys())
        self.group_value_sizes = self._build_group_value_sizes()
        self.group_cardinality = self._build_group_cardinality()
        self.group_affinity = self._build_group_affinity()
        self.attribute_value_text = self._build_attribute_value_text()
        self._init_prototypes()

    @staticmethod
    def _normalize_label(label) -> str:
        return str(label).replace("_", " ").strip()

    @staticmethod
    def _maybe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_attribute_value_text(self) -> Dict[str, List[str]]:
        value_text = {}
        for schema_item in self.schema:
            mapping = schema_item["mapping"]
            ordered_values = []
            if isinstance(mapping, dict):
                max_index = -1
                parsed_pairs = []
                for lhs, rhs in mapping.items():
                    lhs_idx = self._maybe_int(lhs)
                    rhs_idx = self._maybe_int(rhs)
                    if lhs_idx is not None and rhs_idx is None:
                        parsed_pairs.append((lhs_idx, self._normalize_label(rhs)))
                        max_index = max(max_index, lhs_idx)
                    elif rhs_idx is not None and lhs_idx is None:
                        parsed_pairs.append((rhs_idx, self._normalize_label(lhs)))
                        max_index = max(max_index, rhs_idx)
                    else:
                        raise ValueError(
                            f"Cannot infer mapping direction for {schema_item['attribute']}: {mapping}"
                        )
                slots = [None] * (max_index + 1)
                for index, label in parsed_pairs:
                    slots[index] = label
                ordered_values = [value if value is not None else "unknown" for value in slots]
            elif isinstance(mapping, list):
                ordered_values = [self._normalize_label(item) for item in mapping]
            else:
                raise TypeError(f"Unsupported mapping type for {schema_item['attribute']}: {type(mapping)}")
            value_text[schema_item["attribute"]] = ordered_values
        return value_text

    def _build_group_value_sizes(self) -> Dict[str, List[int]]:
        value_sizes = {}
        for group_name, attrs in self.attribute_groups.items():
            dims = []
            for attr in attrs:
                schema_item = self.schema_by_attribute.get(attr)
                if schema_item is not None:
                    dims.append(len(schema_item["mapping"]))
            value_sizes[group_name] = dims
        return value_sizes

    def _build_group_cardinality(self) -> Dict[str, int]:
        cardinality = {}
        for group_name, attrs in self.attribute_groups.items():
            group_size = 1
            for dim in self.group_value_sizes[group_name]:
                group_size *= dim
            cardinality[group_name] = group_size
        return cardinality

    def _build_group_affinity(self) -> Dict[str, torch.Tensor]:
        affinity = {}
        for group_name, dims in self.group_value_sizes.items():
            combinations = list(itertools.product(*[range(dim) for dim in dims]))
            matrix = torch.zeros((len(combinations), len(combinations)), dtype=torch.float32)
            denom = max(len(dims), 1)
            for i, lhs in enumerate(combinations):
                for j, rhs in enumerate(combinations):
                    matrix[i, j] = sum(int(a == b) for a, b in zip(lhs, rhs)) / denom
            affinity[group_name] = matrix
        return affinity

    def _init_prototypes(self):
        for group_name, group_size in self.group_cardinality.items():
            proto = nn.Parameter(torch.randn(group_size, self.semantic_dim) * 0.01)
            setattr(self, f"prototypes_{group_name}", proto)
            self.register_buffer(f"affinity_{group_name}", self.group_affinity[group_name].clone())
            self.register_buffer(f"counts_{group_name}", torch.zeros(group_size, dtype=torch.float32))

    def init_prototypes_with_clip(self, clip_model, tokenizer, context_length: int = 77):
        """Initialize prototypes using CLIP text encoder for better starting point."""
        device = next(self.parameters()).device
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]

        for group_name in self.group_names:
            group_size = self.group_cardinality[group_name]
            proto_feats = []

            for idx in range(group_size):
                text = self._get_prototype_text(group_name, idx)
                tokens = [sot] + tokenizer.encode(text) + [eot]
                token_tensor = torch.zeros(context_length, dtype=torch.long)
                if len(tokens) > context_length:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot
                token_tensor[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                token_tensor = token_tensor.to(device)

                with torch.no_grad():
                    text_feat = clip_model.encode_text(token_tensor.unsqueeze(0))
                    text_feat = text_feat[0, token_tensor.argmax()]
                    text_feat = F.normalize(text_feat.float(), dim=-1)
                proto_feats.append(text_feat)

            proto_tensor = torch.stack(proto_feats)
            proto_param = getattr(self, f"prototypes_{group_name}")
            proto_param.data.copy_(proto_tensor)

    def _get_prototype_text(self, group_name: str, index: int) -> str:
        """Generate text description for a prototype given its index."""
        attrs = self.attribute_groups[group_name]
        values = []
        temp_idx = index

        for attr in reversed(attrs):
            count = len(self.schema_by_attribute[attr]["mapping"])
            value_idx = temp_idx % count
            temp_idx = temp_idx // count
            value_text = self.get_attribute_text(attr, value_idx)
            if value_text:
                values.insert(0, value_text)

        if group_name == "gender":
            return f"a {values[0]} person"
        elif group_name == "hair":
            return f"a person with {' '.join(values)} hair"
        elif group_name == "top":
            return f"a person wearing a {' '.join(values)}"
        elif group_name == "pants":
            return f"a person wearing {' '.join(values)}"
        elif group_name == "shoes":
            return f"a person wearing {' '.join(values)} shoes"
        else:
            return f"a person with {' '.join(values)}"

    def get_group_index(self, group_name: str, attributes: Dict[str, int]) -> int:
        if not attributes:
            return -1

        index = 0
        for attr in self.attribute_groups[group_name]:
            schema_item = self.schema_by_attribute.get(attr)
            if schema_item is None:
                return -1
            count = len(schema_item["mapping"])

            if attr not in attributes or attributes[attr] is None or attributes[attr] < 0:
                return -1

            value = min(int(attributes[attr]), count - 1)
            index = index * count + value
        return index

    def get_group_prototypes(self, group_name: str) -> torch.Tensor:
        return getattr(self, f"prototypes_{group_name}")

    def get_attribute_text(self, attribute_name: str, value_index: int) -> str:
        values = self.attribute_value_text.get(attribute_name, [])
        if value_index < 0 or value_index >= len(values):
            return ""
        return values[int(value_index)]

    def get_group_affinity(self, group_name: str) -> torch.Tensor:
        return getattr(self, f"affinity_{group_name}")

    def update_group_counts(
        self,
        group_name: str,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        if indices.numel() == 0:
            return
        counts = getattr(self, f"counts_{group_name}")
        target_indices = indices.detach().to(counts.device)
        if weights is None:
            update_weights = torch.ones_like(
                target_indices, dtype=counts.dtype, device=counts.device
            )
        else:
            update_weights = weights.detach().to(counts.device, dtype=counts.dtype)
        counts.index_add_(0, target_indices, update_weights)

    def get_group_margin(self, group_name: str, indices: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        counts = getattr(self, f"counts_{group_name}").to(indices.device)
        selected = counts[indices]
        total = counts.sum().clamp_min(1.0)
        frequency = selected / total
        return torch.log(alpha * frequency + beta)


class CerberusSemanticIDBranch(nn.Module):
    group_names = ("gender", "hair", "top", "pants", "shoes")
    unknown_label_tokens = ("unknown", "other", "n/a", "unspecified")
    negative_label_tokens = ("no", "none", "without", "absent", "false", "not carrying")
    positive_generic_tokens = ("yes", "true", "present", "carrying", "with")
    group_context_tokens = {
        "gender": ("man", "woman", "male", "female", "person"),
        "hair": ("hair", "haired", "hairstyle"),
        "top": ("top", "shirt", "jacket", "hoodie", "coat", "sweater", "tee", "t shirt"),
        "pants": ("pants", "trousers", "jeans", "shorts", "skirt"),
        "shoes": ("shoes", "sneakers", "boots", "footwear"),
        "carry": ("bag", "backpack", "umbrella", "carrying", "holding"),
    }

    def __init__(
        self,
        embed_dim: int = 1024,
        feature_dim: int = 2048,
        semantic_dim: int = 512,
        num_patches: int = 32,
        schema_path: str = "generated_schema.json",
        logit_scale: float = 30.0,
        prototype_align_weight: float = 0.25,
        guidance_alpha: float = 0.4,
        guidance_beta: float = 1.8,
        relation_reg_weight: float = 0.05,
        use_heatmap_partition: bool = True,
        heatmap_threshold: float = 0.35,
        prior_gain: float = 1.0,
        region_branch_mode: str = "sfan",
        sfan_threshold: float = 0.2,
        sfan_temperature: float = 0.07,
        sfan_branch_weight: float = 0.5,
        query_unknown_weight: float = 0.35,
        query_group_weight_mode: str = "avg",
        query_text_boost_enabled: bool = True,
        query_text_boost: float = 0.20,
        query_text_max_weight: float = 1.35,
        train_unknown_weight: float = 1.0,
        train_group_weight_mode: str = "avg",
        group_loss_weights: Optional[Sequence[float]] = None,
        use_carry_group: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.num_patches = num_patches
        self.logit_scale = logit_scale
        self.prototype_align_weight = prototype_align_weight
        self.guidance_alpha = guidance_alpha
        self.guidance_beta = guidance_beta
        self.relation_reg_weight = relation_reg_weight
        self.use_heatmap_partition = use_heatmap_partition
        self.heatmap_threshold = heatmap_threshold
        self.prior_gain = prior_gain
        self.region_branch_mode = region_branch_mode.lower()
        self.sfan_threshold = sfan_threshold
        self.sfan_temperature = sfan_temperature
        self.sfan_branch_weight = sfan_branch_weight
        self.query_unknown_weight = float(query_unknown_weight)
        self.query_group_weight_mode = str(query_group_weight_mode).lower()
        self.query_text_boost_enabled = bool(query_text_boost_enabled)
        self.query_text_boost = float(query_text_boost)
        self.query_text_max_weight = float(query_text_max_weight)
        self.train_unknown_weight = float(train_unknown_weight)
        self.train_group_weight_mode = str(train_group_weight_mode).lower()
        if self.query_group_weight_mode not in {"avg", "min"}:
            raise ValueError(
                "query_group_weight_mode must be one of {'avg', 'min'}."
            )
        if self.train_group_weight_mode not in {"avg", "min"}:
            raise ValueError(
                "train_group_weight_mode must be one of {'avg', 'min'}."
            )

        self.prototype_bank = CerberusPrototypeBank(
            schema_path,
            semantic_dim,
            use_carry_group=use_carry_group,
        )
        self.group_names = self.prototype_bank.group_names
        self.part_group_names = tuple(
            group_name for group_name in self.group_names if group_name in ("hair", "top", "pants", "shoes")
        )
        self.use_carry_group = "carry" in self.group_names
        self.group_name_to_index = {
            group_name: idx for idx, group_name in enumerate(self.group_names)
        }
        self.register_buffer(
            "group_loss_weights_tensor",
            self._resolve_group_loss_weights(group_loss_weights),
        )

        self.gender_fc = nn.Linear(embed_dim, semantic_dim)
        self.gender_bn = nn.BatchNorm1d(semantic_dim)
        self.gender_text_fc = nn.Linear(embed_dim, semantic_dim)
        self.gender_text_bn = nn.BatchNorm1d(semantic_dim)
        if self.use_carry_group:
            self.carry_fc = nn.Linear(embed_dim, semantic_dim)
            self.carry_bn = nn.BatchNorm1d(semantic_dim)
            self.carry_text_fc = nn.Linear(embed_dim, semantic_dim)
            self.carry_text_bn = nn.BatchNorm1d(semantic_dim)

        hidden_dim = max(feature_dim // 4, semantic_dim)
        self.prior_heatmap_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.region_refiner = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )
        self.sfan_feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.register_buffer("sfan_text_queries", torch.zeros(4, embed_dim))

        self.hair_fc = nn.Linear(feature_dim, semantic_dim)
        self.hair_bn = nn.BatchNorm1d(semantic_dim)
        self.hair_text_fc = nn.Linear(embed_dim, semantic_dim)
        self.hair_text_bn = nn.BatchNorm1d(semantic_dim)
        self.top_fc = nn.Linear(feature_dim, semantic_dim)
        self.top_bn = nn.BatchNorm1d(semantic_dim)
        self.top_text_fc = nn.Linear(embed_dim, semantic_dim)
        self.top_text_bn = nn.BatchNorm1d(semantic_dim)
        self.pants_fc = nn.Linear(feature_dim, semantic_dim)
        self.pants_bn = nn.BatchNorm1d(semantic_dim)
        self.pants_text_fc = nn.Linear(embed_dim, semantic_dim)
        self.pants_text_bn = nn.BatchNorm1d(semantic_dim)
        self.shoes_fc = nn.Linear(feature_dim, semantic_dim)
        self.shoes_bn = nn.BatchNorm1d(semantic_dim)
        self.shoes_text_fc = nn.Linear(embed_dim, semantic_dim)
        self.shoes_text_bn = nn.BatchNorm1d(semantic_dim)
        self._init_weights()

    def _resolve_group_loss_weights(
        self, group_loss_weights: Optional[Sequence[float]]
    ) -> torch.Tensor:
        num_groups = len(self.prototype_bank.group_names)
        if not group_loss_weights:
            return torch.ones(num_groups, dtype=torch.float32)

        resolved = [float(value) for value in group_loss_weights]
        if len(resolved) < num_groups:
            resolved.extend([1.0] * (num_groups - len(resolved)))
        elif len(resolved) > num_groups:
            resolved = resolved[:num_groups]
        return torch.tensor(resolved, dtype=torch.float32)

    def get_group_loss_weight(self, group_name: str) -> float:
        return float(self.group_loss_weights_tensor[self.group_name_to_index[group_name]].item())

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @torch.no_grad()
    def init_sfan_text_queries(self, clip_model, tokenizer, context_length: int = 77):
        """Encode fixed attribute templates with CLIP for SFAN text queries (DiffPS)."""
        templates = ["head", "shirts", "pants", "shoes"]
        device = next(self.parameters()).device
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]
        all_tokens = torch.zeros(len(templates), context_length, dtype=torch.long)
        for i, text in enumerate(templates):
            encoded = [sot] + tokenizer.encode(text) + [eot]
            all_tokens[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        all_tokens = all_tokens.to(device)
        text_feats = clip_model.encode_text(all_tokens)  # [4, 77, C]
        text_feats = text_feats[
            torch.arange(len(templates)), all_tokens.argmax(dim=-1)
        ]  # [4, C]
        text_feats = F.normalize(text_feats.float(), dim=-1)
        self.sfan_text_queries.copy_(text_feats)

    def _find_active_span(
        self, profile: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak = profile.max(dim=1, keepdim=True).values
        active = profile >= (peak * threshold)
        has_active = active.any(dim=1)
        first = active.float().argmax(dim=1)
        last = profile.shape[1] - 1 - active.flip(1).float().argmax(dim=1)
        default_first = torch.zeros_like(first)
        default_last = torch.full_like(last, profile.shape[1] - 1)
        first = torch.where(has_active, first, default_first)
        last = torch.where(has_active, last, default_last)
        return first, last, has_active

    def _masked_pool(self, feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = feature_map * mask
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return weighted.sum(dim=(2, 3)) / denom

    def _extract_cerberus_parts(
        self, patch_tokens: torch.Tensor, feature_map: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if feature_map is not None and self.use_heatmap_partition:
            return self._extract_heatmap_parts(feature_map)
        pooled_parts, valid_masks = self._extract_patch_parts(patch_tokens)
        valid_masks["gender_valid"] = torch.ones(
            patch_tokens.shape[0], dtype=torch.bool, device=patch_tokens.device
        )
        return pooled_parts, valid_masks

    def _extract_patch_parts(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        patches_per_part = max(self.num_patches // 4, 1)
        pooled_parts = {}
        valid_masks = {}
        for i, part_name in enumerate(self.part_group_names):
            start = i * patches_per_part
            end = self.num_patches if i == len(self.part_group_names) - 1 else min((i + 1) * patches_per_part, patch_tokens.shape[1])
            if end <= start:
                end = min(start + 1, patch_tokens.shape[1])
                start = max(0, end - 1)
            pooled_parts[part_name] = patch_tokens[:, start:end, :].mean(dim=1)
            valid_masks[f"{part_name}_valid"] = torch.ones(
                patch_tokens.shape[0], dtype=torch.bool, device=patch_tokens.device
            )
        return pooled_parts, valid_masks

    def _extract_heatmap_parts(
        self, feature_map: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prior_heatmap = torch.sigmoid(self.prior_heatmap_head(feature_map))
        guided_map = feature_map + self.prior_gain * (feature_map * prior_heatmap)
        guided_map = guided_map + self.region_refiner(guided_map)

        y_profile = prior_heatmap.amax(dim=3).squeeze(1)
        x_profile = prior_heatmap.amax(dim=2).squeeze(1)
        top, bottom, has_vertical = self._find_active_span(y_profile, self.heatmap_threshold)
        left, right, has_horizontal = self._find_active_span(x_profile, self.heatmap_threshold)
        valid_global = has_vertical & has_horizontal

        batch_size, _, height, width = feature_map.shape
        y_grid = torch.arange(height, device=feature_map.device).view(1, height, 1)
        x_grid = torch.arange(width, device=feature_map.device).view(1, 1, width)
        top_f = top.float()
        bottom_f = bottom.float() + 1.0
        left_f = left.float()
        right_f = right.float() + 1.0

        pooled_parts = {}
        valid_masks = {}
        for i, part_name in enumerate(self.part_group_names):
            start = top_f + (bottom_f - top_f) * (float(i) / len(self.part_group_names))
            end = top_f + (bottom_f - top_f) * (float(i + 1) / len(self.part_group_names))
            y_mask = (y_grid >= start.view(-1, 1, 1)) & (y_grid < end.view(-1, 1, 1))
            x_mask = (x_grid >= left_f.view(-1, 1, 1)) & (x_grid < right_f.view(-1, 1, 1))
            part_mask = (y_mask & x_mask).float().unsqueeze(1) * prior_heatmap
            part_mass = part_mask.sum(dim=(1, 2, 3))
            part_valid = valid_global & (part_mass > 1e-3)
            pooled_part = self._masked_pool(guided_map, part_mask)
            pooled_parts[part_name] = torch.where(
                part_valid.unsqueeze(1), pooled_part, torch.zeros_like(pooled_part)
            )
            valid_masks[f"{part_name}_valid"] = part_valid

        valid_masks["gender_valid"] = torch.ones(batch_size, dtype=torch.bool, device=feature_map.device)
        return pooled_parts, valid_masks

    def _extract_sfan_parts(
        self, feature_map: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Step 1: project to embed_dim and normalize (DiffPS)
        projected_map = self.sfan_feature_proj(feature_map)  # [B, C, H, W]
        B, C, H, W = projected_map.shape
        feat = F.normalize(projected_map, dim=1)
        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # Step 2: text queries normalize + matmul (DiffPS)
        text_embeds = F.normalize(self.sfan_text_queries, dim=1)  # [4, C]
        sim = torch.matmul(feat_flat, text_embeds.t())  # [B, HW, 4]
        sim = sim.permute(0, 2, 1).view(B, 4, H, W)  # [B, 4, H, W]

        # Step 3: softmax over parts (DiffPS, not sigmoid)
        semantic_maps = F.softmax(sim, dim=1)  # [B, 4, H, W]

        pooled_parts = {}
        valid_masks = {}
        batch_size, _, height, width = feature_map.shape
        y_grid = torch.arange(height, device=feature_map.device).view(1, height, 1)
        x_grid = torch.arange(width, device=feature_map.device).view(1, 1, width)

        for idx, part_name in enumerate(self.part_group_names):
            semantic_map = semantic_maps[:, idx : idx + 1]
            y_profile = semantic_map.amax(dim=3).squeeze(1)
            x_profile = semantic_map.amax(dim=2).squeeze(1)
            top, bottom, has_vertical = self._find_active_span(y_profile, self.sfan_threshold)
            left, right, has_horizontal = self._find_active_span(x_profile, self.sfan_threshold)

            top_f = top.float()
            bottom_f = bottom.float() + 1.0
            left_f = left.float()
            right_f = right.float() + 1.0

            y_mask = (y_grid >= top_f.view(-1, 1, 1)) & (y_grid < bottom_f.view(-1, 1, 1))
            x_mask = (x_grid >= left_f.view(-1, 1, 1)) & (x_grid < right_f.view(-1, 1, 1))
            crop_mask = (y_mask & x_mask).float().unsqueeze(1)
            map_mask = (semantic_map >= self.sfan_threshold).float()
            part_mask = crop_mask * map_mask * semantic_map
            part_mass = part_mask.sum(dim=(1, 2, 3))
            part_valid = has_vertical & has_horizontal & (part_mass > 1e-3)
            pooled_part = self._masked_pool(feature_map, part_mask)
            pooled_parts[part_name] = torch.where(
                part_valid.unsqueeze(1), pooled_part, torch.zeros_like(pooled_part)
            )
            valid_masks[f"{part_name}_valid"] = part_valid

        valid_masks["gender_valid"] = torch.ones(batch_size, dtype=torch.bool, device=feature_map.device)
        return pooled_parts, valid_masks

    def _merge_part_branches(
        self,
        primary_parts: Dict[str, torch.Tensor],
        primary_valid: Dict[str, torch.Tensor],
        aux_parts: Dict[str, torch.Tensor],
        aux_valid: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        merged_parts = {}
        merged_valid = {"gender_valid": primary_valid["gender_valid"] & aux_valid["gender_valid"]}
        weight = float(self.sfan_branch_weight)
        for part_name in self.part_group_names:
            key = f"{part_name}_valid"
            primary_mask = primary_valid[key].unsqueeze(1).float()
            aux_mask = aux_valid[key].unsqueeze(1).float()
            numer = (1.0 - weight) * primary_parts[part_name] * primary_mask + weight * aux_parts[
                part_name
            ] * aux_mask
            denom = (1.0 - weight) * primary_mask + weight * aux_mask
            merged = numer / denom.clamp_min(1e-6)
            merged_valid[key] = primary_valid[key] | aux_valid[key]
            merged_parts[part_name] = torch.where(
                merged_valid[key].unsqueeze(1), merged, torch.zeros_like(merged)
            )
        return merged_parts, merged_valid

    def _extract_region_parts(
        self, patch_tokens: torch.Tensor, feature_map: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mode = self.region_branch_mode
        if feature_map is None:
            return self._extract_cerberus_parts(patch_tokens, feature_map)

        if mode == "patch":
            pooled_parts, valid_masks = self._extract_patch_parts(patch_tokens)
            valid_masks["gender_valid"] = torch.ones(
                patch_tokens.shape[0], dtype=torch.bool, device=patch_tokens.device
            )
            return pooled_parts, valid_masks
        if mode in ("legacy", "cerberus"):
            return self._extract_cerberus_parts(patch_tokens, feature_map)
        if mode == "sfan":
            return self._extract_sfan_parts(feature_map)
        if mode == "dual":
            legacy_parts, legacy_valid = self._extract_cerberus_parts(patch_tokens, feature_map)
            sfan_parts, sfan_valid = self._extract_sfan_parts(feature_map)
            return self._merge_part_branches(legacy_parts, legacy_valid, sfan_parts, sfan_valid)

        raise ValueError(f"Unsupported region_branch_mode: {self.region_branch_mode}")

    def forward(
        self,
        visual_embs: torch.Tensor,
        patch_tokens: torch.Tensor,
        feature_map: torch.Tensor = None,
        return_valid: bool = False,
    ):
        gender_embedding = F.normalize(self.gender_bn(self.gender_fc(visual_embs)), dim=-1)
        pooled_parts, valid_masks = self._extract_region_parts(patch_tokens, feature_map)
        valid_masks["gender_valid"] = torch.ones(
            visual_embs.shape[0], dtype=torch.bool, device=visual_embs.device
        )
        if self.use_carry_group:
            valid_masks["carry_valid"] = torch.ones(
                visual_embs.shape[0], dtype=torch.bool, device=visual_embs.device
            )

        hair_embedding = F.normalize(self.hair_bn(self.hair_fc(pooled_parts["hair"])), dim=-1)
        top_embedding = F.normalize(self.top_bn(self.top_fc(pooled_parts["top"])), dim=-1)
        pants_embedding = F.normalize(self.pants_bn(self.pants_fc(pooled_parts["pants"])), dim=-1)
        shoes_embedding = F.normalize(self.shoes_bn(self.shoes_fc(pooled_parts["shoes"])), dim=-1)

        embeddings = {
            "gender_embedding": gender_embedding,
            "hair_embedding": hair_embedding,
            "top_embedding": top_embedding,
            "pants_embedding": pants_embedding,
            "shoes_embedding": shoes_embedding,
        }
        if self.use_carry_group:
            embeddings["carry_embedding"] = F.normalize(
                self.carry_bn(self.carry_fc(visual_embs)), dim=-1
            )
        if return_valid:
            return embeddings, valid_masks
        return embeddings

    def flatten_embeddings(self, embedding_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.stack_embeddings(embedding_dict).flatten(1)

    def stack_embeddings(self, embedding_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ordered = []
        for group_name in self.group_names:
            ordered.append(embedding_dict[f"{group_name}_embedding"])
        return torch.stack(ordered, dim=1)

    def stack_valid_masks(self, valid_mask_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ordered = []
        for group_name in self.group_names:
            ordered.append(valid_mask_dict[f"{group_name}_valid"])
        return torch.stack(ordered, dim=1)

    def _compose_prompt(self, prefix: str, values: List[str], suffix: str = "") -> str:
        tokens = [value for value in values if value and not self._is_unknown_label(value)]
        if not tokens:
            return ""
        body = " ".join(tokens).strip()
        parts = [prefix.strip(), body]
        if suffix:
            parts.append(suffix.strip())
        return " ".join(part for part in parts if part).strip()

    def _is_unknown_label(self, label: str) -> bool:
        normalized = self.prototype_bank._normalize_label(label).lower()
        if not normalized:
            return True
        return any(token in normalized for token in self.unknown_label_tokens)

    def _is_negative_label(self, label: str) -> bool:
        normalized = self.prototype_bank._normalize_label(label).lower()
        if not normalized:
            return False
        return any(token in normalized for token in self.negative_label_tokens)

    def _attribute_confidence(
        self,
        attribute_name: str,
        value_index: int,
        unknown_weight: Optional[float] = None,
    ) -> float:
        if value_index < 0:
            return 0.0
        label = self.prototype_bank.get_attribute_text(attribute_name, value_index)
        if self._is_unknown_label(label):
            return self.query_unknown_weight if unknown_weight is None else float(unknown_weight)
        return 1.0

    def _group_confidence(
        self,
        attr_dict: Dict[str, int],
        group_name: str,
        unknown_weight: float,
        mode: str,
    ) -> float:
        attr_scores = []
        for attr_name in self.prototype_bank.attribute_groups[group_name]:
            attr_scores.append(
                self._attribute_confidence(
                    attr_name,
                    self._safe_attr_index(attr_dict, attr_name),
                    unknown_weight=unknown_weight,
                )
            )
        if not attr_scores:
            return 0.0
        if mode == "min":
            return min(attr_scores)
        return sum(attr_scores) / len(attr_scores)

    def _accessory_prompt_token(self, attr_name: str, attr_dict: Dict[str, int]) -> str:
        label = self.prototype_bank.get_attribute_text(
            attr_name, self._safe_attr_index(attr_dict, attr_name)
        )
        normalized = self.prototype_bank._normalize_label(label).lower()
        if not normalized or self._is_unknown_label(normalized) or self._is_negative_label(normalized):
            return ""
        attr_token = attr_name.replace("_", " ").strip()
        if normalized in self.positive_generic_tokens:
            return attr_token
        if attr_token in normalized:
            return normalized
        return attr_token

    @staticmethod
    def _safe_attr_index(attr_dict: Dict[str, int], key: str) -> int:
        if not isinstance(attr_dict, dict):
            return -1
        value = attr_dict.get(key, -1)
        if value is None:
            return -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _normalize_free_text(text: str) -> str:
        normalized = str(text).lower().replace("_", " ")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _contains_text_token(text: str, token: str) -> bool:
        if not text or not token:
            return False
        token = CerberusSemanticIDBranch._normalize_free_text(token)
        if not token:
            return False
        return f" {token} " in f" {text} "

    def _group_expected_text_tokens(
        self, attr_dict: Dict[str, int], group_name: str
    ) -> List[str]:
        tokens = []
        for attr_name in self.prototype_bank.attribute_groups[group_name]:
            attr_index = self._safe_attr_index(attr_dict, attr_name)
            if attr_index < 0:
                continue
            label = self.prototype_bank.get_attribute_text(attr_name, attr_index)
            if not label or self._is_unknown_label(label):
                continue
            if self._is_negative_label(label):
                continue
            normalized_label = self.prototype_bank._normalize_label(label).lower()
            if normalized_label in self.positive_generic_tokens:
                tokens.append(attr_name.replace("_", " ").strip())
            else:
                tokens.append(normalized_label)
        return [token for token in tokens if token]

    def _query_text_group_boost(
        self,
        attr_dict: Dict[str, int],
        group_name: str,
        query_text: str,
        base_weight: float,
    ) -> float:
        if not self.query_text_boost_enabled or base_weight < 0.999:
            return 1.0

        normalized_text = self._normalize_free_text(query_text)
        if not normalized_text:
            return 1.0

        expected_tokens = self._group_expected_text_tokens(attr_dict, group_name)
        if not expected_tokens:
            return 1.0

        matched_tokens = [
            token for token in expected_tokens if self._contains_text_token(normalized_text, token)
        ]
        if not matched_tokens:
            return 1.0

        context_tokens = self.group_context_tokens.get(group_name, ())
        has_context = any(
            self._contains_text_token(normalized_text, token) for token in context_tokens
        )
        if group_name in {"top", "pants", "shoes", "hair"} and not has_context:
            return 1.0

        coverage = len(set(matched_tokens)) / max(len(set(expected_tokens)), 1)
        boost = 1.0 + self.query_text_boost * coverage
        return min(boost, self.query_text_max_weight)

    def build_group_prompts(
        self, attributes_list: List[Dict[str, int]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, torch.Tensor]]:
        prompts = {group_name: [] for group_name in self.group_names}
        valid_masks = {f"{group_name}_valid": [] for group_name in self.group_names}

        for attr_dict in attributes_list:
            gender_value = self.prototype_bank.get_attribute_text(
                "gender", self._safe_attr_index(attr_dict, "gender")
            )
            gender_prompt = ""
            if gender_value and not self._is_unknown_label(gender_value):
                gender_prompt = f"a photo of a {gender_value} person"

            hair_values = [
                self.prototype_bank.get_attribute_text("hair_length", self._safe_attr_index(attr_dict, "hair_length")),
                self.prototype_bank.get_attribute_text("hair_color", self._safe_attr_index(attr_dict, "hair_color")),
            ]
            hair_prompt = self._compose_prompt(
                "a photo of a person with", hair_values, "hair"
            )

            top_values = [
                self.prototype_bank.get_attribute_text("top_color", self._safe_attr_index(attr_dict, "top_color")),
                self.prototype_bank.get_attribute_text("top_style", self._safe_attr_index(attr_dict, "top_style")),
            ]
            top_prompt = self._compose_prompt(
                "a photo of a person wearing a", top_values
            )

            pants_values = [
                self.prototype_bank.get_attribute_text("pants_color", self._safe_attr_index(attr_dict, "pants_color")),
                self.prototype_bank.get_attribute_text("pants_style", self._safe_attr_index(attr_dict, "pants_style")),
            ]
            pants_prompt = self._compose_prompt(
                "a photo of a person wearing", pants_values
            )

            shoes_values = [
                self.prototype_bank.get_attribute_text("shoes_color", self._safe_attr_index(attr_dict, "shoes_color")),
                self.prototype_bank.get_attribute_text("shoes_style", self._safe_attr_index(attr_dict, "shoes_style")),
            ]
            shoes_prompt = self._compose_prompt(
                "a photo of a person wearing", shoes_values, "shoes"
            )
            carry_prompt = ""
            if self.use_carry_group:
                carry_values = [
                    self._accessory_prompt_token(attr_name, attr_dict)
                    for attr_name in self.prototype_bank.attribute_groups["carry"]
                ]
                carry_prompt = self._compose_prompt(
                    "a photo of a person carrying", carry_values
                )

            prompt_map = {
                "gender": gender_prompt,
                "hair": hair_prompt,
                "top": top_prompt,
                "pants": pants_prompt,
                "shoes": shoes_prompt,
            }
            if self.use_carry_group:
                prompt_map["carry"] = carry_prompt
            for group_name in self.group_names:
                prompt = prompt_map[group_name]
                prompts[group_name].append(prompt)
                valid_masks[f"{group_name}_valid"].append(len(prompt) > 0)

        tensor_masks = {
            key: torch.tensor(value, dtype=torch.bool) for key, value in valid_masks.items()
        }
        return prompts, tensor_masks

    def compute_query_group_weights(
        self,
        attributes_list: List[Dict[str, int]],
        device: torch.device,
        query_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        group_weights = []
        if query_texts is None:
            query_texts = [""] * len(attributes_list)
        for attr_dict, query_text in itertools.zip_longest(
            attributes_list, query_texts, fillvalue=""
        ):
            sample_weights = []
            for group_name in self.group_names:
                base_weight = self._group_confidence(
                    attr_dict,
                    group_name,
                    self.query_unknown_weight,
                    self.query_group_weight_mode,
                )
                sample_weights.append(
                    base_weight
                    * self._query_text_group_boost(
                        attr_dict, group_name, query_text, base_weight
                    )
                )
            group_weights.append(sample_weights)
        return torch.tensor(group_weights, dtype=torch.float32, device=device)

    def compute_train_group_weights(
        self, attributes_list: List[Dict[str, int]], device: torch.device
    ) -> torch.Tensor:
        group_weights = []
        for attr_dict in attributes_list:
            sample_weights = []
            for group_name in self.group_names:
                sample_weights.append(
                    self._group_confidence(
                        attr_dict,
                        group_name,
                        self.train_unknown_weight,
                        self.train_group_weight_mode,
                    )
                )
            group_weights.append(sample_weights)
        return torch.tensor(group_weights, dtype=torch.float32, device=device)

    def project_text_embeddings(
        self, text_feature_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        projected = {}
        for group_name in self.group_names:
            text_features = text_feature_dict[f"{group_name}_embedding"]
            fc = getattr(self, f"{group_name}_text_fc")
            bn = getattr(self, f"{group_name}_text_bn")
            projected[f"{group_name}_embedding"] = F.normalize(bn(fc(text_features)), dim=-1)
        return projected

    def encode_query_attributes_grouped(
        self, attributes_list: List[Dict[str, int]], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_embeddings = []
        batch_valid = []
        for attr_dict in attributes_list:
            per_group = []
            valid_mask = []
            for group_name in self.group_names:
                prototypes = self.prototype_bank.get_group_prototypes(group_name)
                index = self.prototype_bank.get_group_index(group_name, attr_dict)
                if index >= 0 and index < prototypes.shape[0]:
                    proto = F.normalize(prototypes[index].detach(), dim=0)
                    per_group.append(proto)
                    valid_mask.append(True)
                else:
                    per_group.append(torch.zeros(self.semantic_dim, device=device))
                    valid_mask.append(False)
            batch_embeddings.append(torch.stack(per_group, dim=0))
            batch_valid.append(torch.tensor(valid_mask, dtype=torch.bool, device=device))
        return torch.stack(batch_embeddings).to(device), torch.stack(batch_valid).to(device)

    def encode_query_attributes_flat(
        self, attributes_list: List[Dict[str, int]], device: torch.device
    ) -> torch.Tensor:
        batch_embeddings, _ = self.encode_query_attributes_grouped(attributes_list, device)
        return batch_embeddings.flatten(1)

    def encode_query_group_labels(
        self, attributes_list: List[Dict[str, int]], device: torch.device
    ) -> torch.Tensor:
        batch_labels = []
        for attr_dict in attributes_list:
            per_group = []
            for group_name in self.group_names:
                per_group.append(self.prototype_bank.get_group_index(group_name, attr_dict))
            batch_labels.append(per_group)
        return torch.tensor(batch_labels, dtype=torch.long, device=device)

    def predict_group_labels(
        self,
        embedding_dict: Dict[str, torch.Tensor],
        valid_mask_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        ordered_labels = []
        for group_name in self.group_names:
            group_embeddings = embedding_dict[f"{group_name}_embedding"]
            prototypes = F.normalize(
                self.prototype_bank.get_group_prototypes(group_name), dim=-1
            )
            logits = F.linear(F.normalize(group_embeddings, dim=-1), prototypes)
            labels = torch.argmax(logits, dim=1)
            if valid_mask_dict is not None:
                group_valid = valid_mask_dict.get(f"{group_name}_valid")
                if group_valid is not None:
                    labels = labels.masked_fill(~group_valid.bool(), -1)
            ordered_labels.append(labels)
        return torch.stack(ordered_labels, dim=1)

    def compute_text_alignment_losses(
        self,
        image_embedding_dict: Dict[str, torch.Tensor],
        text_embedding_dict: Dict[str, torch.Tensor],
        attributes_list: List[Dict[str, int]],
        image_valid_mask_dict: Dict[str, torch.Tensor] = None,
        text_valid_mask_dict: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        device = next(iter(image_embedding_dict.values())).device
        zero = (
            sum(t.sum() for t in image_embedding_dict.values())
            + sum(t.sum() for t in text_embedding_dict.values())
        ) * 0.0
        total_cls = zero
        total_align = zero

        for group_name in self.group_names:
            image_embeddings = image_embedding_dict[f"{group_name}_embedding"]
            text_embeddings = text_embedding_dict[f"{group_name}_embedding"]
            group_loss_weight = self.get_group_loss_weight(group_name)
            image_valid = None
            text_valid = None
            if image_valid_mask_dict is not None:
                image_valid = image_valid_mask_dict.get(f"{group_name}_valid")
            if text_valid_mask_dict is not None:
                text_valid = text_valid_mask_dict.get(f"{group_name}_valid")

            valid_indices = []
            target_indices = []
            target_weights = []
            for idx, attr_dict in enumerate(attributes_list):
                target = self.prototype_bank.get_group_index(group_name, attr_dict)
                image_ok = image_valid is None or bool(image_valid[idx].item())
                text_ok = text_valid is None or bool(text_valid[idx].item())
                sample_weight = self._group_confidence(
                    attr_dict,
                    group_name,
                    self.train_unknown_weight,
                    self.train_group_weight_mode,
                )
                if target >= 0 and image_ok and text_ok and sample_weight > 0:
                    valid_indices.append(idx)
                    target_indices.append(target)
                    target_weights.append(sample_weight)

            if not valid_indices:
                losses[f"loss_cerberus_cls_{group_name}"] = zero
                losses[f"loss_cerberus_align_{group_name}"] = zero
                losses[f"loss_cerberus_guidance_{group_name}"] = zero
                losses[f"loss_cerberus_relation_{group_name}"] = zero
                continue

            valid_indices_t = torch.tensor(valid_indices, device=device)
            target_indices_t = torch.tensor(target_indices, device=device)
            sample_weights_t = torch.tensor(
                target_weights, dtype=image_embeddings.dtype, device=device
            )
            weight_denom = sample_weights_t.sum().clamp_min(1e-6)
            image_valid_embeddings = F.normalize(image_embeddings[valid_indices_t], dim=-1)
            text_valid_embeddings = F.normalize(text_embeddings[valid_indices_t], dim=-1)

            unique_targets, inverse = torch.unique(
                target_indices_t, sorted=True, return_inverse=True
            )
            text_classes = []
            image_classes = []
            for target in unique_targets:
                same_class = target_indices_t == target
                text_classes.append(text_valid_embeddings[same_class].mean(dim=0))
                image_classes.append(image_valid_embeddings[same_class].mean(dim=0))
            text_classes_t = F.normalize(torch.stack(text_classes, dim=0), dim=-1)
            image_classes_t = F.normalize(torch.stack(image_classes, dim=0), dim=-1)

            logits_img = image_valid_embeddings @ text_classes_t.t() * self.logit_scale
            logits_text = text_valid_embeddings @ image_classes_t.t() * self.logit_scale
            cls_loss_img = F.cross_entropy(logits_img, inverse, reduction="none")
            cls_loss_text = F.cross_entropy(logits_text, inverse, reduction="none")
            cls_loss = 0.5 * (
                (cls_loss_img * sample_weights_t).sum() / weight_denom
                + (cls_loss_text * sample_weights_t).sum() / weight_denom
            )
            positive_text = text_classes_t[inverse]
            align_loss = (
                (1 - F.cosine_similarity(image_valid_embeddings, positive_text, dim=-1))
                * sample_weights_t
            ).sum() / weight_denom

            losses[f"loss_cerberus_cls_{group_name}"] = cls_loss * group_loss_weight
            losses[f"loss_cerberus_align_{group_name}"] = align_loss * group_loss_weight
            losses[f"loss_cerberus_guidance_{group_name}"] = zero
            losses[f"loss_cerberus_relation_{group_name}"] = zero
            total_cls = total_cls + cls_loss * group_loss_weight
            total_align = total_align + align_loss * group_loss_weight

        return losses

    def compute_losses(
        self,
        embedding_dict: Dict[str, torch.Tensor],
        attributes_list: List[Dict[str, int]],
        valid_mask_dict: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        device = next(iter(embedding_dict.values())).device
        zero = sum(t.sum() for t in embedding_dict.values()) * 0.0
        total_cls = zero
        total_align = zero
        total_guidance = zero
        total_relation = zero

        for group_name in self.group_names:
            group_embedding = embedding_dict[f"{group_name}_embedding"]
            prototypes = self.prototype_bank.get_group_prototypes(group_name)
            group_loss_weight = self.get_group_loss_weight(group_name)
            group_valid = None
            if valid_mask_dict is not None:
                group_valid = valid_mask_dict.get(f"{group_name}_valid")
                if group_valid is not None:
                    group_valid = group_valid.detach().cpu().tolist()

            valid_indices = []
            target_indices = []
            target_weights = []
            for idx, attr_dict in enumerate(attributes_list):
                target = self.prototype_bank.get_group_index(group_name, attr_dict)
                sample_weight = self._group_confidence(
                    attr_dict,
                    group_name,
                    self.train_unknown_weight,
                    self.train_group_weight_mode,
                )
                if target >= 0 and (group_valid is None or group_valid[idx]) and sample_weight > 0:
                    valid_indices.append(idx)
                    target_indices.append(target)
                    target_weights.append(sample_weight)

            if not valid_indices:
                losses[f"loss_cerberus_cls_{group_name}"] = zero
                losses[f"loss_cerberus_align_{group_name}"] = zero
                losses[f"loss_cerberus_guidance_{group_name}"] = zero
                losses[f"loss_cerberus_relation_{group_name}"] = zero
                continue

            valid_indices_t = torch.tensor(valid_indices, device=device)
            target_indices_t = torch.tensor(target_indices, device=device)
            sample_weights_t = torch.tensor(
                target_weights, dtype=group_embedding.dtype, device=device
            )
            weight_denom = sample_weights_t.sum().clamp_min(1e-6)
            valid_embeddings = group_embedding[valid_indices_t]
            target_prototypes = prototypes[target_indices_t]
            self.prototype_bank.update_group_counts(
                group_name, target_indices_t, weights=sample_weights_t
            )

            logits = F.linear(
                F.normalize(valid_embeddings, dim=-1),
                F.normalize(prototypes, dim=-1),
            ) * self.logit_scale
            cls_loss = (
                F.cross_entropy(logits, target_indices_t, reduction="none") * sample_weights_t
            ).sum() / weight_denom
            cosine_target = F.cosine_similarity(
                F.normalize(valid_embeddings, dim=-1),
                F.normalize(target_prototypes, dim=-1),
                dim=-1,
            )
            align_loss = ((1 - cosine_target) * sample_weights_t).sum() / weight_denom
            margins = self.prototype_bank.get_group_margin(
                group_name, target_indices_t, self.guidance_alpha, self.guidance_beta
            )
            guidance_loss = (
                torch.clamp(1 - margins - cosine_target, min=0) * sample_weights_t
            ).sum() / weight_denom

            proto_cos = F.normalize(prototypes, dim=-1) @ F.normalize(prototypes, dim=-1).t()
            target_affinity = self.prototype_bank.get_group_affinity(group_name).to(device)
            relation_loss = F.mse_loss(proto_cos, target_affinity)

            losses[f"loss_cerberus_cls_{group_name}"] = cls_loss * group_loss_weight
            losses[f"loss_cerberus_align_{group_name}"] = (
                align_loss * self.prototype_align_weight * group_loss_weight
            )
            losses[f"loss_cerberus_guidance_{group_name}"] = guidance_loss * group_loss_weight
            losses[f"loss_cerberus_relation_{group_name}"] = (
                relation_loss * self.relation_reg_weight * group_loss_weight
            )
            total_cls = total_cls + cls_loss * group_loss_weight
            total_align = total_align + align_loss * self.prototype_align_weight * group_loss_weight
            total_guidance = total_guidance + guidance_loss * group_loss_weight
            total_relation = total_relation + relation_loss * self.relation_reg_weight * group_loss_weight

        return losses
