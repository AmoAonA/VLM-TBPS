"""
Semantic Branch for ViPer
Maps image features to semantic space using semantic prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


class SemanticPrototypeManager(nn.Module):
    """
    Manages semantic prototypes for different attribute groups
    """
    def __init__(self, schema_path: str, alpha: float = 0.4, beta: float = 1.8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        try:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {schema_path}. Please ensure generated_schema.json exists in the project root.")

        self.attribute_groups = {
            'gender': ['gender'],
            'hair': ['hair_color', 'hair_length'],
            'top': ['top_color', 'top_style'],
            'pants': ['pants_color', 'pants_style'],
            'shoes': ['shoes_color', 'shoes_style']
        }

        self.prototype_counts = self._calculate_prototype_counts()
        self.prototype_counts_history = {}
        self._initialize_prototypes()

    def _calculate_prototype_counts(self) -> Dict[str, int]:
        counts = {}
        for group_name, attributes in self.attribute_groups.items():
            count = 1
            for attr in attributes:
                for schema_item in self.schema:
                    if schema_item['attribute'] == attr:
                        count *= len(schema_item['mapping'])
                        break
            counts[group_name] = count
        return counts

    def _initialize_prototypes(self):
        for group_name, count in self.prototype_counts.items():
            # Parameters will be automatically registered by nn.Module
            prototypes = nn.Parameter(torch.randn(count, 512) * 0.01)
            setattr(self, f'prototypes_{group_name}', prototypes)
            self.prototype_counts_history[group_name] = torch.zeros(count)

    def get_prototype_index(self, group_name: str, attributes: Dict[str, int]) -> int:
        """
        Correctly calculate the flattened index for multi-dimensional attributes.
        Logic: index = index * dim_of_current_attr + value_of_current_attr
        """
        group_attrs = self.attribute_groups[group_name]
        if not attributes:
            return -1
        index = 0

        for attr in group_attrs:
            if attr not in attributes or attributes[attr] is None or attributes[attr] < 0:
                return -1
            # 1. Find dimension (count) of CURRENT attribute
            count = 1
            for schema_item in self.schema:
                if schema_item['attribute'] == attr:
                    count = len(schema_item['mapping'])
                    break

            # 2. Accumulate index (Row-Major Order)
            index *= count

            # 3. Add current attribute value
            if attr in attributes:
                val = attributes[attr]
                # Safety check to prevent index out of bounds if data is noisy
                if val >= count:
                    print(f"Warning: Attribute {attr} value {val} >= count {count}. Clamping.")
                    val = count - 1
                index += val

        return index

    def update_prototype_counts(self, group_name: str, prototype_indices: List[int]):
        for idx in prototype_indices:
            # Safety check
            if idx < len(self.prototype_counts_history[group_name]):
                self.prototype_counts_history[group_name][idx] += 1

    def get_margin(self, group_name: str, prototype_idx: int, total_samples: int) -> float:
        # Safety check
        if prototype_idx >= len(self.prototype_counts_history[group_name]):
            return 0.0

        n_g = self.prototype_counts_history[group_name][prototype_idx].item()

        # 【修复核心】计算该组属性的历史总样本数，而不是使用传入的 batch size
        total_history_count = self.prototype_counts_history[group_name].sum().item()

        # 计算相对频率 (0.0 ~ 1.0)
        frequency = n_g / (total_history_count + 1e-8)

        # 使用频率计算 Margin
        margin = np.log(self.alpha * frequency + self.beta)
        return margin

class SemanticBranch(nn.Module):
    """
    Semantic branch for mapping image features to semantic space
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        feature_dim: int = 2048,
        semantic_dim: int = 512,
        num_patches: int = 32, # Ensure this matches your config (8x4=32)
        schema_path: str = 'generated_schema.json',
        alpha: float = 0.4,
        beta: float = 1.8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.num_patches = num_patches

        # Prototype Manager (nn.Module)
        self.prototype_manager = SemanticPrototypeManager(schema_path, alpha, beta)

        self.gender_fc = nn.Linear(embed_dim, semantic_dim)
        self.gender_bn = nn.LayerNorm(semantic_dim)

        # Calculate split sizes
        patches_per_part = num_patches // 4

        # Define FC layers
        self.hair_fc = nn.Linear(feature_dim * patches_per_part, semantic_dim)
        self.hair_bn = nn.LayerNorm(semantic_dim)
        self.top_fc = nn.Linear(feature_dim * patches_per_part, semantic_dim)
        self.top_bn = nn.LayerNorm(semantic_dim)
        self.pants_fc = nn.Linear(feature_dim * patches_per_part, semantic_dim)
        self.pants_bn = nn.LayerNorm(semantic_dim)
        self.shoes_fc = nn.Linear(feature_dim * (num_patches - 3 * patches_per_part), semantic_dim)
        self.shoes_bn = nn.LayerNorm(semantic_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, org_lembs: torch.Tensor, embs_ech: torch.Tensor,
                embs_rm: torch.Tensor, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:

        aggregated_features = org_lembs + embs_ech + embs_rm

        gender_emb = self.gender_fc(aggregated_features)
        gender_emb = self.gender_bn(gender_emb)
        gender_emb = F.normalize(gender_emb, dim=-1)

        patches_per_part = self.num_patches // 4

        hair_patches = patch_tokens[:, :patches_per_part, :].flatten(1)
        hair_emb = F.normalize(self.hair_bn(self.hair_fc(hair_patches)), dim=-1)

        top_patches = patch_tokens[:, patches_per_part:2*patches_per_part, :].flatten(1)
        top_emb = F.normalize(self.top_bn(self.top_fc(top_patches)), dim=-1)

        pants_patches = patch_tokens[:, 2*patches_per_part:3*patches_per_part, :].flatten(1)
        pants_emb = F.normalize(self.pants_bn(self.pants_fc(pants_patches)), dim=-1)

        shoes_patches = patch_tokens[:, 3*patches_per_part:, :].flatten(1)
        shoes_emb = F.normalize(self.shoes_bn(self.shoes_fc(shoes_patches)), dim=-1)

        return {
            'gender_embedding': gender_emb,
            'hair_embedding': hair_emb,
            'top_embedding': top_emb,
            'pants_embedding': pants_emb,
            'shoes_embedding': shoes_emb
        }

    # def compute_semantic_loss(self, semantic_embeddings: Dict[str, torch.Tensor], # CE_loss
    #                           attributes: List[Dict[str, int]]) -> Dict[str, torch.Tensor]:
    #     losses = {}
    #     # 获取设备信息
    #     device = list(semantic_embeddings.values())[0].device
    #     total_loss = torch.tensor(0.0, device=device)
    #
    #     for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
    #         emb_key = f'{group_name}_embedding'
    #         if emb_key not in semantic_embeddings:
    #             continue
    #
    #         group_emb = semantic_embeddings[emb_key]  # Shape: [Batch, 512]
    #         prototypes = getattr(self.prototype_manager, f'prototypes_{group_name}')  # Shape: [Num_Classes, 512]
    #
    #         # 1. 计算索引并筛选有效样本 (过滤 -1)
    #         valid_indices = []  # 在 batch 中的位置 (0, 3, 5...)
    #         target_proto_idxs = []  # 对应的原型索引 (类别 ID)
    #
    #         for i, attr_dict in enumerate(attributes):
    #             idx = self.prototype_manager.get_prototype_index(group_name, attr_dict)
    #             # 只保留有效的索引 (>=0)
    #             if idx >= 0:
    #                 valid_indices.append(i)
    #                 target_proto_idxs.append(idx)
    #
    #         # 如果该 batch 中没有有效样本，跳过该组
    #         if not valid_indices:
    #             losses[f'loss_semantic_{group_name}'] = torch.tensor(0.0, device=device)
    #             continue
    #
    #         # 转换为 Tensor
    #         valid_indices = torch.tensor(valid_indices, device=device)
    #         target_proto_idxs = torch.tensor(target_proto_idxs, device=device)
    #
    #         # 【修复报错的关键行】: 提取有效样本的特征
    #         valid_emb = group_emb[valid_indices]
    #
    #         # 2. 更新统计计数 (可选，用于观察数据分布)
    #         self.prototype_manager.update_prototype_counts(group_name, target_proto_idxs.tolist())
    #
    #         # 3. 计算 Logits (Cross Entropy 方案)
    #         # 使用 Temperature = 30.0 来放大差异，帮助收敛
    #         # F.linear(input, weight) -> input * weight^T
    #         logits = F.linear(F.normalize(valid_emb, dim=-1), F.normalize(prototypes, dim=-1)) * 30.0
    #
    #         # 4. 计算分类损失
    #         loss_func = nn.CrossEntropyLoss()
    #         group_loss = loss_func(logits, target_proto_idxs)
    #
    #         losses[f'loss_semantic_{group_name}'] = group_loss
    #         total_loss += group_loss
    #
    #     losses['loss_semantic_total'] = total_loss
    #     return losses

    def compute_semantic_loss(self, semantic_embeddings: Dict[str, torch.Tensor], # margin_loss
                           attributes: List[Dict[str, int]]) -> Dict[str, torch.Tensor]:
        losses = {}
        # Ensure total_loss is a Tensor on the correct device
        device = list(semantic_embeddings.values())[0].device
        total_loss = torch.tensor(0.0, device=device)

        for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
            emb_key = f'{group_name}_embedding'
            if emb_key not in semantic_embeddings:
                continue

            group_emb = semantic_embeddings[emb_key]
            prototypes = getattr(self.prototype_manager, f'prototypes_{group_name}')

            valid_indices = []
            prototype_indices = []
            for i, attr_dict in enumerate(attributes):
                idx = self.prototype_manager.get_prototype_index(group_name, attr_dict)
                if idx >= 0:
                    valid_indices.append(i)
                    prototype_indices.append(idx)

            if not valid_indices:
                losses[f'loss_semantic_{group_name}'] = torch.tensor(0.0, device=device)
                continue

            valid_indices = torch.tensor(valid_indices, device=group_emb.device)
            prototype_indices = torch.tensor(prototype_indices, device=group_emb.device)
            group_emb = group_emb[valid_indices]

            # Update counts (on CPU history)
            self.prototype_manager.update_prototype_counts(group_name, prototype_indices.tolist())

            similarities = F.cosine_similarity(
                group_emb.unsqueeze(1),
                prototypes.unsqueeze(0),
                dim=-1
            )

            # Gather similarity for the target class
            correct_similarities = similarities[
                torch.arange(len(prototype_indices), device=group_emb.device), prototype_indices
            ]

            total_samples = len(attributes)
            margins = []
            for idx in prototype_indices:
                margin = self.prototype_manager.get_margin(group_name, idx.item(), total_samples)
                margins.append(margin)
            margins = torch.tensor(margins, device=group_emb.device)

            group_loss = torch.clamp(1 - margins - correct_similarities, min=0)
            group_loss = group_loss.mean()

            losses[f'loss_semantic_{group_name}'] = group_loss
            total_loss += group_loss

        losses['loss_semantic_total'] = total_loss
        return losses

    def get_prototypes(self) -> Dict[str, torch.Tensor]:
        prototypes = {}
        for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
            prototypes[group_name] = getattr(self.prototype_manager, f'prototypes_{group_name}')
        return prototypes

        # 1. 用于 Query：将离散属性字典 -> 拼接后的属性原型向量
    def encode_query_attributes_flat(self, attributes_list: List[Dict[str, int]], device) -> torch.Tensor:
            """
            返回: [Batch_Size, 5 * 512] 的 Tensor
            """
            batch_attr_embs = []
            for i, attr_dict in enumerate(attributes_list):
                sample_embs = []
                for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
                    prototypes = getattr(self.prototype_manager, f'prototypes_{group_name}')
                    idx = self.prototype_manager.get_prototype_index(group_name, attr_dict)

                    if idx >= 0 and idx < prototypes.size(0):
                        # 取出对应的原型
                        emb = prototypes[idx].detach()
                    else:
                        # 无效属性用零向量填充
                        emb = torch.zeros(self.semantic_dim, device=device)
                    sample_embs.append(emb)

                # 拼接 5 个属性的原型: [5 * 512]
                batch_attr_embs.append(torch.cat(sample_embs, dim=0))

            return torch.stack(batch_attr_embs).to(device)

        # 2. 用于 Gallery：将预测的语义特征字典 -> 拼接后的特征向量
    def flatten_semantic_features(self, semantic_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            """
            输入: {'gender_embedding': [B, 512], ...}
            输出: [B, 5 * 512]
            """
            # 确保按照固定的顺序拼接
            embs = []
            for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
                key = f'{group_name}_embedding'
                if key in semantic_dict:
                    embs.append(semantic_dict[key])
                else:
                    # 理论上 forward 应该返回所有 key，这里做个防护
                    # 获取 batch size
                    b = list(semantic_dict.values())[0].size(0)
                    device = list(semantic_dict.values())[0].device
                    embs.append(torch.zeros(b, self.semantic_dim, device=device))

            # [B, 2560]
            return torch.cat(embs, dim=1)
