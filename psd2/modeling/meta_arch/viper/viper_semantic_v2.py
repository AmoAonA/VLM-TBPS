"""
ViPer Model with Semantic Prototype Branch (Version 2)
Maps image features to semantic space using semantic prototypes

NEW FEATURES:
- Dynamic two-stage training support
- Automatic unfreezing of semantic branch at specified iteration
- Backward compatible with existing configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from typing import Dict, List, Optional, Tuple
from .viper_model import ClipSimple, ClipViper
from .base_tbps import SearchBaseTBPS
from ..build import META_ARCH_REGISTRY
from psd2.config.config import configurable
from psd2.layers import ShapeSpec
from psd2.modeling.extend.semantic_branch import SemanticBranch

from psd2.structures import Instances

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class ClipSimpleSemanticV2(ClipViper):
    """
    ClipSimple with semantic prototype branch (Version 2)

    Features:
    - Support for dynamic two-stage training
    - Automatic unfreezing of semantic branch at specified iteration
    - Backward compatible with existing configurations
    """

    @configurable
    def __init__(
            self,
            semantic_branch=None,
            use_semantic_loss=False,
            semantic_loss_weight=1.0,
            schema_path=None,
            freeze_semantic_branch=False,
            unfreeze_semantic_at_iter=0,
            semantic_train_use_infer_path=True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Semantic branch
        self.semantic_branch = semantic_branch
        self.use_semantic_loss = use_semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.schema_path = schema_path
        self.freeze_semantic_branch = freeze_semantic_branch
        self.unfreeze_semantic_at_iter = unfreeze_semantic_at_iter
        self.semantic_train_use_infer_path = semantic_train_use_infer_path
        self.semantic_branch_unfrozen = False  # 标记是否已解冻

        # 属性特征压缩层 (用于解决推理显存爆炸)
        if self.semantic_branch is not None:
            self.attr_compressor = nn.Linear(2560, 512)
            nn.init.kaiming_normal_(self.attr_compressor.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.attr_compressor.bias, 0)

        # 冻结语义分支参数（第一阶段）
        if self.freeze_semantic_branch and self.semantic_branch is not None:
            logger.info("=" * 60)
            logger.info("ClipSimpleSemanticV2: Two-stage training enabled")
            logger.info("=" * 60)
            logger.info(f"Stage 1: Semantic branch FROZEN (iteration 0-{self.unfreeze_semantic_at_iter})")
            logger.info(f"Stage 2: Semantic branch UNFROZEN (iteration {self.unfreeze_semantic_at_iter}-end)")
            logger.info("=" * 60)

            for param in self.semantic_branch.parameters():
                param.requires_grad = False
            if hasattr(self, 'attr_compressor'):
                for param in self.attr_compressor.parameters():
                    param.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        if cfg.PERSON_SEARCH.REID.get('USE_SEMANTIC_BRANCH', False):
            embed_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
            semantic_dim = cfg.PERSON_SEARCH.REID.get('SEMANTIC_DIM', 512)
            num_patches = cfg.PERSON_SEARCH.REID.get('NUM_PATCHES', 49)
            schema_path = cfg.PERSON_SEARCH.REID.get('SCHEMA_PATH', 'generated_schema.json')
            alpha = cfg.PERSON_SEARCH.REID.get('SEMANTIC_ALPHA', 0.4)
            beta = cfg.PERSON_SEARCH.REID.get('SEMANTIC_BETA', 1.8)

            semantic_branch = SemanticBranch(
                embed_dim=embed_dim,
                semantic_dim=semantic_dim,
                num_patches=num_patches,
                schema_path=schema_path,
                alpha=alpha,
                beta=beta
            )
            ret["semantic_branch"] = semantic_branch
            ret["use_semantic_loss"] = cfg.PERSON_SEARCH.REID.get('USE_SEMANTIC_LOSS', True)
            ret["semantic_loss_weight"] = cfg.PERSON_SEARCH.REID.get('SEMANTIC_LOSS_WEIGHT', 1.0)
            ret["schema_path"] = schema_path
            ret["freeze_semantic_branch"] = cfg.PERSON_SEARCH.REID.get('FREEZE_SEMANTIC_BRANCH', False)
            ret["unfreeze_semantic_at_iter"] = cfg.PERSON_SEARCH.REID.get('UNFREEZE_SEMANTIC_AT_ITER', 0)
            ret["semantic_train_use_infer_path"] = cfg.PERSON_SEARCH.REID.get(
                'SEMANTIC_TRAIN_USE_INFER_PATH', True
            )
        else:
            ret["semantic_branch"] = None
            ret["use_semantic_loss"] = False
            ret["semantic_loss_weight"] = 0.0
            ret["schema_path"] = None
            ret["freeze_semantic_branch"] = False
            ret["unfreeze_semantic_at_iter"] = 0
            ret["semantic_train_use_infer_path"] = True

        return ret

    def _build_default_text_tokens(self, batch_size=1):
        dummy_len = self.clip_model.context_length
        return torch.zeros((batch_size, dummy_len), dtype=torch.long, device=self.device)

    def _build_semantic_visual_embeddings(self, box_features):
        was_training = self.bn_neck.training if isinstance(self.bn_neck, nn.Module) else False
        try:
            if isinstance(self.bn_neck, nn.Module):
                self.bn_neck.eval()
            with torch.no_grad():
                semantic_box_embs = self.img_emb_infer(box_features)
                semantic_box_embs = self.bn_neck(semantic_box_embs)
                semantic_box_embs = F.normalize(semantic_box_embs, dim=-1)
            return semantic_box_embs
        finally:
            if isinstance(self.bn_neck, nn.Module):
                self.bn_neck.train(was_training)

    def _collect_semantic_attributes(self, assign_ids_per_img, pos_match_indices, gt_instances):
        semantic_attributes = []
        for ids_i, match_i, gt_i in zip(assign_ids_per_img, pos_match_indices, gt_instances):
            attrs_i = [{} for _ in range(len(ids_i))]
            if hasattr(gt_i, 'gt_attributes'):
                src_idxs, tgt_idxs = match_i
                src_list = src_idxs.cpu().tolist()
                tgt_list = tgt_idxs.cpu().tolist()
                for src_idx, tgt_idx in zip(src_list, tgt_list):
                    if src_idx < len(attrs_i) and tgt_idx < len(gt_i.gt_attributes):
                        attrs_i[src_idx] = gt_i.gt_attributes[tgt_idx]

            keep_mask = (ids_i > -2).cpu().tolist()
            for keep, attr in zip(keep_mask, attrs_i):
                if keep:
                    semantic_attributes.append(attr)
        return semantic_attributes

    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            # ================= 训练模式 =================
            # 检查是否需要解冻语义分支（动态两阶段训练）
            if self.freeze_semantic_branch and not self.semantic_branch_unfrozen:
                try:
                    from psd2.utils.events import get_event_storage
                    iter = get_event_storage().iter
                    if iter >= self.unfreeze_semantic_at_iter:
                        self._unfreeze_semantic_branch()
                except:
                    pass  # 如果无法获取迭代次数，保持冻结状态

            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, roi_losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )

            losses = {}
            losses.update(proposal_losses)
            losses.update(roi_losses)

            # Identity assignment
            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )

            for i, instances_i in enumerate(pred_instances):
                instances_i.assign_ids = assign_ids_per_img[i]

            assign_ids = torch.cat(assign_ids_per_img)
            semantic_attributes = self._collect_semantic_attributes(
                assign_ids_per_img, pos_match_indices, gt_instances
            )
            pos_mask = assign_ids > -2
            box_features = box_features[pos_mask]
            assign_ids = assign_ids[pos_mask]

            box_embs, assign_ids = self.img_embes(box_features, assign_ids)
            box_embs = self.bn_neck(box_embs)

            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k, v in reid_loss.items():
                losses[k] = v * 0.5

            # Text OIM loss logic
            text_tokens = []
            text_pids = []
            for img_gt in gt_instances:
                gt_ids = img_gt.gt_pids
                gt_tokens = img_gt.descriptions
                for pid, p_tokens in zip(gt_ids, gt_tokens):
                    if pid > -1:
                        num_desc = len(p_tokens)
                        text_tokens.extend(p_tokens)
                        text_pids.extend([pid] * num_desc)

            if len(text_pids) == 0:
                text_tokens_tensor = self._build_default_text_tokens()
                text_pids_tensor = torch.tensor([-2], dtype=torch.long, device=self.device)
                has_valid_text = False
            else:
                text_tokens_tensor = torch.stack(text_tokens).to(self.device)
                text_pids_tensor = torch.stack(text_pids).to(self.device)
                has_valid_text = True

            text_embs = self.text_embeds(text_tokens_tensor)
            reid_loss_text = self.oim_loss(text_embs, text_pids_tensor)

            for k, v in reid_loss_text.items():
                losses[k + "_text"] = v * 0.5

            if has_valid_text:
                lb_box_embs = box_embs[assign_ids > -1]
                lb_assign_ids = assign_ids[assign_ids > -1]
                losses["loss_sdm"] = self.compute_sdm(
                    lb_box_embs, text_embs, lb_assign_ids, text_pids_tensor, 1 / 0.02
                )
            else:
                losses["loss_sdm"] = torch.tensor(0., device=self.device)

            # ===== Semantic Loss =====
            # 动态两阶段训练：
            # - 第一阶段（冻结）：语义损失为占位符（0）
            # - 第二阶段（解冻）：计算真实语义损失
            if self.use_semantic_loss and self.semantic_branch is not None and not self.freeze_semantic_branch:
                # 始终解冻模式：计算真实语义损失
                losses.update(self._compute_semantic_losses(box_embs, box_features, semantic_attributes))
            elif self.freeze_semantic_branch:
                # 冻结模式：添加占位符损失
                if self.semantic_branch_unfrozen:
                    # 已解冻：计算真实语义损失
                    losses.update(self._compute_semantic_losses(box_embs, box_features, semantic_attributes))
                else:
                    # 仍冻结：占位符损失
                    dummy_loss = torch.tensor(0., device=self.device)
                    for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
                        losses[f'loss_semantic_{group_name}'] = dummy_loss
                    losses['loss_semantic_total'] = dummy_loss

            return pred_instances, [feat.detach() for feat in features.values()], losses

        else:
            # ================= 推理模式 =================
            features = self.backbone(image_list.tensor)
            proposals, _ = self.proposal_generator(image_list, features, None)
            pred_instances, box_features, _, _ = self.roi_heads(
                image_list, features, proposals, None
            )

            if len(pred_instances) == 0:
                return pred_instances

            # Visual Part (1024d)
            box_embs = self.img_emb_infer(box_features)
            box_embs = self.bn_neck(box_embs)
            box_embs = F.normalize(box_embs, dim=-1)

            # Semantic Part
            if self.semantic_branch is not None:
                patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)

                semantic_out = self.semantic_branch(
                    org_lembs=box_embs,
                    embs_ech=box_embs,
                    embs_rm=box_embs,
                    patch_tokens=patch_tokens
                )

                # 1. 扁平化
                semantic_feat_flat = self.semantic_branch.flatten_semantic_features(semantic_out)

                # 2. 压缩
                if hasattr(self, 'attr_compressor'):
                    semantic_feat_flat = self.attr_compressor(semantic_feat_flat)

                # 3. 归一化
                semantic_feat_flat = F.normalize(semantic_feat_flat, dim=-1)

                # 4. 拼接
                final_gallery_feats = torch.cat([box_embs, semantic_feat_flat], dim=1)
            else:
                final_gallery_feats = box_embs

            num_instances_per_img = [len(i) for i in pred_instances]
            final_gallery_feats_split = final_gallery_feats.split(num_instances_per_img)

            for i, instances in enumerate(pred_instances):
                instances.reid_feats = final_gallery_feats_split[i]

            return pred_instances

    def forward_query(self, image_list, gt_instances):
        """Query 推理"""
        text_tokens = []
        attributes_list = []

        for inst in gt_instances:
            if hasattr(inst, "descriptions") and len(inst.descriptions) > 0:
                text_tokens.append(inst.descriptions[0][0])
            else:
                text_tokens.append(self._build_default_text_tokens()[0])

            if hasattr(inst, "gt_attributes"):
                attrs = inst.gt_attributes
                if isinstance(attrs, list) and len(attrs) > 0:
                    attributes_list.append(attrs[0])
                elif isinstance(attrs, dict):
                    attributes_list.append(attrs)
                else:
                    attributes_list.append({})
            else:
                attributes_list.append({})

        # A. 文本特征
        text_input = torch.stack(text_tokens).to(self.device)
        text_embs = self.text_embeds(text_input)
        text_embs = F.normalize(text_embs, dim=-1)

        # B. 属性特征
        if self.semantic_branch is not None:
            # 1. 获取原型
            proto_embs = self.semantic_branch.encode_query_attributes_flat(attributes_list, self.device)

            # 2. 压缩
            if hasattr(self, 'attr_compressor'):
                proto_embs = self.attr_compressor(proto_embs)

            # 3. 归一化
            proto_embs = F.normalize(proto_embs, dim=-1)

            # 4. 拼接
            final_query_feats = torch.cat([text_embs, proto_embs], dim=1)
        else:
            final_query_feats = text_embs

        final_query_feats = torch.split(final_query_feats, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=final_query_feats[i])
            for i in range(len(final_query_feats))
        ]

    def _compute_semantic_losses(self, box_embs, box_features, semantic_attributes):
        """计算语义损失"""
        losses = {}

        patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)
        if self.semantic_train_use_infer_path:
            semantic_box_embs = self._build_semantic_visual_embeddings(box_features)
            org_lembs = semantic_box_embs
            embs_ech = semantic_box_embs
            embs_rm = semantic_box_embs
        else:
            num_samples = len(box_embs) // 3
            org_lembs = box_embs[:num_samples, :1024]
            embs_ech = box_embs[num_samples:2 * num_samples, :1024]
            embs_rm = box_embs[2 * num_samples:3 * num_samples, :1024]
            patch_tokens = patch_tokens[:num_samples]
            semantic_attributes = semantic_attributes[:num_samples]

        semantic_embeddings = self.semantic_branch(
            org_lembs=org_lembs,
            embs_ech=embs_ech,
            embs_rm=embs_rm,
            patch_tokens=patch_tokens
        )

        # 确保无论是否有属性，Key 都存在
        if len(semantic_attributes) > 0:
            semantic_losses = self.semantic_branch.compute_semantic_loss(
                semantic_embeddings, semantic_attributes
            )
            for key, value in semantic_losses.items():
                losses[key] = value * self.semantic_loss_weight
        else:
            # 如果当前 batch 没有有效属性，填充 0 Loss 以保持 Key 一致
            dummy_loss = sum([x.sum() for x in semantic_embeddings.values()]) * 0.0
            for group_name in ['gender', 'hair', 'top', 'pants', 'shoes']:
                losses[f'loss_semantic_{group_name}'] = dummy_loss
            losses['loss_semantic_total'] = dummy_loss

        return losses

    def _unfreeze_semantic_branch(self):
        """解冻语义分支参数"""
        if self.semantic_branch is not None and not self.semantic_branch_unfrozen:
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"ClipSimpleSemanticV2: Unfreezing semantic branch")
            logger.info("=" * 70)
            logger.info(f"Iteration: {self.unfreeze_semantic_at_iter}")
            logger.info("Transitioning from Stage 1 (Detection Only) to Stage 2 (Joint Training)")
            logger.info("=" * 70)
            logger.info("✓ Unfreezing semantic branch parameters")
            logger.info("✓ Unfreezing attribute compressor parameters")
            logger.info("✓ Now computing real semantic losses")
            logger.info("=" * 70)
            logger.info("")

            # 解冻语义分支
            for param in self.semantic_branch.parameters():
                param.requires_grad = True

            # 解冻属性压缩层
            if hasattr(self, 'attr_compressor'):
                for param in self.attr_compressor.parameters():
                    param.requires_grad = True

            self.semantic_branch_unfrozen = True


@META_ARCH_REGISTRY.register()
class ClipViperSemanticV2(ClipSimpleSemanticV2):
    """
    ViPer with semantic prototype branch (Version 2)
    Alias for ClipSimpleSemanticV2
    """
    pass
