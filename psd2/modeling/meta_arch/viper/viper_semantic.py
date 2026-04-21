"""
ViPer Model with Semantic Prototype Branch
Maps image features to semantic space using semantic prototypes
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
class ClipSimpleSemantic(ClipViper):
    """
    ClipSimple with semantic prototype branch
    """
    @configurable
    def __init__(
        self,
        semantic_branch=None,
        use_semantic_loss=False,
        semantic_loss_weight=1.0,
        schema_path=None,
        semantic_train_use_infer_path=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Semantic branch
        self.semantic_branch = semantic_branch
        self.use_semantic_loss = use_semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.schema_path = schema_path
        self.semantic_train_use_infer_path = semantic_train_use_infer_path

        # 【核心修改 1】冻结语义分支权重，仅用于测试检测部分是否正常
        # if self.semantic_branch is not None:
        #     logger.info("DEBUG MODE: Freezing Semantic Branch parameters...")
        #     for p in self.semantic_branch.parameters():
        #         p.requires_grad = False
        #
        #     # 如果还有 attr_compressor，也一并冻结（如果之前删了这里会跳过）
        #     if hasattr(self, 'attr_compressor'):
        #         for p in self.attr_compressor.parameters():
        #             p.requires_grad = False

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
            ret["semantic_train_use_infer_path"] = cfg.PERSON_SEARCH.REID.get(
                'SEMANTIC_TRAIN_USE_INFER_PATH', True
            )
        else:
            ret["semantic_branch"] = None
            ret["use_semantic_loss"] = False
            ret["semantic_loss_weight"] = 0.0
            ret["schema_path"] = None
            ret["semantic_train_use_infer_path"] = True

        return ret

    def _build_default_text_tokens(self, batch_size=1):
        dummy_len = self.clip_model.context_length
        return torch.zeros((batch_size, dummy_len), dtype=torch.long, device=self.device)

    def _build_semantic_visual_embeddings(self, box_features):
        if not self.semantic_train_use_infer_path:
            return None

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

    def _compute_semantic_losses(self, box_embs, box_features, semantic_attributes):
        if self.semantic_train_use_infer_path:
            semantic_box_embs = self._build_semantic_visual_embeddings(box_features)
            org_lembs = semantic_box_embs
            embs_ech = semantic_box_embs
            embs_rm = semantic_box_embs
            patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1).detach()
        else:
            num_samples = len(box_embs) // 3
            org_lembs = box_embs[:num_samples].detach()
            embs_ech = box_embs[num_samples:2 * num_samples].detach()
            embs_rm = box_embs[2 * num_samples:].detach()
            patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1).detach()[:num_samples]
            semantic_attributes = semantic_attributes[:num_samples]

        semantic_embeddings = self.semantic_branch(
            org_lembs=org_lembs,
            embs_ech=embs_ech,
            embs_rm=embs_rm,
            patch_tokens=patch_tokens
        )

        losses = {}
        if len(semantic_attributes) > 0:
            semantic_losses = self.semantic_branch.compute_semantic_loss(
                semantic_embeddings, semantic_attributes
            )
            for key, value in semantic_losses.items():
                losses[key] = value * self.semantic_loss_weight
        else:
            dummy_loss = sum([x.sum() for x in semantic_embeddings.values()]) * 0.0
            losses['loss_semantic_total'] = dummy_loss
            for key in [
                'loss_semantic_gender',
                'loss_semantic_hair',
                'loss_semantic_top',
                'loss_semantic_pants',
                'loss_semantic_shoes',
            ]:
                losses[key] = dummy_loss

        return losses

    # def forward_gallery(self, image_list, gt_instances): # cuhk
    #     # 1. 提取公共特征
    #     features = self.backbone(image_list.tensor)
    #
    #     # 2. 生成 Proposals
    #     proposals, proposal_losses = self.proposal_generator(
    #         image_list, features, gt_instances if self.training else None
    #     )
    #
    #     # 3. ROI Heads 提取检测结果
    #     pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
    #         image_list, features, proposals, gt_instances if self.training else None
    #     )
    #
    #     if self.training:
    #         # ========== 训练模式 (保持不变) ==========
    #         losses.update(proposal_losses)
    #
    #         # --- ID Assignment ---
    #         assign_ids_per_img = self.id_assigner(
    #             [inst.pred_boxes.tensor for inst in pred_instances],
    #             [inst.pred_scores for inst in pred_instances],
    #             [inst.gt_boxes.tensor for inst in gt_instances],
    #             [inst.gt_pids for inst in gt_instances],
    #             match_indices=pos_match_indices,
    #         )
    #
    #         for i, instances_i in enumerate(pred_instances):
    #             instances_i.assign_ids = assign_ids_per_img[i]
    #
    #         assign_ids = torch.cat(assign_ids_per_img)
    #
    #         # --- 筛选正样本 ---
    #         pos_mask = assign_ids > -2
    #         box_features = box_features[pos_mask]
    #         assign_ids = assign_ids[pos_mask]
    #
    #         # ClipViper 特征计算
    #         box_embs, assign_ids = self.img_embes(box_features, assign_ids)
    #         box_embs = self.bn_neck(box_embs)
    #
    #         # --- OIM Loss (ReID) ---
    #         reid_loss = self.oim_loss(box_embs, assign_ids)
    #         for k, v in reid_loss.items():
    #             losses[k] = v * 0.5
    #
    #         # --- Text OIM Logic ---
    #         text_tokens = []
    #         text_pids = []
    #         for img_gt in gt_instances:
    #             gt_ids = img_gt.gt_pids
    #             gt_tokens = img_gt.descriptions
    #             for pid, p_tokens in zip(gt_ids, gt_tokens):
    #                 if pid > -1:
    #                     num_desc = len(p_tokens)
    #                     text_tokens.extend(p_tokens)
    #                     text_pids.extend([pid] * num_desc)
    #
    #         if len(text_pids) == 0:
    #             dummy_len = 77
    #             text_tokens_tensor = torch.zeros((1, dummy_len), dtype=torch.long, device=self.device)
    #             text_pids_tensor = torch.tensor([-2], dtype=torch.long, device=self.device)
    #             has_valid_text = False
    #         else:
    #             text_tokens_tensor = torch.stack(text_tokens).to(self.device)
    #             text_pids_tensor = torch.stack(text_pids).to(self.device)
    #             has_valid_text = True
    #
    #         text_embs = self.text_embeds(text_tokens_tensor)
    #         reid_loss_text = self.oim_loss(text_embs, text_pids_tensor)
    #
    #         if has_valid_text:
    #             for k, v in reid_loss_text.items():
    #                 losses[k + "_text"] = v * 0.5
    #             lb_box_embs = box_embs[assign_ids > -1]
    #             lb_assign_ids = assign_ids[assign_ids > -1]
    #             losses["loss_sdm"] = self.compute_sdm(lb_box_embs, text_embs, lb_assign_ids, text_pids_tensor, 1 / 0.02)
    #         else:
    #             losses["loss_oim_text"] = text_embs.sum() * 0.0
    #             losses["loss_sdm"] = torch.tensor(0., device=self.device)
    #
    #         # Semantic 分支训练逻辑
    #         if self.use_semantic_loss and self.semantic_branch is not None:
    #             num_samples = len(box_embs) // 3
    #             org_lembs = box_embs[:num_samples].detach()
    #             embs_ech = box_embs[num_samples:2 * num_samples].detach()
    #             embs_rm = box_embs[2 * num_samples:].detach()
    #             patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1).detach()
    #
    #             semantic_embeddings = self.semantic_branch(
    #                 org_lembs=org_lembs, embs_ech=embs_ech, embs_rm=embs_rm, patch_tokens=patch_tokens
    #             )
    #
    #             org_assign_ids = assign_ids[:num_samples]
    #             gt_attributes_target = []
    #             valid_indices = []
    #             pid_to_attr_map = {}
    #             for img_gt in gt_instances:
    #                 if hasattr(img_gt, 'gt_attributes'):
    #                     pids = img_gt.gt_pids
    #                     attrs_list = img_gt.gt_attributes
    #                     for i, pid in enumerate(pids):
    #                         pid_val = pid.item()
    #                         if pid_val > -1:
    #                             pid_to_attr_map[pid_val] = attrs_list[i]
    #
    #             for i, assigned_id in enumerate(org_assign_ids):
    #                 pid_val = assigned_id.item()
    #                 if pid_val in pid_to_attr_map:
    #                     gt_attributes_target.append(pid_to_attr_map[pid_val])
    #                     valid_indices.append(i)
    #
    #             if len(gt_attributes_target) > 0:
    #                 valid_preds = {}
    #                 for k, v in semantic_embeddings.items():
    #                     if isinstance(v, torch.Tensor) and v.shape[0] == num_samples:
    #                         valid_preds[k] = v[valid_indices]
    #                     else:
    #                         valid_preds[k] = v
    #                 semantic_losses = self.semantic_branch.compute_semantic_loss(valid_preds, gt_attributes_target)
    #                 for key, value in semantic_losses.items():
    #                     losses[key] = value * self.semantic_loss_weight
    #             else:
    #                 dummy_loss = sum([x.sum() for x in semantic_embeddings.values()]) * 0.0
    #                 losses['loss_semantic_total'] = dummy_loss
    #                 keys_to_pad = ['loss_semantic_gender', 'loss_semantic_hair', 'loss_semantic_top',
    #                                'loss_semantic_pants', 'loss_semantic_shoes']
    #                 for k in keys_to_pad: losses[k] = dummy_loss
    #
    #         return pred_instances, [feat.detach() for feat in features.values()], losses
    #
    #     else:
    #         # ================= 推理模式 (修复 AttributeError 版) =================
    #         if len(pred_instances) == 0:
    #             return pred_instances
    #
    #         # 1. 拆分特征 (Batch -> List)
    #         num_boxes_per_img = [len(x) for x in pred_instances]
    #         box_features_split = box_features.split(num_boxes_per_img)
    #
    #         final_pred_instances = []
    #         final_box_features = []
    #
    #         # 导入工具
    #         from psd2.layers import batched_nms
    #         from psd2.structures import Boxes, BoxMode
    #
    #         # 获取阈值
    #         score_t = self.roi_heads.box_predictor.test_score_thresh
    #         iou_t = self.roi_heads.box_predictor.test_nms_thresh
    #         if score_t < 0.05: score_t = 0.05  # 强制过滤极低分框
    #
    #         # 2. 循环过滤每张图片
    #         for pred_i, box_feat_i, gt_i in zip(pred_instances, box_features_split, gt_instances):
    #             boxes = pred_i.pred_boxes.tensor
    #             scores = pred_i.pred_scores
    #
    #             # A. 分数过滤
    #             keep_mask = scores >= score_t
    #             boxes = boxes[keep_mask]
    #             scores = scores[keep_mask]
    #             box_feat_i = box_feat_i[keep_mask]
    #
    #             if len(boxes) == 0:
    #                 # 如果过滤后为空，添加空实例
    #                 empty_inst = pred_i[[]]
    #                 # 修复 BoxMode 和 坐标
    #                 pred_boxes_obj = Boxes(torch.empty((0, 4), device=self.device))
    #                 pred_boxes_obj.box_mode = BoxMode.XYXY_ABS
    #                 empty_inst.pred_boxes = pred_boxes_obj
    #                 final_pred_instances.append(empty_inst)
    #                 # 添加空特征占位 (0, C, H, W)
    #                 final_box_features.append(box_feat_i)
    #                 continue
    #
    #             # B. NMS 过滤
    #             cate_idx = scores.new_zeros(scores.shape[0], dtype=torch.int64)
    #             keep_nms = batched_nms(boxes, scores, cate_idx, iou_t)
    #
    #             # C. Top-K 截断
    #             # if len(keep_nms) > 100:
    #             #     keep_nms = keep_nms[:100]
    #
    #             # 应用过滤
    #             pred_i = pred_i[keep_mask][keep_nms]
    #             box_feat_i = box_feat_i[keep_nms]
    #
    #             # D. 坐标还原
    #             org_h, org_w = gt_i.org_img_size
    #             h, w = gt_i.image_size
    #             pred_boxes_tensor = pred_i.pred_boxes.tensor
    #             new_boxes_obj = Boxes(pred_boxes_tensor)
    #             new_boxes_obj.box_mode = BoxMode.XYXY_ABS
    #             new_boxes_obj.scale(org_w / w, org_h / h)
    #             pred_i.pred_boxes = new_boxes_obj
    #
    #             final_pred_instances.append(pred_i)
    #             final_box_features.append(box_feat_i)
    #
    #         # 3. 拼接特征
    #         # 【重要修复】即使全是空的，也要拼接，生成 (0, C, H, W) 的张量送入网络
    #         # 这样网络会输出 (0, D) 的特征，正好用来赋值 reid_feats
    #         batch_box_features = torch.cat(final_box_features, dim=0)
    #
    #         # ==========================================
    #         # 4. ReID & Semantic 推理
    #         # ==========================================
    #
    #         # Visual Embedding
    #         box_embs = self.img_emb_infer(batch_box_features)
    #         box_embs = self.bn_neck(box_embs)
    #         box_embs = F.normalize(box_embs, dim=-1)
    #
    #         # Semantic Embedding
    #         if self.semantic_branch is not None:
    #             patch_tokens = batch_box_features.flatten(2, 3).permute(0, 2, 1)
    #
    #             semantic_out = self.semantic_branch(
    #                 org_lembs=box_embs, embs_ech=box_embs, embs_rm=box_embs, patch_tokens=patch_tokens
    #             )
    #
    #             semantic_feat_flat = self.semantic_branch.flatten_semantic_features(semantic_out)
    #             if hasattr(self, 'attr_compressor'):
    #                 semantic_feat_flat = self.attr_compressor(semantic_feat_flat)
    #             semantic_feat_flat = F.normalize(semantic_feat_flat, dim=-1)
    #
    #             final_gallery_feats = torch.cat([box_embs, semantic_feat_flat * 0.1], dim=1)
    #         else:
    #             final_gallery_feats = box_embs
    #
    #         # 5. 将 ReID 特征填回 Instances
    #         # 使用 split 切分回每个 instance (包括空的)
    #         split_sizes = [len(inst) for inst in final_pred_instances]
    #         split_feats = final_gallery_feats.split(split_sizes)
    #
    #         for inst, feats in zip(final_pred_instances, split_feats):
    #             # 无论是否为空，都赋值 reid_feats
    #             # 如果是空的，feats 就是 shape (0, D) 的张量
    #             inst.reid_feats = feats
    #
    #             # 如果不为空且开启 CWS，则加权
    #             if len(inst) > 0 and getattr(self, 'cws', False):
    #                 inst.reid_feats = inst.reid_feats * inst.pred_scores.view(-1, 1)
    #
    #         return final_pred_instances

    def forward_gallery(self, image_list, gt_instances): # prw
        # 1. 提取公共特征
        features = self.backbone(image_list.tensor)

        # 2. 生成 Proposals
        # 注意：推理模式下 gt_instances 为 None，proposal_generator 会自动处理
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances if self.training else None
        )

        # 3. ROI Heads 提取检测结果和特征
        pred_instances, box_features, losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances if self.training else None
        )

        if self.training:
            losses.update(proposal_losses)

            # --- ID Assignment ---
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

            # --- 筛选正样本 ---
            pos_mask = assign_ids > -2
            box_features = box_features[pos_mask]
            assign_ids = assign_ids[pos_mask]

            # =========================================================
            # 【关键修复点】ClipViper 必须传入 assign_ids 并接收两个返回值
            # =========================================================
            box_embs, assign_ids = self.img_embes(box_features, assign_ids)
            box_embs = self.bn_neck(box_embs)

            # --- OIM Loss (ReID) ---
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k, v in reid_loss.items():
                losses[k] = v * 0.5

            # --- Text OIM Logic ---
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

            # 修复 DDP 死锁问题：无论是否有文本，都要跑一遍 oim_loss
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

            if has_valid_text:
                for k, v in reid_loss_text.items():
                    losses[k + "_text"] = v * 0.5

                lb_box_embs = box_embs[assign_ids > -1]
                lb_assign_ids = assign_ids[assign_ids > -1]
                losses["loss_sdm"] = self.compute_sdm(lb_box_embs, text_embs, lb_assign_ids, text_pids_tensor, 1 / 0.02)
            else:
                losses["loss_oim_text"] = text_embs.sum() * 0.0
                losses["loss_sdm"] = torch.tensor(0., device=self.device)
            if self.use_semantic_loss and self.semantic_branch is not None:
                losses.update(
                    self._compute_semantic_losses(box_embs, box_features, semantic_attributes)
                )

            return pred_instances, [feat.detach() for feat in features.values()], losses
        else:
            # ================= 推理模式 (修复版) =================
            if len(pred_instances) == 0:
                return pred_instances

            # 1. Visual Embedding
            box_embs = self.img_emb_infer(box_features)
            box_embs = self.bn_neck(box_embs)
            box_embs = F.normalize(box_embs, dim=-1)

            # 2. Semantic Embedding
            if self.semantic_branch is not None:
                patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)

                semantic_out = self.semantic_branch(
                    org_lembs=box_embs,
                    embs_ech=box_embs,
                    embs_rm=box_embs,
                    patch_tokens=patch_tokens
                )

                semantic_feat_flat = self.semantic_branch.flatten_semantic_features(semantic_out)

                if hasattr(self, 'attr_compressor'):
                    semantic_feat_flat = self.attr_compressor(semantic_feat_flat)

                semantic_feat_flat = F.normalize(semantic_feat_flat, dim=-1)

                # 加权融合 (0.1)
                final_gallery_feats = torch.cat([box_embs, semantic_feat_flat * 0.1], dim=1)
            else:
                final_gallery_feats = box_embs

            # 3. 【核心修复】NMS、分数过滤、坐标缩放
            # 将特征切分回每张图片
            reid_feats_split = torch.split(
                final_gallery_feats, [len(instances_i) for instances_i in pred_instances]
            )

            # 读取测试阈值 (从 roi_heads.box_predictor 获取)
            score_t = self.roi_heads.box_predictor.test_score_thresh
            iou_t = self.roi_heads.box_predictor.test_nms_thresh

            # 引入 batched_nms
            from psd2.layers import batched_nms
            from psd2.structures import Boxes, BoxMode

            # 遍历每张图片的预测结果进行后处理
            for pred_i, gt_i, reid_feats_i in zip(
                    pred_instances, gt_instances, reid_feats_split
            ):
                pred_boxes_t = pred_i.pred_boxes.tensor
                pred_scores = pred_i.pred_scores

                # A. 分数过滤
                filter_mask = pred_scores >= score_t
                pred_boxes_t = pred_boxes_t[filter_mask]
                pred_scores = pred_scores[filter_mask]

                # 为 NMS 准备类别索引 (通常全是 0 类，因为只有人)
                cate_idx = pred_scores.new_zeros(
                    pred_scores.shape[0], dtype=torch.int64
                )

                # B. NMS (非极大值抑制)
                keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)

                # 应用 NMS 结果
                pred_boxes_t = pred_boxes_t[keep]
                pred_scores = pred_scores[keep]

                # 重建 Instances 对象
                pred_i.pred_boxes = Boxes(pred_boxes_t, BoxMode.XYXY_ABS)
                pred_i.pred_scores = pred_scores

                # 对应的特征也要过滤
                pred_feats = reid_feats_i[filter_mask][keep]

                # CWS (Confidence Weighted Similarity) - 如果配置开启
                if getattr(self, 'cws', False):
                    pred_feats = pred_feats * pred_scores.view(-1, 1)

                pred_i.reid_feats = pred_feats

                # 过滤 pred_classes (如果有的话)
                if hasattr(pred_i, 'pred_classes'):
                    pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]

                # C. 【至关重要】坐标还原 (Scale back to original size)
                # gt_instances 在测试时包含原始图片尺寸信息
                org_h, org_w = gt_i.org_img_size
                h, w = gt_i.image_size
                pred_i.pred_boxes.scale(org_w / w, org_h / h)

            return pred_instances
    def forward_query(self, image_list, gt_instances):
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

        # A. Text
        text_input = torch.stack(text_tokens).to(self.device)
        text_embs = self.text_embeds(text_input)
        text_embs = F.normalize(text_embs, dim=-1)

        # B. Semantic
        if self.semantic_branch is not None:
            proto_embs = self.semantic_branch.encode_query_attributes_flat(attributes_list, self.device)

            if hasattr(self, 'attr_compressor'):
                proto_embs = self.attr_compressor(proto_embs)

            proto_embs = F.normalize(proto_embs, dim=-1)

            final_query_feats = torch.cat([text_embs, proto_embs * 0.1], dim=1)
        else:
            final_query_feats = text_embs

        final_query_feats = torch.split(final_query_feats, 1)
        return [
            Instances(gt_instances[i].image_size, reid_feats=final_query_feats[i])
            for i in range(len(final_query_feats))
        ]


# @META_ARCH_REGISTRY.register()
# class ClipViperSemantic(ClipSimpleSemantic):
#     pass
