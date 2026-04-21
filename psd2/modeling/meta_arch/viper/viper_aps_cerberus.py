import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from psd2.config.config import configurable
from psd2.modeling.extend.cerberus_aps_branch import CerberusAPSBranch
from psd2.structures import Instances
from psd2.utils.simple_tokenizer import SimpleTokenizer

from ..build import META_ARCH_REGISTRY
from .viper_model import ClipViper

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class ClipSimpleAPSCerberusPS(ClipViper):
    @configurable
    def __init__(
        self,
        aps_branch=None,
        aps_loss_weight=1.0,
        aps_group_score_weight=1.0,
        aps_detach_visual_inputs=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aps_branch = aps_branch
        self.aps_loss_weight = aps_loss_weight
        self.aps_group_score_weight = aps_group_score_weight
        self.aps_detach_visual_inputs = aps_detach_visual_inputs
        self.attribute_tokenizer = SimpleTokenizer()
        self.attribute_token_cache = {}
        self._freeze_unused_identity_modules()

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        branch_enabled = cfg.PERSON_SEARCH.REID.get(
            "USE_APS_CERBERUS_BRANCH",
            cfg.PERSON_SEARCH.REID.get("USE_CERBERUS_BRANCH", False),
        )
        if not branch_enabled:
            ret["aps_branch"] = None
            ret["aps_loss_weight"] = 0.0
            ret["aps_group_score_weight"] = 0.0
            ret["aps_detach_visual_inputs"] = False
            return ret

        semantic_dim = cfg.PERSON_SEARCH.REID.get("CERBERUS_SEMANTIC_DIM", 512)
        num_patches = cfg.PERSON_SEARCH.REID.get("NUM_PATCHES", 32)
        schema_path = cfg.PERSON_SEARCH.REID.get("SCHEMA_PATH", "generated_schema.json")
        logit_scale = cfg.PERSON_SEARCH.REID.get("CERBERUS_LOGIT_SCALE", 30.0)
        prototype_align_weight = cfg.PERSON_SEARCH.REID.get(
            "CERBERUS_PROTOTYPE_ALIGN_WEIGHT", 0.25
        )
        guidance_alpha = cfg.PERSON_SEARCH.REID.get("CERBERUS_GUIDANCE_ALPHA", 0.4)
        guidance_beta = cfg.PERSON_SEARCH.REID.get("CERBERUS_GUIDANCE_BETA", 1.8)
        relation_reg_weight = cfg.PERSON_SEARCH.REID.get("CERBERUS_RELATION_REG_WEIGHT", 0.05)
        use_heatmap_partition = cfg.PERSON_SEARCH.REID.get("CERBERUS_USE_HEATMAP_PARTITION", True)
        heatmap_threshold = cfg.PERSON_SEARCH.REID.get("CERBERUS_HEATMAP_THRESHOLD", 0.35)
        prior_gain = cfg.PERSON_SEARCH.REID.get("CERBERUS_PRIOR_GAIN", 1.0)
        region_branch_mode = cfg.PERSON_SEARCH.REID.get("CERBERUS_REGION_BRANCH_MODE", "sfan")
        sfan_threshold = cfg.PERSON_SEARCH.REID.get("CERBERUS_SFAN_THRESHOLD", 0.35)
        sfan_temperature = cfg.PERSON_SEARCH.REID.get("CERBERUS_SFAN_TEMPERATURE", 0.07)
        sfan_branch_weight = cfg.PERSON_SEARCH.REID.get("CERBERUS_SFAN_BRANCH_WEIGHT", 0.5)

        ret["aps_branch"] = CerberusAPSBranch(
            embed_dim=cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM,
            semantic_dim=semantic_dim,
            num_patches=num_patches,
            schema_path=schema_path,
            logit_scale=logit_scale,
            prototype_align_weight=prototype_align_weight,
            guidance_alpha=guidance_alpha,
            guidance_beta=guidance_beta,
            relation_reg_weight=relation_reg_weight,
            use_heatmap_partition=use_heatmap_partition,
            heatmap_threshold=heatmap_threshold,
            prior_gain=prior_gain,
            region_branch_mode=region_branch_mode,
            sfan_threshold=sfan_threshold,
            sfan_temperature=sfan_temperature,
            sfan_branch_weight=sfan_branch_weight,
        )
        ret["aps_loss_weight"] = cfg.PERSON_SEARCH.REID.get(
            "APS_CERBERUS_LOSS_WEIGHT",
            cfg.PERSON_SEARCH.REID.get("CERBERUS_LOSS_WEIGHT", 1.0),
        )
        ret["aps_group_score_weight"] = cfg.PERSON_SEARCH.REID.get(
            "APS_CERBERUS_GROUP_SCORE_WEIGHT",
            cfg.PERSON_SEARCH.REID.get("CERBERUS_GROUP_SCORE_WEIGHT", 1.0),
        )
        ret["aps_detach_visual_inputs"] = cfg.PERSON_SEARCH.REID.get(
            "APS_CERBERUS_DETACH_VISUAL_INPUTS",
            cfg.PERSON_SEARCH.REID.get("CERBERUS_DETACH_VISUAL_INPUTS", False),
        )
        return ret

    def _freeze_unused_identity_modules(self):
        for module_name in ("bn_neck_text", "cross_modal_transformer", "ln_post", "mim_norm", "mim_head"):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False

        if hasattr(self, "mim_token"):
            self.mim_token.requires_grad = False

        for module_name in ("token_embedding", "transformer", "ln_final"):
            module = getattr(self.clip_model, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False

        for parameter_name in ("positional_embedding", "text_projection", "logit_scale"):
            parameter = getattr(self.clip_model, parameter_name, None)
            if isinstance(parameter, torch.Tensor):
                parameter.requires_grad = False

    def _build_default_text_tokens(self, batch_size=1):
        return torch.zeros(
            (batch_size, self.clip_model.context_length), dtype=torch.long, device=self.device
        )

    def _tokenize_attribute_prompt(self, caption: str) -> torch.LongTensor:
        if not caption:
            return self._build_default_text_tokens()[0].cpu()
        if caption in self.attribute_token_cache:
            return self.attribute_token_cache[caption].clone()

        sot_token = self.attribute_tokenizer.encoder["<|startoftext|>"]
        eot_token = self.attribute_tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.attribute_tokenizer.encode(caption) + [eot_token]
        text_length = self.clip_model.context_length
        result = torch.zeros(text_length, dtype=torch.long)
        if len(tokens) > text_length:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        result[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        self.attribute_token_cache[caption] = result.clone()
        return result

    def _encode_clip_text_features(self, text_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            text_feats = self.clip_model.encode_text_zero_dropout(text_tokens, ckpt=False)
            text_feats = text_feats[torch.arange(text_feats.shape[0]), text_tokens.argmax(dim=-1)]
        return F.normalize(text_feats, dim=-1)

    def _collect_semantic_attributes(self, assign_ids_per_img, pos_match_indices, gt_instances):
        semantic_attributes = []
        for ids_i, match_i, gt_i in zip(assign_ids_per_img, pos_match_indices, gt_instances):
            attrs_i = [{} for _ in range(len(ids_i))]
            if hasattr(gt_i, "gt_attributes"):
                src_idxs, tgt_idxs = match_i
                for src_idx, tgt_idx in zip(src_idxs.cpu().tolist(), tgt_idxs.cpu().tolist()):
                    if src_idx < len(attrs_i) and tgt_idx < len(gt_i.gt_attributes):
                        attrs_i[src_idx] = gt_i.gt_attributes[tgt_idx]

            keep_mask = (ids_i > -2).cpu().tolist()
            for keep, attr in zip(keep_mask, attrs_i):
                if keep:
                    semantic_attributes.append(attr)
        return semantic_attributes

    def _build_aps_visual_embeddings(self, box_features):
        visual_embs = self.img_emb_infer(box_features)
        visual_embs = self.bn_neck(visual_embs)
        visual_embs = F.normalize(visual_embs, dim=-1)
        return visual_embs

    def _compute_aps_losses(self, box_features, semantic_attributes):
        device = box_features.device
        if self.aps_branch is None:
            return {}
        if box_features.numel() == 0 or not semantic_attributes:
            return self.aps_branch.empty_aps_losses(device)

        feature_map = box_features.detach() if self.aps_detach_visual_inputs else box_features
        patch_tokens = feature_map.flatten(2, 3).permute(0, 2, 1)
        visual_embs = self._build_aps_visual_embeddings(feature_map)
        if self.aps_detach_visual_inputs:
            visual_embs = visual_embs.detach()

        aps_embeddings, aps_valid_masks = self.aps_branch(
            visual_embs,
            patch_tokens,
            feature_map=feature_map,
            return_valid=True,
        )
        raw_losses = self.aps_branch.compute_aps_losses(
            aps_embeddings,
            semantic_attributes,
            tokenize_fn=self._tokenize_attribute_prompt,
            encode_text_fn=self._encode_clip_text_features,
            device=device,
            image_valid_mask_dict=aps_valid_masks,
        )
        return {key: value * self.aps_loss_weight for key, value in raw_losses.items()}

    def _build_gallery_aps_features(self, box_features):
        visual_embs = self._build_aps_visual_embeddings(box_features)
        patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)
        aps_embeddings, aps_valid_masks = self.aps_branch(
            visual_embs,
            patch_tokens,
            feature_map=box_features,
            return_valid=True,
        )
        group_feats = self.aps_branch.stack_embeddings(aps_embeddings)
        group_valid = self.aps_branch.stack_valid_masks(aps_valid_masks)
        group_feats = group_feats * group_valid.unsqueeze(-1).to(group_feats.dtype)
        flat_feats = group_feats.flatten(1)
        return flat_feats, group_feats, group_valid

    def forward_gallery(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances if self.training else None
        )
        pred_instances, box_features, roi_losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances if self.training else None
        )

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(roi_losses)

            assign_ids_per_img = self.id_assigner(
                [inst.pred_boxes.tensor for inst in pred_instances],
                [inst.pred_scores for inst in pred_instances],
                [inst.gt_boxes.tensor for inst in gt_instances],
                [inst.gt_pids for inst in gt_instances],
                match_indices=pos_match_indices,
            )
            for instances_i, ids_i in zip(pred_instances, assign_ids_per_img):
                instances_i.assign_ids = ids_i

            assign_ids = torch.cat(assign_ids_per_img)
            semantic_attributes = self._collect_semantic_attributes(
                assign_ids_per_img, pos_match_indices, gt_instances
            )
            pos_mask = assign_ids > -2
            box_features = box_features[pos_mask]
            losses.update(self._compute_aps_losses(box_features, semantic_attributes))
            return pred_instances, [feat.detach() for feat in features.values()], losses

        if len(pred_instances) == 0:
            return pred_instances

        aps_flat_feats, aps_group_feats, aps_group_valid = self._build_gallery_aps_features(box_features)
        split_sizes = [len(inst) for inst in pred_instances]
        aps_flat_feats_split = torch.split(aps_flat_feats, split_sizes)
        aps_group_feats_split = torch.split(aps_group_feats, split_sizes)
        aps_group_valid_split = torch.split(aps_group_valid, split_sizes)

        score_t = self.roi_heads.box_predictor.test_score_thresh
        iou_t = self.roi_heads.box_predictor.test_nms_thresh
        from psd2.layers import batched_nms
        from psd2.structures import Boxes, BoxMode

        for pred_i, gt_i, flat_i, group_i, valid_i in zip(
            pred_instances,
            gt_instances,
            aps_flat_feats_split,
            aps_group_feats_split,
            aps_group_valid_split,
        ):
            pred_boxes_t = pred_i.pred_boxes.tensor
            pred_scores = pred_i.pred_scores
            filter_mask = pred_scores >= score_t
            pred_boxes_t = pred_boxes_t[filter_mask]
            pred_scores = pred_scores[filter_mask]
            cate_idx = pred_scores.new_zeros(pred_scores.shape[0], dtype=torch.int64)
            keep = batched_nms(pred_boxes_t, pred_scores, cate_idx, iou_t)
            pred_i.pred_boxes = Boxes(pred_boxes_t[keep], BoxMode.XYXY_ABS)
            pred_i.pred_scores = pred_scores[keep]

            pred_flat_feats = flat_i[filter_mask][keep]
            pred_group_feats = group_i[filter_mask][keep]
            pred_group_valid = valid_i[filter_mask][keep]
            if getattr(self, "cws", False):
                pred_flat_feats = pred_flat_feats * pred_scores[keep].view(-1, 1)
                pred_group_feats = pred_group_feats * pred_scores[keep].view(-1, 1, 1)

            # Keep reid_feats for compatibility with the existing evaluator stack.
            pred_i.reid_feats = pred_flat_feats
            pred_i.aps_group_feats = pred_group_feats
            pred_i.aps_group_valid = pred_group_valid
            pred_i.aps_group_score_weight = pred_scores[keep].new_full(
                (pred_group_feats.shape[0],), self.aps_group_score_weight
            )
            pred_i.cerberus_group_feats = pred_group_feats
            pred_i.cerberus_group_valid = pred_group_valid
            pred_i.cerberus_group_score_weight = pred_scores[keep].new_full(
                (pred_group_feats.shape[0],), self.aps_group_score_weight
            )

            if hasattr(pred_i, "pred_classes"):
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            pred_i.pred_boxes.scale(org_w / w, org_h / h)
        return pred_instances

    def forward_query(self, image_list, gt_instances):
        attributes_list = []
        for inst in gt_instances:
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

        flat_feats, group_feats, group_valid = self.aps_branch.encode_query_from_attributes(
            attributes_list,
            tokenize_fn=self._tokenize_attribute_prompt,
            encode_text_fn=self._encode_clip_text_features,
            device=self.device,
        )
        flat_feats_split = torch.split(flat_feats, 1)
        group_feats_split = torch.split(group_feats, 1)
        group_valid_split = torch.split(group_valid, 1)

        outputs = []
        for i, flat_feat in enumerate(flat_feats_split):
            inst_kwargs = {
                "reid_feats": flat_feat,
                "aps_group_feats": group_feats_split[i],
                "aps_group_valid": group_valid_split[i],
                "aps_group_score_weight": flat_feat.new_full((1,), self.aps_group_score_weight),
                "cerberus_group_feats": group_feats_split[i],
                "cerberus_group_valid": group_valid_split[i],
                "cerberus_group_score_weight": flat_feat.new_full((1,), self.aps_group_score_weight),
            }
            outputs.append(Instances(gt_instances[i].image_size, **inst_kwargs))
        return outputs


@META_ARCH_REGISTRY.register()
class ClipViperAPSCerberusPS(ClipSimpleAPSCerberusPS):
    pass
