"""
Additional Cerberus-inspired reproduction path for person search.

This implementation is intentionally isolated from the existing semantic branch.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from psd2.config.config import configurable
from psd2.utils.simple_tokenizer import SimpleTokenizer
from psd2.modeling.extend.cerberus_semantic_id import CerberusSemanticIDBranch
# from psd2.modeling.extend.cerberus_memory_bank import CerberusMemoryBank
# from psd2.modeling.extend.cerberus_gada_loss import CerberusGADALoss
# from psd2.modeling.extend.cerberus_brda_loss import CerberusBRDALoss
from psd2.structures import Boxes, Instances

from ..build import META_ARCH_REGISTRY
from .viper_model import ClipViper

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class ClipSimpleCerberusPS(ClipViper):
    @configurable
    def __init__(
        self,
        cerberus_branch=None,
        use_cerberus_loss=False,
        cerberus_loss_weight=1.0,
        cerberus_group_score_weight=1.0,
        freeze_cerberus_branch=False,
        unfreeze_cerberus_at_iter=0,
        cerberus_train_use_infer_path=True,
        cerberus_detach_visual_inputs=False,
        cerberus_train_branch_only=False,
        cerberus_warmup_iters=0,
        cerberus_stage1_baseline_iters=0,
        cerberus_stage2_freeze_non_branch=False,
        cerberus_stage2_branch_only=False,
        cerberus_stage2_detach_visual_inputs=False,
        memory_bank=None,
        gada_module=None,
        brda_module=None,
        memory_loss_weight=0.5,
        gada_loss_weight=0.5,
        brda_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cerberus_branch = cerberus_branch
        self.use_cerberus_loss = use_cerberus_loss
        self.cerberus_loss_weight = cerberus_loss_weight
        self.cerberus_group_score_weight = cerberus_group_score_weight
        self.freeze_cerberus_branch = freeze_cerberus_branch
        self.unfreeze_cerberus_at_iter = unfreeze_cerberus_at_iter
        self.cerberus_train_use_infer_path = cerberus_train_use_infer_path
        self.cerberus_detach_visual_inputs = cerberus_detach_visual_inputs
        self.cerberus_train_branch_only = cerberus_train_branch_only
        self.cerberus_warmup_iters = cerberus_warmup_iters
        self.cerberus_stage1_baseline_iters = cerberus_stage1_baseline_iters
        self.cerberus_stage2_freeze_non_branch = cerberus_stage2_freeze_non_branch
        self.cerberus_stage2_branch_only = cerberus_stage2_branch_only
        self.cerberus_stage2_detach_visual_inputs = cerberus_stage2_detach_visual_inputs
        self.cerberus_branch_unfrozen = False
        self.cerberus_stage2_activated = False
        self.memory_bank = memory_bank
        self.gada_module = gada_module
        self.brda_module = brda_module
        self.memory_loss_weight = memory_loss_weight
        self.gada_loss_weight = gada_loss_weight
        self.brda_loss_weight = brda_loss_weight
        self.attribute_tokenizer = SimpleTokenizer()
        self.attribute_token_cache = {}
        self._frozen_module_mode_logged = False

        if self.freeze_cerberus_branch and self.cerberus_branch is not None:
            for parameter in self.cerberus_branch.parameters():
                parameter.requires_grad = False

        if (
            self.cerberus_branch is not None
            and self.cerberus_branch.region_branch_mode in ("sfan", "dual")
        ):
            self.cerberus_branch.init_sfan_text_queries(
                self.clip_model, self.attribute_tokenizer,
                context_length=self.clip_model.context_length,
            )

        # Initialize prototypes with CLIP text encoder
        if self.cerberus_branch is not None and hasattr(
            self.cerberus_branch, "init_prototypes_with_clip"
        ):
            logger.info("Initializing Cerberus prototypes with CLIP text encoder...")
            self.cerberus_branch.init_prototypes_with_clip(
                self.clip_model, self.attribute_tokenizer,
                context_length=self.clip_model.context_length,
            )
            logger.info("Cerberus prototypes initialized with CLIP.")

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        if cfg.PERSON_SEARCH.REID.get("USE_CERBERUS_BRANCH", False):
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
            query_unknown_weight = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_QUERY_UNKNOWN_WEIGHT", 0.35
            )
            query_group_weight_mode = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_QUERY_GROUP_WEIGHT_MODE", "avg"
            )
            query_text_boost_enabled = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_QUERY_TEXT_BOOST_ENABLED", True
            )
            query_text_boost = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_QUERY_TEXT_BOOST", 0.20
            )
            query_text_max_weight = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_QUERY_TEXT_MAX_WEIGHT", 1.35
            )
            train_unknown_weight = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_TRAIN_UNKNOWN_WEIGHT", 1.0
            )
            train_group_weight_mode = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_TRAIN_GROUP_WEIGHT_MODE", "avg"
            )
            group_loss_weights = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_GROUP_LOSS_WEIGHTS", []
            )
            use_carry_group = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_USE_CARRY_GROUP", False
            )

            ret["cerberus_branch"] = CerberusSemanticIDBranch(
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
                query_unknown_weight=query_unknown_weight,
                query_group_weight_mode=query_group_weight_mode,
                query_text_boost_enabled=query_text_boost_enabled,
                query_text_boost=query_text_boost,
                query_text_max_weight=query_text_max_weight,
                train_unknown_weight=train_unknown_weight,
                train_group_weight_mode=train_group_weight_mode,
                group_loss_weights=group_loss_weights,
                use_carry_group=use_carry_group,
            )
            ret["use_cerberus_loss"] = cfg.PERSON_SEARCH.REID.get("USE_CERBERUS_LOSS", True)
            ret["cerberus_loss_weight"] = cfg.PERSON_SEARCH.REID.get("CERBERUS_LOSS_WEIGHT", 1.0)
            group_score_weight = cfg.PERSON_SEARCH.REID.get("CERBERUS_GROUP_SCORE_WEIGHT", None)
            if group_score_weight is None:
                group_score_weight = cfg.PERSON_SEARCH.REID.get("CERBERUS_FEAT_WEIGHT", 1.0)
            ret["cerberus_group_score_weight"] = group_score_weight
            ret["freeze_cerberus_branch"] = cfg.PERSON_SEARCH.REID.get("FREEZE_CERBERUS_BRANCH", False)
            ret["unfreeze_cerberus_at_iter"] = cfg.PERSON_SEARCH.REID.get("UNFREEZE_CERBERUS_AT_ITER", 0)
            ret["cerberus_train_use_infer_path"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_TRAIN_USE_INFER_PATH", True
            )
            ret["cerberus_detach_visual_inputs"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_DETACH_VISUAL_INPUTS", False
            )
            ret["cerberus_train_branch_only"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_TRAIN_BRANCH_ONLY", False
            )
            ret["cerberus_warmup_iters"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_WARMUP_ITERS", 0
            )
            ret["cerberus_stage1_baseline_iters"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_STAGE1_BASELINE_ITERS", 0
            )
            ret["cerberus_stage2_freeze_non_branch"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_STAGE2_FREEZE_NON_BRANCH", False
            )
            ret["cerberus_stage2_branch_only"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_STAGE2_BRANCH_ONLY", False
            )
            ret["cerberus_stage2_detach_visual_inputs"] = cfg.PERSON_SEARCH.REID.get(
                "CERBERUS_STAGE2_DETACH_VISUAL_INPUTS", False
            )

            # --- addon modules (Memory Bank / GADA / BRDA) ---
            if cfg.PERSON_SEARCH.REID.get("USE_CERBERUS_MEMORY_BANK", False):
                queue_size = cfg.PERSON_SEARCH.REID.get(
                    "CERBERUS_MEMORY_QUEUE_SIZE", 256
                )
                ret["memory_bank"] = CerberusMemoryBank(
                    queue_size=queue_size,
                    embed_dim=semantic_dim,
                    group_names=ret["cerberus_branch"].group_names,
                )
                ret["memory_loss_weight"] = cfg.PERSON_SEARCH.REID.get(
                    "CERBERUS_MEMORY_LOSS_WEIGHT", 0.5
                )

            if cfg.PERSON_SEARCH.REID.get("USE_CERBERUS_GADA", False):
                ret["gada_module"] = CerberusGADALoss(
                    global_dim=cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM,
                    semantic_dim=semantic_dim,
                    group_names=ret["cerberus_branch"].group_names,
                )
                ret["gada_loss_weight"] = cfg.PERSON_SEARCH.REID.get(
                    "CERBERUS_GADA_WEIGHT", 0.5
                )

            if cfg.PERSON_SEARCH.REID.get("USE_CERBERUS_BRDA", False):
                ret["brda_module"] = CerberusBRDALoss(
                    smoothing_floor=cfg.PERSON_SEARCH.REID.get(
                        "CERBERUS_BRDA_SMOOTHING_FLOOR", 0.05
                    ),
                    smoothing_cap=cfg.PERSON_SEARCH.REID.get(
                        "CERBERUS_BRDA_SMOOTHING_CAP", 0.6
                    ),
                    group_names=ret["cerberus_branch"].group_names,
                )
                ret["brda_loss_weight"] = cfg.PERSON_SEARCH.REID.get(
                    "CERBERUS_BRDA_WEIGHT", 1.0
                )
        else:
            ret["cerberus_branch"] = None
            ret["use_cerberus_loss"] = False
            ret["cerberus_loss_weight"] = 0.0
            ret["cerberus_group_score_weight"] = 0.0
            ret["freeze_cerberus_branch"] = False
            ret["unfreeze_cerberus_at_iter"] = 0
            ret["cerberus_train_use_infer_path"] = True
            ret["cerberus_detach_visual_inputs"] = False
            ret["cerberus_train_branch_only"] = False
            ret["cerberus_stage1_baseline_iters"] = 0
            ret["cerberus_stage2_freeze_non_branch"] = False
            ret["cerberus_stage2_branch_only"] = False
            ret["cerberus_stage2_detach_visual_inputs"] = False
        return ret

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

    def _encode_attribute_text_groups(self, attributes_list):
        prompts_by_group, valid_masks = self.cerberus_branch.build_group_prompts(attributes_list)
        text_features = {}
        for group_name in self.cerberus_branch.group_names:
            group_tokens = torch.stack(
                [self._tokenize_attribute_prompt(prompt) for prompt in prompts_by_group[group_name]]
            ).to(self.device)
            text_features[f"{group_name}_embedding"] = self._encode_clip_text_features(group_tokens)

        text_embeddings = self.cerberus_branch.project_text_embeddings(text_features)
        valid_masks = {key: value.to(self.device) for key, value in valid_masks.items()}
        return text_embeddings, valid_masks

    def _maybe_unfreeze_cerberus_branch(self):
        if not self.freeze_cerberus_branch or self.cerberus_branch_unfrozen:
            return

        try:
            from psd2.utils.events import get_event_storage

            if get_event_storage().iter < self.unfreeze_cerberus_at_iter:
                return
        except Exception:
            return

        for parameter in self.cerberus_branch.parameters():
            parameter.requires_grad = True
        self.cerberus_branch_unfrozen = True
        logger.info("Cerberus-inspired branch unfrozen")

    def _get_current_training_iter(self):
        try:
            from psd2.utils.events import get_event_storage

            return int(get_event_storage().iter)
        except Exception:
            pass
        return int(getattr(self, "iter", 0))

    def _should_use_cerberus_loss(self):
        if not self.use_cerberus_loss:
            return False
        if not self.training:
            return True
        if self.cerberus_stage1_baseline_iters <= 0:
            return True
        return self._get_current_training_iter() >= self.cerberus_stage1_baseline_iters

    def _current_cerberus_train_branch_only(self):
        if self.cerberus_stage2_activated:
            return self.cerberus_stage2_branch_only
        return self.cerberus_train_branch_only

    def _current_cerberus_detach_visual_inputs(self):
        if self.cerberus_stage2_activated:
            return self.cerberus_stage2_detach_visual_inputs
        return self.cerberus_detach_visual_inputs

    def _get_stage2_trainable_prefixes(self):
        prefixes = ["cerberus_branch."]
        if self.memory_bank is not None:
            prefixes.append("memory_bank.")
        if self.gada_module is not None:
            prefixes.append("gada_module.")
        if self.brda_module is not None:
            prefixes.append("brda_module.")
        return tuple(prefixes)

    def _should_force_frozen_modules_eval(self):
        if not self.training:
            return False
        return (
            self._current_cerberus_train_branch_only()
            or self.cerberus_stage2_freeze_non_branch
        )

    def _enforce_frozen_module_modes(self):
        if not self.training:
            return

        force_eval = self._should_force_frozen_modules_eval()
        if not force_eval:
            self._frozen_module_mode_logged = False
            return

        # Keep the model on the training path, but freeze mutable-state layers
        # inside non-branch modules so BatchNorm/Dropout do not drift the stage-1 trunk.
        branch_prefixes = self._get_stage2_trainable_prefixes()
        for name, module in self.named_modules():
            if not name:
                continue
            if any(name.startswith(prefix.rstrip(".")) for prefix in branch_prefixes):
                continue
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
            elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.eval()

        if force_eval and not self._frozen_module_mode_logged:
            logger.info(
                "Keeping frozen non-branch BatchNorm/Dropout layers in eval mode during attr-only training"
            )
            self._frozen_module_mode_logged = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._enforce_frozen_module_modes()
        else:
            self._frozen_module_mode_logged = False
        return self

    def _activate_cerberus_stage2(self):
        if self.cerberus_stage2_activated:
            return

        trainable_count = 0
        frozen_count = 0
        prefixes = ()
        if self.cerberus_stage2_freeze_non_branch:
            prefixes = self._get_stage2_trainable_prefixes()
            for name, parameter in self.named_parameters():
                keep_trainable = any(name.startswith(prefix) for prefix in prefixes)
                parameter.requires_grad = keep_trainable
                if keep_trainable:
                    trainable_count += parameter.numel()
                else:
                    frozen_count += parameter.numel()

        self.cerberus_stage2_activated = True
        self._enforce_frozen_module_modes()
        logger.info(
            "Activated Cerberus attr-only stage at iter %d; trainable prefixes=%s, "
            "freeze_non_branch=%s, trainable_params=%d, frozen_params=%d, "
            "branch_only=%s, detach_visual_inputs=%s",
            self._get_current_training_iter(),
            prefixes,
            self.cerberus_stage2_freeze_non_branch,
            trainable_count,
            frozen_count,
            self.cerberus_stage2_branch_only,
            self.cerberus_stage2_detach_visual_inputs,
        )

    def _maybe_activate_cerberus_stage2(self):
        if (
            not self.training
            or self.cerberus_stage2_activated
            or self.cerberus_stage1_baseline_iters <= 0
        ):
            return
        if self._get_current_training_iter() < self.cerberus_stage1_baseline_iters:
            return
        self._activate_cerberus_stage2()

    def _use_cerberus_group_scores(self):
        return self.cerberus_branch is not None and self.cerberus_group_score_weight > 0

    def _build_cerberus_visual_embeddings(self, box_features, allow_grad=False):
        if not self.cerberus_train_use_infer_path:
            return None

        was_training = self.bn_neck.training if isinstance(self.bn_neck, nn.Module) else False
        try:
            if isinstance(self.bn_neck, nn.Module):
                self.bn_neck.eval()
            if allow_grad:
                visual_embs = self.img_emb_infer(box_features)
                visual_embs = self.bn_neck(visual_embs)
                visual_embs = F.normalize(visual_embs, dim=-1)
            else:
                with torch.no_grad():
                    visual_embs = self.img_emb_infer(box_features)
                    visual_embs = self.bn_neck(visual_embs)
                    visual_embs = F.normalize(visual_embs, dim=-1)
            return visual_embs
        finally:
            if isinstance(self.bn_neck, nn.Module):
                self.bn_neck.train(was_training)

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

    def _get_cerberus_loss_weight(self):
        """Compute Cerberus loss weight with warmup."""
        if self.cerberus_warmup_iters <= 0 or not self.training:
            return self.cerberus_loss_weight

        current_iter = self._get_current_training_iter()

        if current_iter >= self.cerberus_warmup_iters:
            return self.cerberus_loss_weight
        else:
            # Linear warmup from 0 to cerberus_loss_weight
            warmup_factor = float(current_iter) / float(self.cerberus_warmup_iters)
            return self.cerberus_loss_weight * warmup_factor

    def _compute_cerberus_losses(self, box_embs, box_features, semantic_attributes):
        if self.cerberus_train_use_infer_path:
            visual_embs = self._build_cerberus_visual_embeddings(box_features)
            patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1).detach()
            feature_map = box_features
        else:
            num_samples = len(box_embs) // 3
            visual_embs = box_embs[:num_samples, :1024].detach()
            patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1).detach()[:num_samples]
            feature_map = box_features[:num_samples]
            semantic_attributes = semantic_attributes[:num_samples]

        if self._current_cerberus_detach_visual_inputs() and feature_map is not None:
            feature_map = feature_map.detach()

        cerberus_embeddings, valid_masks = self.cerberus_branch(
            visual_embs, patch_tokens, feature_map=feature_map, return_valid=True
        )
        if semantic_attributes:
            raw_losses = self.cerberus_branch.compute_losses(
                cerberus_embeddings, semantic_attributes, valid_mask_dict=valid_masks,
            )
            current_weight = self._get_cerberus_loss_weight()
            return {key: value * current_weight for key, value in raw_losses.items()}

        dummy = sum(x.sum() for x in cerberus_embeddings.values()) * 0.0
        losses = {}
        for group_name in self.cerberus_branch.group_names:
            losses[f"loss_cerberus_cls_{group_name}"] = dummy
            losses[f"loss_cerberus_align_{group_name}"] = dummy
            losses[f"loss_cerberus_guidance_{group_name}"] = dummy
            losses[f"loss_cerberus_relation_{group_name}"] = dummy
        return losses

    def _has_addon_modules(self):
        return (
            self.memory_bank is not None
            or self.gada_module is not None
            or self.brda_module is not None
        )

    def _compute_memory_augmented_cls(
        self,
        cerberus_embedding_dict,
        attributes_list,
        valid_mask_dict,
    ):
        """CE loss with memory-augmented key set for each group."""
        losses = {}
        device = next(iter(cerberus_embedding_dict.values())).device
        zero = torch.tensor(0.0, device=device)
        bank = self.cerberus_branch.prototype_bank
        scale = self.cerberus_branch.logit_scale

        for group_name in self.cerberus_branch.group_names:
            emb = cerberus_embedding_dict[f"{group_name}_embedding"]
            prototypes = bank.get_group_prototypes(group_name)
            group_valid = None
            if valid_mask_dict is not None:
                gv = valid_mask_dict.get(f"{group_name}_valid")
                if gv is not None:
                    group_valid = gv.detach().cpu().tolist()

            valid_indices = []
            target_indices = []
            for idx, attr_dict in enumerate(attributes_list):
                target = bank.get_group_index(group_name, attr_dict)
                if target >= 0 and (group_valid is None or group_valid[idx]):
                    valid_indices.append(idx)
                    target_indices.append(target)

            if not valid_indices:
                losses[f"loss_memory_cls_{group_name}"] = zero
                continue

            vi = torch.tensor(valid_indices, device=device)
            ti = torch.tensor(target_indices, device=device)
            valid_emb = F.normalize(emb[vi], dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)

            mem_embs, mem_indices = self.memory_bank.get_memory(group_name)
            if mem_embs.shape[0] == 0:
                losses[f"loss_memory_cls_{group_name}"] = zero
                continue

            mem_norm = F.normalize(mem_embs, dim=-1)
            mem_logits = valid_emb @ mem_norm.t() * scale
            same_class_mask = ti.view(-1, 1).eq(mem_indices.to(device).view(1, -1))
            mem_logits = mem_logits.masked_fill(same_class_mask, -1e4)
            proto_logits = valid_emb @ proto_norm.t() * scale
            all_logits = torch.cat([proto_logits, mem_logits], dim=1)

            # targets index into prototypes (first C rows)
            loss = F.cross_entropy(all_logits, ti)
            losses[f"loss_memory_cls_{group_name}"] = loss

        return losses

    def _compute_cerberus_losses_extended(
        self, box_embs, box_features, semantic_attributes
    ):
        """Like _compute_cerberus_losses but routes through addon modules."""
        # 1. prepare visual inputs (same as original)
        if self.cerberus_train_use_infer_path:
            visual_embs = self._build_cerberus_visual_embeddings(box_features)
            gada_visual_embs = visual_embs
            if (
                self.gada_module is not None
                and not self._current_cerberus_detach_visual_inputs()
                and not self._current_cerberus_train_branch_only()
            ):
                gada_visual_embs = self._build_cerberus_visual_embeddings(
                    box_features, allow_grad=True
                )
            patch_tokens = (
                box_features.flatten(2, 3).permute(0, 2, 1).detach()
            )
            feature_map = box_features
        else:
            num_samples = len(box_embs) // 3
            visual_embs = box_embs[:num_samples, :1024].detach()
            gada_visual_embs = visual_embs
            if (
                self.gada_module is not None
                and not self._current_cerberus_detach_visual_inputs()
                and not self._current_cerberus_train_branch_only()
            ):
                gada_visual_embs = box_embs[:num_samples, :1024]
            patch_tokens = (
                box_features.flatten(2, 3)
                .permute(0, 2, 1)
                .detach()[:num_samples]
            )
            feature_map = box_features[:num_samples]
            semantic_attributes = semantic_attributes[:num_samples]

        if self._current_cerberus_detach_visual_inputs() and feature_map is not None:
            feature_map = feature_map.detach()

        # 2. forward through cerberus branch — expose embeddings
        cerberus_embeddings, valid_masks = self.cerberus_branch(
            visual_embs,
            patch_tokens,
            feature_map=feature_map,
            return_valid=True,
        )

        if not semantic_attributes:
            dummy = sum(x.sum() for x in cerberus_embeddings.values()) * 0.0
            losses = {}
            for gn in self.cerberus_branch.group_names:
                losses[f"loss_cerberus_cls_{gn}"] = dummy
                losses[f"loss_cerberus_align_{gn}"] = dummy
                losses[f"loss_cerberus_guidance_{gn}"] = dummy
                losses[f"loss_cerberus_relation_{gn}"] = dummy
            return losses

        # 3. original losses (compute_losses untouched)
        raw_losses = self.cerberus_branch.compute_losses(
            cerberus_embeddings,
            semantic_attributes,
            valid_mask_dict=valid_masks,
        )
        losses = {
            k: v * self.cerberus_loss_weight for k, v in raw_losses.items()
        }

        # 4. [BRDA] — replace cls losses with smoothed version
        if self.brda_module is not None:
            brda_losses = self.brda_module(
                cerberus_embeddings,
                semantic_attributes,
                valid_masks,
                self.cerberus_branch.prototype_bank,
                logit_scale=self.cerberus_branch.logit_scale,
            )
            for gn in self.cerberus_branch.group_names:
                key_orig = f"loss_cerberus_cls_{gn}"
                key_brda = f"loss_brda_cls_{gn}"
                if key_brda in brda_losses:
                    losses[key_orig] = (
                        brda_losses[key_brda]
                        * self.brda_loss_weight
                        * self.cerberus_loss_weight
                    )

        # 5. [Memory] — augmented cls + queue update
        if self.memory_bank is not None:
            mem_losses = self._compute_memory_augmented_cls(
                cerberus_embeddings,
                semantic_attributes,
                valid_masks,
            )
            for k, v in mem_losses.items():
                losses[k] = v * self.memory_loss_weight * self.cerberus_loss_weight

            # enqueue current batch (no grad)
            bank = self.cerberus_branch.prototype_bank
            for gn in self.cerberus_branch.group_names:
                emb = cerberus_embeddings[f"{gn}_embedding"]
                gv = valid_masks.get(f"{gn}_valid")
                indices = []
                for idx, attr_dict in enumerate(semantic_attributes):
                    target = bank.get_group_index(gn, attr_dict)
                    if gv is not None and not gv[idx]:
                        target = -1
                    indices.append(target)
                idx_t = torch.tensor(indices, device=emb.device)
                self.memory_bank.update(gn, emb, idx_t)

        # 6. [GADA] — global-attribute alignment
        if self.gada_module is not None:
            gada_losses = self.gada_module(
                gada_visual_embs,
                cerberus_embeddings,
                self.cerberus_branch.prototype_bank,
                logit_scale=self.cerberus_branch.logit_scale,
            )
            for k, v in gada_losses.items():
                losses[k] = v * self.gada_loss_weight * self.cerberus_loss_weight

        return losses

    def forward_gallery(self, image_list, gt_instances):
        if self.training:
            self._maybe_unfreeze_cerberus_branch()
            self._maybe_activate_cerberus_stage2()
            self._enforce_frozen_module_modes()

        features = self.backbone(image_list.tensor)
        proposals, proposal_losses = self.proposal_generator(
            image_list, features, gt_instances if self.training else None
        )
        pred_instances, box_features, roi_losses, pos_match_indices = self.roi_heads(
            image_list, features, proposals, gt_instances if self.training else None
        )
        #
        if self.training:
            losses = {}
            if not self._current_cerberus_train_branch_only():
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
            assign_ids = assign_ids[pos_mask]

            box_embs, assign_ids = self.img_embes(box_features, assign_ids)
            box_embs = self.bn_neck(box_embs)

            if not self._current_cerberus_train_branch_only():
                reid_loss = self.oim_loss(box_embs, assign_ids)
                for key, value in reid_loss.items():
                    losses[key] = value * 0.5

                text_tokens = []
                text_pids = []
                for img_gt in gt_instances:
                    for pid, p_tokens in zip(img_gt.gt_pids, img_gt.descriptions):
                        if pid > -1:
                            text_tokens.extend(p_tokens)
                            text_pids.extend([pid] * len(p_tokens))
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
                    for key, value in reid_loss_text.items():
                        losses[f"{key}_text"] = value * 0.5
                    lb_box_embs = box_embs[assign_ids > -1]
                    lb_assign_ids = assign_ids[assign_ids > -1]
                    losses["loss_sdm"] = self.compute_sdm(
                        lb_box_embs, text_embs, lb_assign_ids, text_pids_tensor, 1 / 0.02
                    )
                else:
                    losses["loss_oim_text"] = text_embs.sum() * 0.0
                    losses["loss_sdm"] = torch.tensor(0.0, device=self.device)

            branch_ready = self._should_use_cerberus_loss() and self.cerberus_branch is not None
            if branch_ready and (not self.freeze_cerberus_branch or self.cerberus_branch_unfrozen):
                if self._has_addon_modules():
                    losses.update(self._compute_cerberus_losses_extended(
                        box_embs, box_features, semantic_attributes
                    ))
                else:
                    losses.update(self._compute_cerberus_losses(
                        box_embs, box_features, semantic_attributes
                    ))
            elif self.freeze_cerberus_branch and self.cerberus_branch is not None:
                if self._has_addon_modules():
                    losses.update(self._compute_cerberus_losses_extended(
                        box_embs, box_features, []
                    ))
                else:
                    losses.update(self._compute_cerberus_losses(
                        box_embs, box_features, []
                    ))

            return pred_instances, [feat.detach() for feat in features.values()], losses

        if len(pred_instances) == 0:
            return pred_instances

        box_embs = self.img_emb_infer(box_features)
        box_embs = self.bn_neck(box_embs)
        box_embs = F.normalize(box_embs, dim=-1)

        cerberus_group_feats_split = [None] * len(pred_instances)
        cerberus_group_valid_split = [None] * len(pred_instances)
        cerberus_group_labels_split = [None] * len(pred_instances)
        if self._use_cerberus_group_scores():
            patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)
            cerberus_embeddings, cerberus_valid_masks = self.cerberus_branch(
                box_embs, patch_tokens, feature_map=box_features, return_valid=True
            )
            cerberus_group_feats = self.cerberus_branch.stack_embeddings(cerberus_embeddings)
            cerberus_group_valid = self.cerberus_branch.stack_valid_masks(cerberus_valid_masks)
            cerberus_group_labels = self.cerberus_branch.predict_group_labels(
                cerberus_embeddings, cerberus_valid_masks
            )
            split_sizes = [len(inst) for inst in pred_instances]
            cerberus_group_feats_split = torch.split(cerberus_group_feats, split_sizes)
            cerberus_group_valid_split = torch.split(cerberus_group_valid, split_sizes)
            cerberus_group_labels_split = torch.split(cerberus_group_labels, split_sizes)

        reid_feats_split = torch.split(box_embs, [len(inst) for inst in pred_instances])
        score_t = self.roi_heads.box_predictor.test_score_thresh
        iou_t = self.roi_heads.box_predictor.test_nms_thresh
        from psd2.layers import batched_nms
        from psd2.structures import Boxes, BoxMode

        for pred_i, gt_i, reid_feats_i, group_feats_i, group_valid_i, group_labels_i in zip(
            pred_instances,
            gt_instances,
            reid_feats_split,
            cerberus_group_feats_split,
            cerberus_group_valid_split,
            cerberus_group_labels_split,
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
            pred_feats = reid_feats_i[filter_mask][keep]
            if getattr(self, "cws", False):
                pred_feats = pred_feats * pred_scores[keep].view(-1, 1)
            pred_i.reid_feats = pred_feats
            if group_feats_i is not None and group_valid_i is not None:
                pred_group_feats = group_feats_i[filter_mask][keep]
                if getattr(self, "cws", False):
                    pred_group_feats = pred_group_feats * pred_scores[keep].view(-1, 1, 1)
                pred_i.cerberus_group_feats = pred_group_feats
                pred_i.cerberus_group_valid = group_valid_i[filter_mask][keep]
                pred_i.cerberus_group_labels = group_labels_i[filter_mask][keep]
                pred_i.cerberus_group_score_weight = pred_scores[keep].new_full(
                    (pred_group_feats.shape[0],), self.cerberus_group_score_weight
                )
            if hasattr(pred_i, "pred_classes"):
                pred_i.pred_classes = pred_i.pred_classes[filter_mask][keep]
            org_h, org_w = gt_i.org_img_size
            h, w = gt_i.image_size
            pred_i.pred_boxes.scale(org_w / w, org_h / h)
        return pred_instances

    def forward_gallery_gt(self, image_list, gt_instances):
        features = self.backbone(image_list.tensor)
        gt_boxes = [instances_i.gt_boxes for instances_i in gt_instances]
        num_boxes_per_image = [len(boxes_i) for boxes_i in gt_boxes]

        outputs = []
        if sum(num_boxes_per_image) == 0:
            return [
                Instances(
                    gt_i.org_img_size,
                    pred_boxes=Boxes(gt_i.org_gt_boxes.tensor[:0]),
                    pred_scores=torch.zeros(0, device=self.device),
                    reid_feats=torch.zeros(
                        0,
                        self.bn_neck.num_features if hasattr(self.bn_neck, "num_features") else 1024,
                        device=self.device,
                    ),
                )
                for gt_i in gt_instances
            ]

        box_features = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], gt_boxes
        )
        box_embs = self.img_emb_infer(box_features)
        box_embs = self.bn_neck(box_embs)
        box_embs = F.normalize(box_embs, dim=-1)

        cerberus_group_feats_split = [None] * len(gt_instances)
        cerberus_group_valid_split = [None] * len(gt_instances)
        cerberus_group_labels_split = [None] * len(gt_instances)
        if self._use_cerberus_group_scores():
            patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)
            cerberus_embeddings, cerberus_valid_masks = self.cerberus_branch(
                box_embs, patch_tokens, feature_map=box_features, return_valid=True
            )
            cerberus_group_feats = self.cerberus_branch.stack_embeddings(cerberus_embeddings)
            cerberus_group_valid = self.cerberus_branch.stack_valid_masks(cerberus_valid_masks)
            cerberus_group_labels = self.cerberus_branch.predict_group_labels(
                cerberus_embeddings, cerberus_valid_masks
            )
            cerberus_group_feats_split = torch.split(cerberus_group_feats, num_boxes_per_image)
            cerberus_group_valid_split = torch.split(cerberus_group_valid, num_boxes_per_image)
            cerberus_group_labels_split = torch.split(cerberus_group_labels, num_boxes_per_image)

        reid_feats_split = torch.split(box_embs, num_boxes_per_image)
        for gt_i, reid_feats_i, group_feats_i, group_valid_i, group_labels_i in zip(
            gt_instances,
            reid_feats_split,
            cerberus_group_feats_split,
            cerberus_group_valid_split,
            cerberus_group_labels_split,
        ):
            pred_scores = reid_feats_i.new_ones((reid_feats_i.shape[0],))
            inst_kwargs = {
                "pred_boxes": Boxes(gt_i.org_gt_boxes.tensor.clone()),
                "pred_scores": pred_scores,
                "reid_feats": reid_feats_i,
            }
            if group_feats_i is not None and group_valid_i is not None:
                inst_kwargs["cerberus_group_feats"] = group_feats_i
                inst_kwargs["cerberus_group_valid"] = group_valid_i
                inst_kwargs["cerberus_group_labels"] = group_labels_i
                inst_kwargs["cerberus_group_score_weight"] = pred_scores.new_full(
                    (group_feats_i.shape[0],), self.cerberus_group_score_weight
                )
            outputs.append(Instances(gt_i.org_img_size, **inst_kwargs))
        return outputs

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

        text_input = torch.stack(text_tokens).to(self.device)
        text_embs = self.text_embeds(text_input)
        text_embs = F.normalize(text_embs, dim=-1)

        query_group_feats = None
        query_group_valid = None
        query_group_labels = None
        if self._use_cerberus_group_scores():
            query_group_feats, query_group_valid = self.cerberus_branch.encode_query_attributes_grouped(
                attributes_list, self.device
            )
            query_group_labels = self.cerberus_branch.encode_query_group_labels(
                attributes_list, self.device
            )
            query_texts = []
            for inst in gt_instances:
                raw_text = ""
                if hasattr(inst, "raw_descriptions") and len(inst.raw_descriptions) > 0:
                    raw_group = inst.raw_descriptions[0]
                    if isinstance(raw_group, list) and len(raw_group) > 0:
                        raw_text = raw_group[0]
                query_texts.append(raw_text)
            query_group_valid = query_group_valid.float() * self.cerberus_branch.compute_query_group_weights(
                attributes_list, self.device, query_texts=query_texts
            )

        query_feats_split = torch.split(text_embs, 1)
        if query_group_feats is not None:
            query_group_feats_split = torch.split(query_group_feats, 1)
            query_group_valid_split = torch.split(query_group_valid, 1)
            query_group_labels_split = torch.split(query_group_labels, 1)
        else:
            query_group_feats_split = [None] * len(query_feats_split)
            query_group_valid_split = [None] * len(query_feats_split)
            query_group_labels_split = [None] * len(query_feats_split)

        outputs = []
        for i, query_feat in enumerate(query_feats_split):
            inst_kwargs = {"reid_feats": query_feat}
            if query_group_feats_split[i] is not None and query_group_valid_split[i] is not None:
                inst_kwargs["cerberus_group_feats"] = query_group_feats_split[i]
                inst_kwargs["cerberus_group_valid"] = query_group_valid_split[i]
                inst_kwargs["cerberus_group_labels"] = query_group_labels_split[i]
                inst_kwargs["cerberus_group_score_weight"] = query_feat.new_full(
                    (1,), self.cerberus_group_score_weight
                )
            outputs.append(Instances(gt_instances[i].image_size, **inst_kwargs))
        return outputs


@META_ARCH_REGISTRY.register()
class ClipViperCerberusPS(ClipSimpleCerberusPS):
    pass
