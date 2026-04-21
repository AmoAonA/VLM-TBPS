"""
ViPer Model with Attribute Feature Extraction
Integrates DIFFER's attribute feature extraction into ViPer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import os
from typing import Dict, List, Optional, Tuple
from .viper_model import ClipSimple, ClipRes5ROIHeadsPs
from .base_tbps import SearchBaseTBPS
from ..build import META_ARCH_REGISTRY
from psd2.config.config import configurable
from psd2.layers import ShapeSpec
from psd2.modeling.extend.attribute_extractor import AttributeExtractor
from psd2.modeling.extend.attribute_loss import compute_attribute_losses
from psd2.modeling.extend.semantic_branch import SemanticBranch
from psd2.modeling.extend.text_guided_refinement import TextGuidedFeatureRefinementModule

logger = logging.getLogger(__name__)


def convert_frozen_bn(module):
    """Convert BatchNorm layers to frozen version"""
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = FrozenBatchNorm2d(module.num_features)
        if module.affine:
            module_output.weight.data = module.weight.data.clone()
            module_output.bias.data = module.bias.data.clone()
        module_output.running_mean.data = module.running_mean.data.clone()
        module_output.running_var.data = module.running_var.data.clone()
        module_output.num_batches_tracked.data = module.num_batches_tracked.data.clone()
    for name, child in module.named_children():
        module_output.add_module(name, convert_frozen_bn(child))
    del module
    return module_output


class FrozenBatchNorm2d(nn.Module):
    """Frozen BatchNorm2d"""
    def __init__(self, num_features):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


@META_ARCH_REGISTRY.register()
class ClipSimpleAttribute(ClipSimple):
    """
    ClipSimple with attribute feature extraction and semantic branch
    """
    @configurable
    def __init__(
        self,
        clip_model,
        freeze_at_stage2,
        proposal_generator,
        roi_heads,
        id_assigner,
        bn_neck,
        oim_loss,
        cws,
        attribute_extractor=None,
        use_attribute_loss=False,
        attribute_loss_weight=1.0,
        semantic_branch=None,
        use_semantic_loss=False,
        semantic_loss_weight=1.0,
        schema_path=None,
        text_guided_refinement=None,
        use_text_guided_refinement=False,
        text_guided_refinement_loss_weight=0.0,
        *args,
        **kws,
    ) -> None:
        super().__init__(
            clip_model, freeze_at_stage2, proposal_generator, roi_heads,
            id_assigner, bn_neck, oim_loss, cws, *args, **kws
        )
        
        # Attribute feature extraction
        self.attribute_extractor = attribute_extractor
        self.use_attribute_loss = use_attribute_loss
        self.attribute_loss_weight = attribute_loss_weight
        
        # Semantic branch
        self.semantic_branch = semantic_branch
        self.use_semantic_loss = use_semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.schema_path = schema_path
        
        # Text-guided feature refinement (DIFFER/TVFR inspired)
        self.text_guided_refinement = text_guided_refinement
        self.use_text_guided_refinement = use_text_guided_refinement
        self.text_guided_refinement_loss_weight = text_guided_refinement_loss_weight
    
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        
        # Initialize attribute extractor if enabled
        if cfg.PERSON_SEARCH.REID.get('USE_ATTRIBUTES', False):
            embed_dim = cfg.PERSON_SEARCH.REID.MODEL.EMB_DIM
            clip_feat_dim = cfg.PERSON_SEARCH.REID.get('CLIP_DIM', 1024)
            num_nonbio = cfg.PERSON_SEARCH.REID.get('NUM_NONBIO', 3)
            subspace_dim = cfg.PERSON_SEARCH.REID.get('SUBSPACE_DIM', 512)
            last_layer = cfg.PERSON_SEARCH.REID.get('LAST_LAYER', 'clipFc')
            clip_loss_type = cfg.PERSON_SEARCH.REID.get('CLIP_LOSS_TYPE', 'constant')
            reverse_bio = cfg.PERSON_SEARCH.REID.get('REVERSE_BIO', False)
            
            attribute_extractor = AttributeExtractor(
                embed_dim=embed_dim,
                clip_feat_dim=clip_feat_dim,
                num_nonbio=num_nonbio,
                subspace_dim=subspace_dim,
                last_layer=last_layer,
                clip_loss_type=clip_loss_type,
                reverse_bio=reverse_bio
            )
            ret["attribute_extractor"] = attribute_extractor
            ret["use_attribute_loss"] = cfg.PERSON_SEARCH.REID.get('USE_ATTRIBUTE_LOSS', True)
            ret["attribute_loss_weight"] = cfg.PERSON_SEARCH.REID.get('ATTRIBUTE_LOSS_WEIGHT', 1.0)
        else:
            ret["attribute_extractor"] = None
            ret["use_attribute_loss"] = False
            ret["attribute_loss_weight"] = 0.0
        
        # Initialize semantic branch if enabled
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
        else:
            ret["semantic_branch"] = None
            ret["use_semantic_loss"] = False
            ret["semantic_loss_weight"] = 0.0
            ret["schema_path"] = None
        
        # Initialize text-guided refinement module if enabled
        if cfg.PERSON_SEARCH.REID.get('USE_TEXT_GUIDED_REFINEMENT', False):
            visual_feat_dim = cfg.PERSON_SEARCH.REID.get('VISUAL_FEAT_DIM', 2048)
            text_feat_dim = cfg.PERSON_SEARCH.REID.get('TEXT_FEAT_DIM', 1024)
            num_stripes = cfg.PERSON_SEARCH.REID.get('NUM_STRIPES', 6)
            num_stages = cfg.PERSON_SEARCH.REID.get('REFINEMENT_STAGES', 2)
            
            text_guided_refinement = TextGuidedFeatureRefinementModule(
                visual_feat_dim=visual_feat_dim,
                text_feat_dim=text_feat_dim,
                num_stripes=num_stripes,
                num_stages=num_stages
            )
            ret["text_guided_refinement"] = text_guided_refinement
            ret["use_text_guided_refinement"] = True
            ret["text_guided_refinement_loss_weight"] = cfg.PERSON_SEARCH.REID.get('TEXT_GUIDED_REFINEMENT_LOSS_WEIGHT', 1.0)
        else:
            ret["text_guided_refinement"] = None
            ret["use_text_guided_refinement"] = False
            ret["text_guided_refinement_loss_weight"] = 0.0
        
        return ret
    
    def extract_attribute_features(self, features):
        """
        Extract attribute features from image features
        
        Args:
            features: Image features [B, D]
        
        Returns:
            dict: Dictionary containing attribute features
        """
        if self.attribute_extractor is None:
            return None
        
        return self.attribute_extractor(features)
    
    def compute_attribute_losses(self, image_features, caption_features):
        """
        Compute attribute-related losses
        
        Args:
            image_features: Image features from attribute extractor
            caption_features: Text features from data loader
        
        Returns:
            dict: Dictionary of losses
        """
        if not self.use_attribute_loss or self.attribute_extractor is None:
            return {}
        
        # Build configuration dict
        cfg = {
            'CLIP_LOSS_TYPE': self.attribute_extractor.clip_loss_type,
            'logit_scale_bio': self.attribute_extractor.logit_scale_bio,
            'logit_bias_bio': self.attribute_extractor.logit_bias_bio,
        }
        
        # Extract bio and nonbio features
        bio_features = image_features.get('clip_bio_score', None)
        nonbio_features = image_features.get('clip_nonbio_score', None)
        
        if bio_features is None:
            return {}
        
        # Compute losses
        losses = compute_attribute_losses(
            bio_features, nonbio_features, caption_features, cfg
        )
        
        # Scale losses
        for key in losses:
            if key.startswith('loss_'):
                losses[key] = losses[key] * self.attribute_loss_weight
        
        return losses
    
    def losses(self, batched_inputs):
        """
        Compute losses with attribute loss
        """
        losses = super().losses(batched_inputs)
        
        # Add attribute losses if enabled
        if self.use_attribute_loss and self.attribute_extractor is not None:
            # Extract image features
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            
            # Get proposals
            proposals, _ = self.proposal_generator(images, features)
            
            # Get ROI features
            if proposals:
                proposal_boxes = [p.proposal_boxes for p in proposals]
                features_list = [features[f] for f in self.roi_heads.in_features]
                roi_features = self.roi_heads._shared_roi_transform(
                    features_list, proposal_boxes
                )
                
                # Extract attribute features
                attr_features = self.extract_attribute_features(roi_features)
                
                # Get caption features from batch
                caption_features = None
                for batch_input in batched_inputs:
                    if 'instances' in batch_input:
                        instances = batch_input['instances']
                        if hasattr(instances, 'caption_features'):
                            caption_features = instances.caption_features
                            break
                
                if caption_features is not None and attr_features is not None:
                    # Compute attribute losses
                    attr_losses = self.compute_attribute_losses(attr_features, caption_features)
                    losses.update(attr_losses)
        
        return losses
    
    def forward_gallery(self, image_list, gt_instances):
        """
        Forward pass for gallery images with attribute feature extraction and concatenation
        
        Args:
            image_list: List of images
            gt_instances: Ground truth instances
            
        Returns:
            pred_instances: Predicted instances with concatenated features
            features: Backbone features
            losses: Dictionary of losses
        """
        # Call parent forward_gallery to get base features
        if self.training:
            # Training mode: get features and losses
            features = self.backbone(image_list.tensor)
            proposals, proposal_losses = self.proposal_generator(
                image_list, features, gt_instances
            )
            pred_instances, box_features, roi_losses, pos_match_indices = self.roi_heads(
                image_list, features, proposals, gt_instances
            )
            
            # Combine losses
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
            pos_mask = assign_ids > -2
            box_features = box_features[pos_mask]
            assign_ids = assign_ids[pos_mask]
            
            # Extract base image embeddings (ViPer features)
            box_embs = self.img_embes(box_features)
            box_embs = self.bn_neck(box_embs)
            
            # ===== NEW: Extract and concatenate attribute features =====
            if self.attribute_extractor is not None:
                # Flatten ROI features for attribute extraction
                import pdb;pdb.set_trace()
                roi_features_flat = box_features.flatten(2, 3).mean(dim=[2, 3])  # [N, 2048]
                
                # Extract attribute features
                attr_output = self.extract_attribute_features(roi_features_flat)
                
                if attr_output is not None:
                    # Get biometric and non-biometric features
                    bio_features = attr_output.get('clip_bio_score', None)  # [N, 1024]
                    nonbio_features = attr_output.get('clip_nonbio_score', None)  # [N, num_nonbio, 1024]
                    
                    if bio_features is not None:
                        # Concatenate: [ViPer features, Biometric features]
                        # box_embs: [N, 1024] -> [N, 2048]
                        box_embs = torch.cat([box_embs, bio_features], dim=1)
                        
                        # Optionally concatenate non-biometric features
                        if nonbio_features is not None:
                            # Average non-biometric features
                            nonbio_avg = nonbio_features.mean(dim=1)  # [N, 1024]
                            # Concatenate: [ViPer, Bio, NonBio] -> [N, 3072]
                            box_embs = torch.cat([box_embs, nonbio_avg], dim=1)
            
            # ===== NEW: Text-guided feature refinement (DIFFER/TVFR) =====
            if self.use_text_guided_refinement and self.text_guided_refinement is not None:
                # Get text features for refinement
                text_features_for_refinement = []
                for img_gt in gt_instances:
                    if hasattr(img_gt, 'text_features'):
                        text_features_for_refinement.append(img_gt.text_features)
                
                if len(text_features_for_refinement) > 0:
                    # Average text features across the batch
                    text_features_avg = torch.stack(text_features_for_refinement).mean(dim=0)  # [text_feat_dim]
                    
                    # Expand to match batch size
                    text_features_batch = text_features_avg.unsqueeze(0).expand(box_features.shape[0], -1)  # [N, text_feat_dim]
                    
                    # Apply text-guided refinement to ROI features
                    # Reshape box_features to 4D: [N, C, H, W]
                    B, C = box_features.shape[0], box_features.shape[1]
                    # Assume square spatial dimension (e.g., 16x16 for 2048 features)
                    H = W = int((C // 16) ** 0.5) if C == 2048 else 16
                    box_features_4d = box_features.view(B, C, H, W)
                    
                    # Apply refinement
                    refined_features, intermediate_features = self.text_guided_refinement(
                        box_features_4d, text_features_batch
                    )
                    
                    # Update box_embs with refined features
                    # Concatenate refined features with original embeddings
                    box_embs = torch.cat([box_embs, refined_features], dim=1)
                    
                    # Store intermediate features for visualization or analysis
                    losses['refinement_features'] = [f.detach() for f in intermediate_features]
            
            # Compute OIM loss with concatenated features
            reid_loss = self.oim_loss(box_embs, assign_ids)
            for k, v in reid_loss.items():
                losses[k] = v * 0.5
            
            # Text OIM loss
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
            
            if len(text_pids) > 0:
                text_pids = torch.stack(text_pids).to(self.device)
                text_embs = self.text_embeds(torch.stack(text_tokens).to(self.device))
                reid_loss_text = self.oim_loss(text_embs, text_pids)
                for k, v in reid_loss_text.items():
                    losses[k + "_text"] = v * 0.5
                
                lb_box_embs = box_embs[assign_ids > -1]
                lb_assign_ids = assign_ids[assign_ids > -1]
                losses["loss_sdm"] = self.compute_sdm(
                    lb_box_embs, text_embs, lb_assign_ids, text_pids, 1/0.02
                )
            else:
                losses["loss_oim_text"] = torch.tensor(0., device=self.device)
                losses["loss_sdm"] = torch.tensor(0., device=self.device)
            
            # ===== NEW: Compute attribute losses if enabled =====
            if self.use_attribute_loss and self.attribute_extractor is not None:
                # Get caption features from data
                caption_features = []
                for img_gt in gt_instances:
                    if hasattr(img_gt, 'caption_features'):
                        caption_features.append(img_gt.caption_features)
                
                if len(caption_features) > 0:
                    caption_features = torch.cat(caption_features, dim=0)
                    # Extract attribute features again for loss computation
                    attr_output = self.extract_attribute_features(roi_features_flat)
                    
                    if attr_output is not None:
                        attr_losses = self.compute_attribute_losses(attr_output, caption_features)
                        losses.update(attr_losses)
            
            # ===== NEW: Compute semantic loss if enabled =====
            if self.use_semantic_loss and self.semantic_branch is not None:
                # Get patch tokens for spatial semantic embeddings
                patch_tokens = box_features.flatten(2, 3).permute(0, 2, 1)  # [N, 49, 2048]
                
                # Extract semantic embeddings
                semantic_embeddings = self.semantic_branch(
                    org_lembs=box_embs[:, :1024],  # Original ViPer features
                    embs_ech=box_embs[:, :1024],  # ECHO features (use same as original for now)
                    embs_rm=box_embs[:, :1024],   # RM features (use same as original for now)
                    patch_tokens=patch_tokens
                )
                
                # Get attributes from ground truth
                attributes_list = []
                for img_gt in gt_instances:
                    if hasattr(img_gt, 'attributes'):
                        # Get attributes for each person in this image
                        gt_ids = img_gt.gt_pids
                        for i, pid in enumerate(gt_ids):
                            if pid > -1 and hasattr(img_gt, 'gt_attributes'):
                                attributes_list.append(img_gt.gt_attributes[i])
                
                # Compute semantic loss
                if len(attributes_list) > 0:
                    semantic_losses = self.semantic_branch.compute_semantic_loss(
                        semantic_embeddings, attributes_list
                    )
                    for key, value in semantic_losses.items():
                        losses[key] = value * self.semantic_loss_weight
            
            return pred_instances, [feat.detach() for feat in features.values()], losses
            
        else:
            # Inference mode
            return super().forward_gallery(image_list, gt_instances)


@META_ARCH_REGISTRY.register()
class ClipViperAttribute(ClipSimpleAttribute):
    """
    ClipViper with attribute feature extraction
    Enhanced version with spatial attention
    """
    pass


# Register the new models
from psd2.modeling.extend.clip_model import build_CLIP_from_openai_pretrained
from psd2.modeling.proposal_generator import build_proposal_generator
from psd2.modeling.id_assign import build_id_assigner
from psd2.layers.mem_matching_losses import OIMLoss
from torch.nn import init