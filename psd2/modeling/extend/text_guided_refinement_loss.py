"""
Loss functions for Text-Guided Feature Refinement
Inspired by DIFFER/TVFR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class TextGuidedRefinementLoss(nn.Module):
    """
    Losses for text-guided feature refinement
    Encourages the refined features to be better aligned with text descriptions
    """
    def __init__(self, 
                 temperature: float = 0.07,
                 use_contrastive_loss: bool = True,
                 use_consistency_loss: bool = True,
                 use_diversity_loss: bool = True):
        """
        Args:
            temperature: Temperature for contrastive loss
            use_contrastive_loss: Whether to use contrastive loss
            use_consistency_loss: Whether to use consistency loss
            use_diversity_loss: Whether to use diversity loss
        """
        super().__init__()
        self.temperature = temperature
        self.use_contrastive_loss = use_contrastive_loss
        self.use_consistency_loss = use_consistency_loss
        self.use_diversity_loss = use_diversity_loss
    
    def contrastive_loss(self, 
                        refined_features: torch.Tensor, 
                        text_features: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss to align refined features with text features
        
        Args:
            refined_features: Refined visual features [N, D]
            text_features: Text features [N, D]
            labels: Identity labels [N]
        
        Returns:
            Contrastive loss
        """
        # Normalize features
        refined_features = F.normalize(refined_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(refined_features, text_features.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(refined_features.device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(mask.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute contrastive loss
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def consistency_loss(self,
                        refined_features: torch.Tensor,
                        original_features: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """
        Consistency loss: refined features should preserve identity information
        
        Args:
            refined_features: Refined visual features [N, D]
            original_features: Original visual features [N, D]
            labels: Identity labels [N]
        
        Returns:
            Consistency loss
        """
        # Normalize features
        refined_features = F.normalize(refined_features, dim=-1)
        original_features = F.normalize(original_features, dim=-1)
        
        # Compute similarity between refined and original features
        similarity = torch.sum(refined_features * original_features, dim=-1)
        
        # Loss: maximize similarity for same identity
        loss = 1.0 - similarity.mean()
        
        return loss
    
    def diversity_loss(self,
                      stripe_features: List[torch.Tensor],
                      labels: torch.Tensor) -> torch.Tensor:
        """
        Diversity loss: encourage different stripes to capture different semantic information
        
        Args:
            stripe_features: List of stripe features, each [N, D]
            labels: Identity labels [N]
        
        Returns:
            Diversity loss
        """
        num_stripes = len(stripe_features)
        if num_stripes < 2:
            return torch.tensor(0.0, device=stripe_features[0].device)
        
        # Normalize all stripe features
        normalized_stripes = [F.normalize(feat, dim=-1) for feat in stripe_features]
        
        # Compute pairwise diversity
        diversity_loss = 0.0
        count = 0
        
        for i in range(num_stripes):
            for j in range(i + 1, num_stripes):
                # Cosine similarity between different stripes
                similarity = torch.sum(normalized_stripes[i] * normalized_stripes[j], dim=-1)
                # Loss: minimize similarity (encourage diversity)
                diversity_loss += similarity.mean()
                count += 1
        
        diversity_loss = diversity_loss / count if count > 0 else torch.tensor(0.0)
        
        return diversity_loss
    
    def forward(self,
                refined_features: torch.Tensor,
                text_features: torch.Tensor,
                original_features: torch.Tensor,
                labels: torch.Tensor,
                stripe_features: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all text-guided refinement losses
        
        Args:
            refined_features: Refined visual features [N, D]
            text_features: Text features [N, D]
            original_features: Original visual features [N, D]
            labels: Identity labels [N]
            stripe_features: Optional list of stripe features for diversity loss
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        if self.use_contrastive_loss:
            loss_contrastive = self.contrastive_loss(refined_features, text_features, labels)
            losses['loss_refinement_contrastive'] = loss_contrastive
        
        if self.use_consistency_loss:
            loss_consistency = self.consistency_loss(refined_features, original_features, labels)
            losses['loss_refinement_consistency'] = loss_consistency
        
        if self.use_diversity_loss and stripe_features is not None:
            loss_diversity = self.diversity_loss(stripe_features, labels)
            losses['loss_refinement_diversity'] = loss_diversity
        
        return losses


class StripeAlignmentLoss(nn.Module):
    """
    Loss to align stripe features with corresponding text parts
    """
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for alignment loss
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                stripe_features: List[torch.Tensor],
                text_features: torch.Tensor,
                stripe_text_attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute stripe-text alignment loss
        
        Args:
            stripe_features: List of stripe features, each [N, D]
            text_features: Text features [N, D]
            stripe_text_attention: Optional attention weights [N, num_stripes] indicating which stripes are relevant
        
        Returns:
            Alignment loss
        """
        num_stripes = len(stripe_features)
        if num_stripes == 0:
            return torch.tensor(0.0, device=text_features.device)
        
        # Normalize features
        normalized_stripes = [F.normalize(feat, dim=-1) for feat in stripe_features]
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute alignment loss for each stripe
        alignment_losses = []
        
        for i, stripe_feat in enumerate(normalized_stripes):
            # Similarity between stripe and text
            similarity = torch.sum(stripe_feat * text_features, dim=-1) / self.temperature
            
            if stripe_text_attention is not None:
                # Weight by attention (higher attention for relevant stripes)
                weight = stripe_text_attention[:, i]
                loss = -torch.mean(weight * similarity)
            else:
                # Simple alignment: maximize similarity
                loss = -similarity.mean()
            
            alignment_losses.append(loss)
        
        # Average across stripes
        total_loss = torch.stack(alignment_losses).mean()
        
        return total_loss


def compute_text_guided_refinement_losses(
    refined_features: torch.Tensor,
    text_features: torch.Tensor,
    original_features: torch.Tensor,
    labels: torch.Tensor,
    stripe_features: Optional[List[torch.Tensor]] = None,
    stripe_text_attention: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute text-guided refinement losses with configurable weights
    
    Args:
        refined_features: Refined visual features [N, D]
        text_features: Text features [N, D]
        original_features: Original visual features [N, D]
        labels: Identity labels [N]
        stripe_features: Optional list of stripe features
        stripe_text_attention: Optional attention weights for stripes
        loss_weights: Optional dictionary of loss weights
    
    Returns:
        Dictionary of weighted losses
    """
    if loss_weights is None:
        loss_weights = {
            'contrastive': 1.0,
            'consistency': 0.5,
            'diversity': 0.3,
            'alignment': 0.5
        }
    
    # Initialize loss modules
    refinement_loss_module = TextGuidedRefinementLoss()
    alignment_loss_module = StripeAlignmentLoss()
    
    # Compute losses
    losses = {}
    
    # Refinement losses
    refinement_losses = refinement_loss_module(
        refined_features, text_features, original_features, labels, stripe_features
    )
    
    for key, loss in refinement_losses.items():
        weight_key = key.replace('loss_refinement_', '')
        if weight_key in loss_weights:
            losses[key] = loss * loss_weights[weight_key]
        else:
            losses[key] = loss
    
    # Stripe-text alignment loss
    if stripe_features is not None:
        loss_alignment = alignment_loss_module(stripe_features, text_features, stripe_text_attention)
        if 'alignment' in loss_weights:
            losses['loss_refinement_alignment'] = loss_alignment * loss_weights['alignment']
        else:
            losses['loss_refinement_alignment'] = loss_alignment
    
    return losses