"""
Text-Guided Visual Feature Refinement Module
Inspired by DIFFER/TVFR: Text-guided Visual Feature Refinement for Text-Based Person Search

This module implements:
1. Horizontal Stripe Partitioning: Splits visual features into horizontal stripes for fine-grained refinement
2. Text-based Filter Generation: Generates description-customized filters from text features
3. Text-guided Feature Refinement: Fuses part-level visual features adaptively for each description
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class HorizontalStripePartitioner(nn.Module):
    """
    Partitions visual features into horizontal stripes for fine-grained refinement
    """
    def __init__(self, num_stripes: int = 6):
        """
        Args:
            num_stripes: Number of horizontal stripes to partition
        """
        super().__init__()
        self.num_stripes = num_stripes
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Partition visual features into horizontal stripes
        
        Args:
            visual_features: Input features [B, C, H, W]
        
        Returns:
            Partitioned features [B, num_stripes, C, H//num_stripes, W]
        """
        B, C, H, W = visual_features.shape
        
        # Ensure height is divisible by num_stripes
        if H % self.num_stripes != 0:
            # Pad if necessary
            pad_size = self.num_stripes - (H % self.num_stripes)
            visual_features = F.pad(visual_features, (0, 0, 0, pad_size), mode='constant', value=0)
            H = H + pad_size
        
        stripe_height = H // self.num_stripes
        
        # Split into horizontal stripes
        stripes = visual_features.view(B, C, self.num_stripes, stripe_height, W)
        stripes = stripes.permute(0, 2, 1, 3, 4)  # [B, num_stripes, C, H//num_stripes, W]
        
        return stripes


class TextBasedFilterGenerator(nn.Module):
    """
    Generates description-customized filters from text features
    """
    def __init__(self, text_feat_dim: int, visual_feat_dim: int, num_stripes: int = 6):
        """
        Args:
            text_feat_dim: Dimension of text features
            visual_feat_dim: Dimension of visual features per stripe
            num_stripes: Number of horizontal stripes
        """
        super().__init__()
        self.num_stripes = num_stripes
        
        # Project text features to filter space
        self.text_to_filter = nn.Sequential(
            nn.Linear(text_feat_dim, visual_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_feat_dim, visual_feat_dim),
        )
        
        # Generate separate filters for each stripe
        self.stripe_filters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_feat_dim, visual_feat_dim),
                nn.Sigmoid()  # Generate attention weights [0, 1]
            ) for _ in range(num_stripes)
        ])
    
    def forward(self, text_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate filters for each stripe
        
        Args:
            text_features: Text features [B, text_feat_dim]
        
        Returns:
            List of filters for each stripe, each [B, 1, 1, visual_feat_dim]
        """
        # Project text features
        projected_text = self.text_to_filter(text_features)  # [B, visual_feat_dim]
        
        # Generate filters for each stripe
        filters = []
        for i in range(self.num_stripes):
            filter_i = self.stripe_filters[i](projected_text)  # [B, visual_feat_dim]
            filter_i = filter_i.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, visual_feat_dim]
            filters.append(filter_i)
        
        return filters


class TextGuidedFeatureRefinement(nn.Module):
    """
    Refines visual features using text guidance
    """
    def __init__(self, visual_feat_dim: int, text_feat_dim: int, num_stripes: int = 6):
        """
        Args:
            visual_feat_dim: Dimension of visual features
            text_feat_dim: Dimension of text features
            num_stripes: Number of horizontal stripes
        """
        super().__init__()
        self.num_stripes = num_stripes
        
        # Initialize sub-modules
        self.partitioner = HorizontalStripePartitioner(num_stripes)
        self.filter_generator = TextBasedFilterGenerator(text_feat_dim, visual_feat_dim, num_stripes)
        
        # Feature refinement network for each stripe
        self.refinement_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(visual_feat_dim, visual_feat_dim, kernel_size=1),
                nn.BatchNorm2d(visual_feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(visual_feat_dim, visual_feat_dim, kernel_size=1),
            ) for _ in range(num_stripes)
        ])
        
        # Cross-stripe attention for context
        self.cross_stripe_attention = nn.MultiheadAttention(
            embed_dim=visual_feat_dim,
            num_heads=8,
            batch_first=True
        )
    
    def refine_stripe(self, stripe_features: torch.Tensor, text_filter: torch.Tensor, 
                      refinement_net: nn.Module) -> torch.Tensor:
        """
        Refine a single stripe using text guidance
        
        Args:
            stripe_features: Stripe features [B, C, H_s, W]
            text_filter: Text-generated filter [B, 1, 1, C]
            refinement_net: Refinement network for this stripe
        
        Returns:
            Refined stripe features [B, C, H_s, W]
        """
        # Apply text filter (element-wise multiplication)
        filtered_features = stripe_features * text_filter  # [B, C, H_s, W]
        
        # Apply refinement network
        refined_features = refinement_net(filtered_features)  # [B, C, H_s, W]
        
        # Residual connection
        refined_features = refined_features + stripe_features
        
        return refined_features
    
    def apply_cross_stripe_attention(self, stripes: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-stripe attention to capture contextual information
        
        Args:
            stripes: List of stripe features, each [B, C, H_s, W]
        
        Returns:
            List of context-aware stripe features
        """
        B = stripes[0].shape[0]
        C = stripes[0].shape[1]
        
        # Pool each stripe to get global representation
        stripe_pools = []
        for stripe in stripes:
            pool = F.adaptive_avg_pool2d(stripe, 1).squeeze(-1).squeeze(-1)  # [B, C]
            stripe_pools.append(pool)
        
        # Stack as sequence [B, num_stripes, C]
        stripe_sequence = torch.stack(stripe_pools, dim=1)
        
        # Apply self-attention
        attended_sequence, _ = self.cross_stripe_attention(
            stripe_sequence, stripe_sequence, stripe_sequence
        )
        
        # Distribute attended features back to stripes
        refined_stripes = []
        for i, stripe in enumerate(stripe):
            # Add attended context
            context = attended_sequence[:, i, :].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            context_broadcasted = context.expand_as(stripe)
            refined_stripe = stripe + context_broadcasted
            refined_stripes.append(refined_stripe)
        
        return refined_stripes
    
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: refine visual features using text guidance
        
        Args:
            visual_features: Input visual features [B, C, H, W]
            text_features: Text features [B, text_feat_dim]
        
        Returns:
            Refined visual features [B, C, H, W]
        """
        B, C, H, W = visual_features.shape
        original_H = H
        
        # Step 1: Partition into horizontal stripes
        stripes = self.partitioner(visual_features)  # [B, num_stripes, C, H_s, W]
        
        # Step 2: Generate text-based filters
        filters = self.filter_generator(text_features)  # List of [B, 1, 1, C]
        
        # Step 3: Refine each stripe
        refined_stripes = []
        for i in range(self.num_stripes):
            stripe_i = stripes[:, i, :, :, :]  # [B, C, H_s, W]
            filter_i = filters[i]  # [B, 1, 1, C]
            refined_stripe_i = self.refine_stripe(
                stripe_i, filter_i, self.refinement_networks[i]
            )
            refined_stripes.append(refined_stripe_i)
        
        # Step 4: Apply cross-stripe attention
        refined_stripes = self.apply_cross_stripe_attention(refined_stripes)
        
        # Step 5: Concatenate stripes back
        refined_features = torch.cat(refined_stripes, dim=2)  # [B, C, H, W]
        
        # Step 6: Resize back to original height if padded
        if refined_features.shape[2] != original_H:
            refined_features = refined_features[:, :, :original_H, :]
        
        return refined_features


class TextGuidedFeatureRefinementModule(nn.Module):
    """
    Complete Text-Guided Feature Refinement Module
    Can be inserted into existing pipelines
    """
    def __init__(self, 
                 visual_feat_dim: int = 2048,
                 text_feat_dim: int = 1024,
                 num_stripes: int = 6,
                 num_stages: int = 2):
        """
        Args:
            visual_feat_dim: Dimension of visual features
            text_feat_dim: Dimension of text features
            num_stripes: Number of horizontal stripes
            num_stages: Number of refinement stages (cascaded)
        """
        super().__init__()
        self.num_stages = num_stages
        
        # Multi-stage refinement
        self.refinement_stages = nn.ModuleList([
            TextGuidedFeatureRefinement(visual_feat_dim, text_feat_dim, num_stripes)
            for _ in range(num_stages)
        ])
        
        # Optional: Feature projection if dimensions don't match
        self.visual_proj = nn.Linear(visual_feat_dim, visual_feat_dim)
        self.text_proj = nn.Linear(text_feat_dim, text_feat_dim)
    
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with multi-stage refinement
        
        Args:
            visual_features: Input visual features [B, C, H, W] or [B, C]
            text_features: Text features [B, text_feat_dim]
        
        Returns:
            refined_features: Refined visual features
            intermediate_features: List of intermediate features from each stage
        """
        # Project features
        visual_features = self.visual_proj(visual_features)
        text_features = self.text_proj(text_features)
        
        # Handle 2D features (reshape to 4D if needed)
        if visual_features.dim() == 2:
            B, C = visual_features.shape
            # Assume square spatial dimension
            H = W = int((C // 16) ** 0.5)  # Heuristic: reshape to reasonable spatial size
            visual_features = visual_features.view(B, -1, H, W)
        
        intermediate_features = []
        current_features = visual_features
        
        # Apply multi-stage refinement
        for stage in self.refinement_stages:
            current_features = stage(current_features, text_features)
            intermediate_features.append(current_features)
        
        # Global average pooling to get final feature vector
        refined_features = F.adaptive_avg_pool2d(current_features, 1)
        refined_features = refined_features.flatten(1)
        
        return refined_features, intermediate_features


def build_text_guided_refinement_module(cfg) -> TextGuidedFeatureRefinementModule:
    """
    Build text-guided refinement module from config
    
    Args:
        cfg: Configuration object
    
    Returns:
        TextGuidedFeatureRefinementModule instance
    """
    visual_feat_dim = cfg.PERSON_SEARCH.REID.get('VISUAL_FEAT_DIM', 2048)
    text_feat_dim = cfg.PERSON_SEARCH.REID.get('TEXT_FEAT_DIM', 1024)
    num_stripes = cfg.PERSON_SEARCH.REID.get('NUM_STRIPES', 6)
    num_stages = cfg.PERSON_SEARCH.REID.get('REFINEMENT_STAGES', 2)
    
    enable_refinement = cfg.PERSON_SEARCH.REID.get('USE_TEXT_GUIDED_REFINEMENT', False)
    
    if not enable_refinement:
        return None
    
    return TextGuidedFeatureRefinementModule(
        visual_feat_dim=visual_feat_dim,
        text_feat_dim=text_feat_dim,
        num_stripes=num_stripes,
        num_stages=num_stages
    )