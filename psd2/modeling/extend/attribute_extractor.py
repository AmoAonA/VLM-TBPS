"""
Attribute Feature Extractor
Extracts biometric and non-biometric features from images
Adapted from DIFFER: Disentangling Identity Features via Semantic Cues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
# codex resume 019cdfd7-5959-7bb3-8c54-c9bd0845007e


class Mlp(nn.Module):
    """
    MLP module for feature projection
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttributeExtractor(nn.Module):
    """
    Extracts biometric and non-biometric attribute features
    """
    def __init__(self, 
                 embed_dim: int,
                 clip_feat_dim: int = 1024,
                 num_nonbio: int = 3,
                 subspace_dim: int = 512,
                 last_layer: str = 'clipFc',
                 clip_loss_type: str = 'constant',
                 reverse_bio: bool = False):
        """
        Args:
            embed_dim: Input feature dimension
            clip_feat_dim: CLIP feature dimension
            num_nonbio: Number of non-biometric attributes
            subspace_dim: Dimension of subspace for MLP head
            last_layer: Type of last layer ('clipFc' or 'clipMLP')
            clip_loss_type: Type of CLIP loss ('constant', 'contrastive', 'sigmoid')
            reverse_bio: Whether to use gradient reversal for adversarial training
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.clip_feat_dim = clip_feat_dim
        self.num_nonbio = num_nonbio
        self.last_layer = last_layer
        self.clip_loss_type = clip_loss_type
        self.reverse_bio = reverse_bio
        
        # Biometric feature extraction head
        if last_layer == 'clipFc':
            self.head_clip_bio = nn.Linear(embed_dim, clip_feat_dim) if clip_feat_dim > 0 else nn.Identity()
        elif last_layer == 'clipMLP':
            self.head_clip_bio = Mlp(
                in_features=embed_dim,
                hidden_features=subspace_dim,
                out_features=clip_feat_dim,
            )
        else:
            raise ValueError(f"Unknown last_layer type: {last_layer}")
        
        # Non-biometric feature extraction heads
        if num_nonbio > 0:
            self.head_clip_nonbio = nn.ModuleList([
                nn.Linear(embed_dim, clip_feat_dim) for _ in range(num_nonbio)
            ])
        
        # Gradient reversal classifiers for adversarial training
        if reverse_bio and num_nonbio > 0:
            from .attribute_loss import GradientReversalClassifier
            
            if last_layer == 'clipFc':
                self.head_clip_bio_reverse = nn.ModuleList([
                    GradientReversalClassifier(
                        in_num=embed_dim,
                        class_num=clip_feat_dim,
                        alpha=1.0
                    ) for _ in range(num_nonbio)
                ])
            elif last_layer == 'clipMLP':
                self.head_clip_bio_reverse = nn.ModuleList([
                    GradientReversalClassifier(
                        in_num=embed_dim,
                        class_num=clip_feat_dim,
                        alpha=1.0
                    ) for _ in range(num_nonbio)
                ])
        
        # Learnable scale and bias for CLIP loss
        if clip_loss_type == 'constant':
            self.logit_scale_bio = 1.0
            self.logit_bias_bio = 0.0
            if num_nonbio > 0:
                self.logit_scale_nonbios = [1.0] * num_nonbio
                self.logit_bias_nonbios = [0.0] * num_nonbio
        elif clip_loss_type == 'contrastive':
            self.logit_scale_bio = nn.Parameter(torch.ones([]) * (1 / 0.07).log())
            self.logit_bias_bio = 0.0
            if num_nonbio > 0:
                self.logit_scale_nonbios = nn.ParameterList([
                    nn.Parameter(torch.ones([]) * (1 / 0.07).log()) 
                    for _ in range(num_nonbio)
                ])
                self.logit_bias_nonbios = [0.0] * num_nonbio
        elif clip_loss_type == 'sigmoid':
            self.logit_scale_bio = nn.Parameter(torch.randn(1))
            self.logit_bias_bio = nn.Parameter(torch.randn(1))
            if num_nonbio > 0:
                self.logit_scale_nonbios = nn.ParameterList([
                    nn.Parameter(torch.randn(1)) for _ in range(num_nonbio)
                ])
                self.logit_bias_nonbios = nn.ParameterList([
                    nn.Parameter(torch.randn(1)) for _ in range(num_nonbio)
                ])
    
    def forward(self, features, return_all_features=False):
        """
        Forward pass
        
        Args:
            features: Input features [B, D]
            return_all_features: Whether to return all features
        
        Returns:
            dict: Dictionary containing extracted features and scores
        """
        output = {}
        
        # Extract biometric features
        bio_features = self.head_clip_bio(features)
        bio_features = F.normalize(bio_features, dim=-1)
        output['clip_bio_score'] = bio_features
        output['clip_bio_scale'] = self.logit_scale_bio
        output['clip_bio_bias'] = self.logit_bias_bio
        
        # Extract non-biometric features
        if self.num_nonbio > 0:
            nonbio_features_list = []
            for i, head in enumerate(self.head_clip_nonbio):
                nonbio_feat = head(features)
                nonbio_feat = F.normalize(nonbio_feat, dim=-1)
                nonbio_features_list.append(nonbio_feat)
            
            nonbio_features = torch.stack(nonbio_features_list, dim=1)  # [B, N, D]
            output['clip_nonbio_score'] = nonbio_features
            output['clip_nonbio_scale'] = self.logit_scale_nonbios
            output['clip_nonbio_bias'] = self.logit_bias_nonbios
            
            if return_all_features:
                output['bio_features'] = bio_features
                output['nonbio_features'] = nonbio_features
            
            # Gradient reversal for adversarial training
            if self.reverse_bio:
                reverse_features_list = []
                for head in self.head_clip_bio_reverse:
                    reverse_feat = head(features)
                    reverse_features_list.append(reverse_feat)
                
                reverse_features = torch.stack(reverse_features_list, dim=1)  # [B, N, D]
                output['clip_bio_reverse_score'] = reverse_features
        
        return output
    
    def get_learnable_params(self):
        """Get learnable parameters"""
        params = []
        params.extend(self.head_clip_bio.parameters())
        if self.num_nonbio > 0:
            for head in self.head_clip_nonbio:
                params.extend(head.parameters())
        
        if self.clip_loss_type != 'constant':
            if isinstance(self.logit_scale_bio, nn.Parameter):
                params.append(self.logit_scale_bio)
            if isinstance(self.logit_bias_bio, nn.Parameter):
                params.append(self.logit_bias_bio)
            
            if self.num_nonbio > 0:
                if isinstance(self.logit_scale_nonbios, nn.ParameterList):
                    params.extend(list(self.logit_scale_nonbios))
                if isinstance(self.logit_bias_nonbios, nn.ParameterList):
                    params.extend(list(self.logit_bias_nonbios))
        
        if self.reverse_bio and self.num_nonbio > 0:
            for head in self.head_clip_bio_reverse:
                params.extend(head.parameters())
        
        return params


class AttributeFeatureLoader:
    """
    Loads pre-computed attribute features from .npz files
    """
    def __init__(self, caption_dir, caption_model='EVA02-CLIP-bigE-14'):
        """
        Args:
            caption_dir: Directory containing caption features
            caption_model: Name of the caption model
        """
        self.caption_dir = caption_dir
        self.caption_model = caption_model
        self.ft_name = f'ft_{caption_model}'
    
    def load_caption_features(self, dataset_dir, bio_index=[0], nonbio_index=[1, 2, 3]):
        """
        Load caption features for a dataset
        
        Args:
            dataset_dir: Directory containing images
            bio_index: Indices of biometric features
            nonbio_index: Indices of non-biometric features
        
        Returns:
            dict: Mapping from image filename to caption features
        """
        import os
        import numpy as np
        
        caption_path = dataset_dir.replace('images', self.caption_dir) + '.npz'
        
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found: {caption_path}")
        
        all_caption = np.load(caption_path, allow_pickle=True)
        all_caption_features = all_caption['data']
        all_caption_files = all_caption['metadata']
        
        ft_nums = int(all_caption_features.shape[0] / all_caption_files.shape[0])
        all_caption_files = list(all_caption_files)
        
        # Create mapping
        caption_feature_map = {}
        for i, filename in enumerate(all_caption_files):
            start_idx = i * ft_nums
            end_idx = (i + 1) * ft_nums
            caption_feature_load = all_caption_features[start_idx:end_idx]
            
            # Select biometric and non-biometric features
            selected_indices = bio_index + nonbio_index
            caption_feature = caption_feature_load[selected_indices]
            
            caption_feature_map[filename] = caption_feature
        
        return caption_feature_map
    
    def get_feature_for_image(self, image_filename, caption_feature_map):
        """
        Get caption features for a specific image
        
        Args:
            image_filename: Image filename (without extension)
            caption_feature_map: Mapping from filename to features
        
        Returns:
            np.ndarray: Caption features [N, D]
        """
        if image_filename not in caption_feature_map:
            # Return zeros if not found
            return np.zeros((4, 1024), dtype=np.float32)
        
        return caption_feature_map[image_filename]