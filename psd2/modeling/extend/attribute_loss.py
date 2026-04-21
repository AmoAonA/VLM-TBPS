"""
Attribute Feature Loss Functions
Adapted from DIFFER: Disentangling Identity Features via Semantic Cues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalClassifier(nn.Module):
    """
    Gradient Reversal Classifier for removing non-biometric features
    """
    def __init__(self, in_num, class_num, weight=None, alpha=1.0):
        super(GradientReversalClassifier, self).__init__()
        self.grl = GradientReversalFunction.apply
        self.alpha = alpha
        self.fc = nn.Linear(in_num, class_num, bias=False)
        if weight is not None:
            self.fc.weight.data = weight.clone()

    def forward(self, x):
        x = self.grl(x, self.alpha)
        x = self.fc(x)
        return x


def clip_contrastive_loss(image_features, text_features, logit_scale=1.0):
    """
    Contrastive loss between image features and text features
    """
    logits_per_image = logit_scale * image_features @ text_features.T
    labels = torch.arange(image_features.shape[0], device=image_features.device)
    loss = F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss, acc


def clip_sigmoid_loss(image_features, text_features, logit_scale=1.0, logit_bias=0.0):
    """
    Sigmoid loss for image-text matching
    """
    logit_scale = logit_scale.type(image_features.type())
    logit_bias = logit_bias.type(image_features.type())
    logits_per_image = logit_scale * image_features @ text_features.T + logit_bias
    labels = 2 * torch.eye(image_features.shape[0], device=image_features.device) - 1
    loss = F.binary_cross_entropy_with_logits(logits_per_image, labels)
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss, acc


def clip_l2_loss(score, logit_scale=torch.zeros(1)):
    """
    L2 loss for image-text matching
    """
    positive_score = score.diag()
    loss = (1 - positive_score).sum() / score.shape[0]
    
    labels = torch.arange(score.shape[0], device=score.device)
    acc = (score.argmax(-1) == labels).sum() / score.shape[0]
    return loss, acc


def attribute_contrastive_loss(image_features, text_features, loss_type='constant', 
                               logit_scale=None, logit_bias=None):
    """
    Attribute contrastive loss with multiple attribute types
    """
    total_loss = 0.0
    total_acc = 0.0
    
    # Handle single attribute type
    if len(image_features.shape) == 2:
        image_features = image_features.unsqueeze(1)
        text_features = text_features.unsqueeze(1)
    
    num_attrs = image_features.shape[1]
    
    for i in range(num_attrs):
        img_feat = image_features[:, i]
        text_feat = text_features[:, i]
        
        if loss_type == 'constant':
            loss, acc = clip_contrastive_loss(img_feat, text_feat, logit_scale=1.0)
        elif loss_type == 'contrastive':
            scale = logit_scale.exp() if logit_scale is not None else 1.0
            loss, acc = clip_contrastive_loss(img_feat, text_feat, logit_scale=scale)
        elif loss_type == 'sigmoid':
            scale = logit_scale if logit_scale is not None else 1.0
            bias = logit_bias if logit_bias is not None else 0.0
            loss, acc = clip_sigmoid_loss(img_feat, text_feat, logit_scale=scale, logit_bias=bias)
        elif loss_type == 'l2':
            score = img_feat @ text_feat.T
            loss, acc = clip_l2_loss(score)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        total_loss += loss
        total_acc += acc
    
    return total_loss / num_attrs, total_acc / num_attrs


def compute_attribute_losses(image_features_bio, image_features_nonbio, 
                            caption_features, cfg):
    """
    Compute attribute-related losses
    
    Args:
        image_features_bio: Image features for biometric attributes [B, D] or [B, N, D]
        image_features_nonbio: Image features for non-biometric attributes [B, M, D]
        caption_features: Text features [B, N+M, D]
        cfg: Configuration object
    
    Returns:
        dict: Dictionary of losses
    """
    losses = {}
    
    # Normalize features
    image_features_bio = F.normalize(image_features_bio, dim=-1)
    if image_features_nonbio is not None:
        image_features_nonbio = F.normalize(image_features_nonbio, dim=-1)
    
    # Split caption features into bio and non-bio
    if len(image_features_bio.shape) == 2:
        num_bio = 1
    else:
        num_bio = image_features_bio.shape[1]
    
    text_features_bio = caption_features[:, :num_bio]
    text_features_bio = F.normalize(text_features_bio, dim=-1)
    
    if image_features_nonbio is not None:
        text_features_nonbio = caption_features[:, num_bio:]
        text_features_nonbio = F.normalize(text_features_nonbio, dim=-1)
    
    # Biometric contrastive loss
    loss_type = cfg.get('CLIP_LOSS_TYPE', 'constant')
    logit_scale = cfg.get('logit_scale_bio', 1.0)
    logit_bias = cfg.get('logit_bias_bio', 0.0)
    
    loss_bio, acc_bio = attribute_contrastive_loss(
        image_features_bio, text_features_bio, 
        loss_type=loss_type, logit_scale=logit_scale, logit_bias=logit_bias
    )
    losses['loss_clip_bio'] = loss_bio
    losses['acc_clip_bio'] = acc_bio
    
    # Non-biometric contrastive loss
    if image_features_nonbio is not None:
        loss_nonbio, acc_nonbio = attribute_contrastive_loss(
            image_features_nonbio, text_features_nonbio,
            loss_type=loss_type, logit_scale=logit_scale, logit_bias=logit_bias
        )
        losses['loss_clip_nonbio'] = loss_nonbio
        losses['acc_clip_nonbio'] = acc_nonbio
    
    return losses