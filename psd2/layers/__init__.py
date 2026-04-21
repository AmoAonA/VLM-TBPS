# Copyright (c) Facebook, Inc. and its affiliates.

# TODO figure out _C error


def _missing_detectron2_op(name, import_error):
    class _MissingDetectron2Op:  # pragma: no cover - only used on environments without _C
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} requires a working detectron2 C++/CUDA extension"
            ) from import_error

    return _MissingDetectron2Op


try:
    from .deform_conv import DeformConv, ModulatedDeformConv, DeformConvPack
except ImportError as _deform_conv_error:
    DeformConv = _missing_detectron2_op("DeformConv", _deform_conv_error)
    ModulatedDeformConv = _missing_detectron2_op(
        "ModulatedDeformConv", _deform_conv_error
    )
    DeformConvPack = _missing_detectron2_op("DeformConvPack", _deform_conv_error)
from .mask_ops import paste_masks_in_image
from .nms import batched_nms, batched_nms_rotated, nms, nms_rotated
from .roi_align import ROIAlign, roi_align

try:
    from .roi_align_rotated import ROIAlignRotated, roi_align_rotated
except ImportError as _roi_align_rotated_error:
    ROIAlignRotated = _missing_detectron2_op(
        "ROIAlignRotated", _roi_align_rotated_error
    )

    def roi_align_rotated(*args, **kwargs):  # pragma: no cover - runtime guard only
        raise ImportError(
            "roi_align_rotated requires a working detectron2 C++/CUDA extension"
        ) from _roi_align_rotated_error
from .blocks import CNNBlockBase, DepthwiseSeparableConv2d
from .aspp import ASPP
from .batch_norm import (
    FrozenBatchNorm2d,
    get_norm,
    get_norm1d,
    NaiveSyncBatchNorm,
    CycleBatchNormList,
)
from .wrappers import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    cat,
    interpolate,
    Linear,
    nonzero_tuple,
    cross_entropy,
    move_device_like,
)
from .shape_spec import ShapeSpec
from .drop import DropPath, DropBlock2d
import torch.nn as nn
import copy
from .losses import ciou_loss, diou_loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


__all__ = [k for k in globals().keys() if not k.startswith("_")]
