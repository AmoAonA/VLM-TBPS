import json
import logging
from os.path import exists

import torch
from torchvision.ops.boxes import box_iou

from .query_evaluator import QueryEvaluator

logger = logging.getLogger(__name__)


class APSQueryEvaluator(QueryEvaluator):
    group_names = ("gender", "hair", "top", "pants", "shoes")
    attribute_groups = {
        "gender": ["gender"],
        "hair": ["hair_color", "hair_length"],
        "top": ["top_color", "top_style"],
        "pants": ["pants_color", "pants_style"],
        "shoes": ["shoes_color", "shoes_style"],
    }

    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        schema_path,
        aps_match_mode="group_exact",
        s_threds=None,
        **kwargs,
    ):
        if s_threds is None:
            s_threds = [0.05, 0.2, 0.5, 0.7]
        super().__init__(dataset_name, distributed, output_dir, s_threds, **kwargs)
        self.aps_match_mode = str(aps_match_mode).lower()
        self.schema_path = schema_path
        self.group_value_sizes = self._load_group_value_sizes(schema_path)
        if len(self.gt_attrs) == 0:
            raise ValueError(
                "APS evaluation requires gallery GT attributes in _gallery_gt_inf.pt. "
                "Please rerun gallery inference with the updated evaluator first."
            )

    @staticmethod
    def _maybe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _load_group_value_sizes(self, schema_path):
        if not schema_path or not exists(schema_path):
            raise FileNotFoundError(f"APS schema path not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as handle:
            schema = json.load(handle)
        schema_dims = {}
        for item in schema:
            mapping = item["mapping"]
            if isinstance(mapping, dict):
                schema_dims[item["attribute"]] = len(mapping)
            elif isinstance(mapping, list):
                schema_dims[item["attribute"]] = len(mapping)
            else:
                raise TypeError(f"Unsupported mapping type for {item['attribute']}: {type(mapping)}")

        group_sizes = {}
        for group_name, attrs in self.attribute_groups.items():
            group_sizes[group_name] = [schema_dims[attr] for attr in attrs]
        return group_sizes

    def _extract_query_attributes(self, q_instances):
        if not hasattr(q_instances, "gt_attributes"):
            return {}
        attrs = q_instances.gt_attributes
        if isinstance(attrs, list) and len(attrs) > 0:
            return attrs[0]
        if isinstance(attrs, dict):
            return attrs
        return {}

    def _group_index(self, group_name, attributes):
        if not isinstance(attributes, dict) or len(attributes) == 0:
            return -1

        index = 0
        for attr, count in zip(
            self.attribute_groups[group_name], self.group_value_sizes[group_name]
        ):
            value = attributes.get(attr, -1)
            value = self._maybe_int(value)
            if value is None or value < 0:
                return -1
            value = min(value, count - 1)
            index = index * count + value
        return index

    def _count_valid_groups(self, attributes):
        return sum(self._group_index(group_name, attributes) >= 0 for group_name in self.group_names)

    def _match_attributes(self, query_attributes, gallery_attributes):
        if self.aps_match_mode != "group_exact":
            raise ValueError(f"Unsupported APS match mode: {self.aps_match_mode}")

        valid_groups = 0
        for group_name in self.group_names:
            q_index = self._group_index(group_name, query_attributes)
            if q_index < 0:
                continue
            valid_groups += 1
            if self._group_index(group_name, gallery_attributes) != q_index:
                return False
        return valid_groups > 0

    def _get_positive_gt_boxes(self, img_id, query_attributes):
        gt_boxes = self._get_gt_boxs(img_id)
        gt_attrs = self.gt_attrs.get(img_id, [{} for _ in range(len(gt_boxes))])
        matched = []
        for box, attr in zip(gt_boxes, gt_attrs):
            if self._match_attributes(query_attributes, attr):
                matched.append(box)
        if len(matched) == 0:
            return gt_boxes.new_zeros((0, 4))
        return torch.stack(matched, dim=0)

    @staticmethod
    def _match_detection_to_gt(roi, gt_boxes):
        if gt_boxes.numel() == 0:
            return False
        ious = box_iou(roi[None, :], gt_boxes).squeeze(0)
        widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        thresholds = torch.minimum(
            torch.full_like(widths, 0.5),
            (widths * heights) / ((widths + 10.0) * (heights + 10.0)),
        )
        return bool((ious >= thresholds).any().item())
