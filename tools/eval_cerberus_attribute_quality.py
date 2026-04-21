import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from psd2.modeling.extend.cerberus_semantic_id import CerberusPrototypeBank
from psd2.data.datasets.prw_tbps import get_resort_id


UNKNOWN_LABEL_TOKENS = ("unknown", "other", "n/a", "unspecified")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline attribute/group quality analysis from _gallery_gt_inf.pt."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing _gallery_gt_inf.pt produced by eval-only run.",
    )
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--semantic-dim", type=int, default=512)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Only keep detections whose saved score is >= this threshold.",
    )
    parser.add_argument(
        "--gallery-attrs-json",
        default="",
        help="Optional PRW test JSON used to recover gallery GT attributes when cache lacks gt_attrs.",
    )
    return parser.parse_args()


def is_unknown_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    return any(token in normalized for token in UNKNOWN_LABEL_TOKENS)


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint structure: {type(checkpoint)}")


def infer_use_carry_group(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(key.endswith("prototype_bank.prototypes_carry") or key.endswith("prototypes_carry") for key in state_dict)


def load_prototype_bank(
    checkpoint_path: Path, schema_path: str, semantic_dim: int
) -> CerberusPrototypeBank:
    state_dict = load_checkpoint_state(checkpoint_path)
    bank = CerberusPrototypeBank(
        schema_path=schema_path,
        semantic_dim=semantic_dim,
        use_carry_group=infer_use_carry_group(state_dict),
    )
    missing = []
    for group_name in bank.group_names:
        target = f"prototypes_{group_name}"
        candidates = [
            f"cerberus_branch.prototype_bank.{target}",
            f"module.cerberus_branch.prototype_bank.{target}",
            f"prototype_bank.{target}",
            target,
        ]
        tensor = None
        for key in candidates:
            if key in state_dict:
                tensor = state_dict[key]
                break
        if tensor is None:
            missing.append(target)
            continue
        getattr(bank, target).data.copy_(tensor.float())
    if missing:
        raise KeyError(f"Missing prototype tensors in checkpoint: {missing}")
    return bank


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(1e-6)


def greedy_match(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh: float) -> List[Tuple[int, int, float]]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []
    ious = box_iou(pred_boxes, gt_boxes)
    candidates: List[Tuple[float, int, int]] = []
    for p_idx in range(ious.shape[0]):
        for g_idx in range(ious.shape[1]):
            iou_val = float(ious[p_idx, g_idx].item())
            if iou_val >= iou_thresh:
                candidates.append((iou_val, p_idx, g_idx))
    candidates.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    matches = []
    for iou_val, p_idx, g_idx in candidates:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        used_pred.add(p_idx)
        used_gt.add(g_idx)
        matches.append((p_idx, g_idx, iou_val))
    return matches


def decode_group_index(bank: CerberusPrototypeBank, group_name: str, index: int) -> Dict[str, int]:
    attrs = bank.attribute_groups[group_name]
    dims = bank.group_value_sizes[group_name]
    values = [0] * len(attrs)
    temp = int(index)
    for pos in range(len(attrs) - 1, -1, -1):
        count = dims[pos]
        values[pos] = temp % count
        temp //= count
    return {attr_name: value for attr_name, value in zip(attrs, values)}


def init_metric_row() -> Dict[str, float]:
    return {"total": 0, "valid": 0, "correct": 0, "known_total": 0, "known_correct": 0}


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def build_prw_gallery_attr_lookup(json_path: str) -> Dict[str, Dict[int, Dict[str, int]]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw_items = json.load(handle)
    lookup: Dict[str, Dict[int, Dict[str, int]]] = {}
    for item in raw_items:
        image_id = item.get("pic_path")
        pid = item.get("id")
        attrs = item.get("attributes", {})
        if image_id is None or pid is None:
            continue
        image_id = f"{image_id}.jpg" if not str(image_id).endswith(".jpg") else str(image_id)
        remapped_pid = get_resort_id(int(pid))
        lookup.setdefault(image_id, {})[remapped_pid] = attrs
    return lookup


def maybe_recover_gt_attrs(
    saved: Dict[str, object], gallery_attrs_json: str
) -> Dict[str, List[Dict[str, int]]]:
    gt_attrs = saved.get("gt_attrs")
    if isinstance(gt_attrs, dict) and len(gt_attrs) > 0:
        return gt_attrs
    if not gallery_attrs_json:
        return {}

    lookup = build_prw_gallery_attr_lookup(gallery_attrs_json)
    recovered: Dict[str, List[Dict[str, int]]] = {}
    for img_id, gt_tensor in saved["gts"].items():
        per_image_lookup = lookup.get(img_id, {})
        attrs = []
        for row in gt_tensor:
            pid = int(row[4].item())
            attrs.append(per_image_lookup.get(pid, {}))
        recovered[img_id] = attrs
    return recovered


def main():
    args = parse_args()
    eval_path = Path(args.eval_dir) / "_gallery_gt_inf.pt"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval cache: {eval_path}")

    bank = load_prototype_bank(Path(args.checkpoint), args.schema_path, args.semantic_dim)
    saved = torch.load(eval_path, map_location="cpu")

    gts = saved["gts"]
    gt_attrs = maybe_recover_gt_attrs(saved, args.gallery_attrs_json)
    infs = saved["infs"]
    inf_group_feats = saved.get("inf_group_feats", {})
    inf_group_valids = saved.get("inf_group_valids", {})

    group_metrics = {group_name: init_metric_row() for group_name in bank.group_names}
    attr_metrics = {
        attr_name: init_metric_row()
        for group_name in bank.group_names
        for attr_name in bank.attribute_groups[group_name]
    }

    matched_gt = 0
    total_gt = 0
    image_matches = 0

    for img_id, gt_tensor in gts.items():
        gt_boxes = gt_tensor[:, :4].float()
        attrs_list = gt_attrs.get(img_id, [{} for _ in range(len(gt_boxes))])
        total_gt += len(gt_boxes)

        pred_all = infs.get(img_id)
        group_feats_all = inf_group_feats.get(img_id)
        group_valid_all = inf_group_valids.get(img_id)
        if pred_all is None or group_feats_all is None or group_valid_all is None:
            continue

        score_mask = pred_all[:, 4] >= float(args.score_thresh)
        pred_all = pred_all[score_mask]
        group_feats_all = group_feats_all[score_mask]
        group_valid_all = group_valid_all[score_mask]
        if pred_all.shape[0] == 0:
            continue

        pred_boxes = pred_all[:, :4].float()
        matches = greedy_match(pred_boxes, gt_boxes, args.iou_thresh)
        if matches:
            image_matches += 1

        for pred_idx, gt_idx, _ in matches:
            matched_gt += 1
            attr_dict = attrs_list[gt_idx] if gt_idx < len(attrs_list) else {}
            pred_group_feats = group_feats_all[pred_idx]
            pred_group_valid = group_valid_all[pred_idx].float()

            for group_offset, group_name in enumerate(bank.group_names):
                target_group_idx = bank.get_group_index(group_name, attr_dict)
                if target_group_idx < 0:
                    continue

                group_metrics[group_name]["total"] += 1
                pred_is_valid = group_offset < pred_group_valid.numel() and pred_group_valid[group_offset].item() > 0
                if not pred_is_valid:
                    continue

                group_metrics[group_name]["valid"] += 1
                prototypes = F.normalize(bank.get_group_prototypes(group_name).float(), dim=-1)
                pred_embedding = F.normalize(pred_group_feats[group_offset].float(), dim=-1)
                pred_group_idx = int(torch.argmax(pred_embedding @ prototypes.t()).item())
                if pred_group_idx == target_group_idx:
                    group_metrics[group_name]["correct"] += 1

                pred_attr_values = decode_group_index(bank, group_name, pred_group_idx)
                gt_attr_values = decode_group_index(bank, group_name, target_group_idx)
                for attr_name in bank.attribute_groups[group_name]:
                    attr_metrics[attr_name]["total"] += 1
                    attr_metrics[attr_name]["valid"] += 1
                    if pred_attr_values[attr_name] == gt_attr_values[attr_name]:
                        attr_metrics[attr_name]["correct"] += 1

                    gt_label = bank.get_attribute_text(attr_name, gt_attr_values[attr_name])
                    if not is_unknown_label(gt_label):
                        attr_metrics[attr_name]["known_total"] += 1
                        if pred_attr_values[attr_name] == gt_attr_values[attr_name]:
                            attr_metrics[attr_name]["known_correct"] += 1

                representative_attr = bank.attribute_groups[group_name][0]
                rep_label = bank.get_attribute_text(representative_attr, gt_attr_values[representative_attr])
                if not is_unknown_label(rep_label):
                    group_metrics[group_name]["known_total"] += 1
                    if pred_group_idx == target_group_idx:
                        group_metrics[group_name]["known_correct"] += 1

    print("Attribute Quality Report")
    print(f"eval_cache: {eval_path}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"iou_thresh: {args.iou_thresh:.2f}, score_thresh: {args.score_thresh:.2f}")
    print(
        f"matched_gt: {matched_gt}/{total_gt} "
        f"({safe_div(matched_gt, total_gt):.4f}), images_with_match: {image_matches}"
    )
    print("")
    print("Group Metrics")
    for group_name in bank.group_names:
        metric = group_metrics[group_name]
        coverage = safe_div(metric["valid"], metric["total"])
        acc_valid = safe_div(metric["correct"], metric["valid"])
        acc_all = safe_div(metric["correct"], metric["total"])
        acc_known = safe_div(metric["known_correct"], metric["known_total"])
        print(
            f"{group_name:>8s}  total={int(metric['total']):5d}  "
            f"valid={int(metric['valid']):5d}  coverage={coverage:.4f}  "
            f"acc_valid={acc_valid:.4f}  acc_all={acc_all:.4f}  acc_known={acc_known:.4f}"
        )
    print("")
    print("Attribute Metrics")
    for attr_name, metric in attr_metrics.items():
        if metric["total"] == 0:
            continue
        acc_valid = safe_div(metric["correct"], metric["valid"])
        acc_all = safe_div(metric["correct"], metric["total"])
        acc_known = safe_div(metric["known_correct"], metric["known_total"])
        print(
            f"{attr_name:>12s}  total={int(metric['total']):5d}  "
            f"acc_valid={acc_valid:.4f}  acc_all={acc_all:.4f}  acc_known={acc_known:.4f}"
        )
    print("")
    if "gender" in attr_metrics and attr_metrics["gender"]["known_total"] > 0:
        gender_known_acc = safe_div(
            attr_metrics["gender"]["known_correct"], attr_metrics["gender"]["known_total"]
        )
        print(
            f"Gender hard-filter survival ceiling on matched known-gender positives: "
            f"{gender_known_acc:.4f}"
        )


if __name__ == "__main__":
    main()
