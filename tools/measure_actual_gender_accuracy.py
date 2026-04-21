#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append("./")

import torch
from torchvision.ops import box_iou

from psd2.checkpoint import DetectionCheckpointer
from psd2.config import get_cfg

from train_ps_net import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure actual query/gallery gender-label accuracy used by evaluator."
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--eval-dir", required=True, help="Directory that contains inference/_gallery_gt_inf.pt")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--query-dataset", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.weights
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_gender_from_attrs(attrs):
    if isinstance(attrs, list):
        if not attrs:
            return None
        attrs = attrs[0]
    if not isinstance(attrs, dict):
        return None
    return _to_int(attrs.get("gender", None))


def measure_query_gender_accuracy(cfg, dataset_name, eval_dir):
    query_cache_path = os.path.join(eval_dir, "inference", "_query_inf.pt")
    if os.path.exists(query_cache_path):
        return measure_query_gender_accuracy_from_cache(query_cache_path, dataset_name)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    model.eval()

    loader = Trainer.build_test_loader(cfg, dataset_name)
    total = 0
    valid_gt = 0
    valid_pred = 0
    correct = 0
    unknown_pred = 0

    with torch.no_grad():
        for inputs in loader:
            outputs = model(inputs)
            for in_dict, pred in zip(inputs, outputs):
                query_instances = in_dict["query"]["instances"]
                gt_gender = _get_gender_from_attrs(getattr(query_instances, "gt_attributes", None))
                if gt_gender is not None and gt_gender >= 0:
                    valid_gt += 1

                total += 1
                pred = pred.to(torch.device("cpu"))
                if not pred.has("cerberus_group_labels"):
                    unknown_pred += 1
                    continue
                pred_gender = int(pred.cerberus_group_labels[0, 0].item())
                if pred_gender < 0:
                    unknown_pred += 1
                    continue
                valid_pred += 1
                if gt_gender is not None and gt_gender >= 0 and pred_gender == gt_gender:
                    correct += 1

    return {
        "dataset": dataset_name,
        "source": "model_forward",
        "total_queries": total,
        "valid_gt_queries": valid_gt,
        "valid_pred_queries": valid_pred,
        "unknown_pred_queries": unknown_pred,
        "acc_on_valid_gt": (correct / valid_gt) if valid_gt > 0 else 0.0,
        "acc_on_valid_pred": (correct / valid_pred) if valid_pred > 0 else 0.0,
    }


def measure_query_gender_accuracy_from_cache(query_cache_path, dataset_name):
    saved = torch.load(query_cache_path, map_location="cpu")
    payload = saved.get("payload", [])

    total = 0
    valid_gt = 0
    valid_pred = 0
    correct = 0
    unknown_pred = 0

    for batch in payload:
        for item in batch:
            if len(item) == 2:
                q_gt_instances, q_pred_instances = item
            elif len(item) == 3:
                q_gt_instances, _, q_pred_instances = item
            else:
                continue

            gt_gender = _get_gender_from_attrs(getattr(q_gt_instances, "gt_attributes", None))
            if gt_gender is not None and gt_gender >= 0:
                valid_gt += 1

            total += 1
            if not q_pred_instances.has("cerberus_group_labels"):
                unknown_pred += 1
                continue
            pred_gender = int(q_pred_instances.cerberus_group_labels[0, 0].item())
            if pred_gender < 0:
                unknown_pred += 1
                continue
            valid_pred += 1
            if gt_gender is not None and gt_gender >= 0 and pred_gender == gt_gender:
                correct += 1

    return {
        "dataset": dataset_name,
        "source": "query_cache",
        "total_queries": total,
        "valid_gt_queries": valid_gt,
        "valid_pred_queries": valid_pred,
        "unknown_pred_queries": unknown_pred,
        "acc_on_valid_gt": (correct / valid_gt) if valid_gt > 0 else 0.0,
        "acc_on_valid_pred": (correct / valid_pred) if valid_pred > 0 else 0.0,
    }


def _match_det_to_gt(det_box, gt_boxes):
    if gt_boxes.numel() == 0:
        return -1
    ious = box_iou(det_box[None, :], gt_boxes).squeeze(0)
    best_iou, best_idx = torch.max(ious, dim=0)
    gt = gt_boxes[best_idx]
    w = gt[2] - gt[0]
    h = gt[3] - gt[1]
    thr = min(0.5, (w * h * 1.0 / ((w + 10) * (h + 10))).item())
    if best_iou.item() >= thr:
        return int(best_idx.item())
    return -1


def measure_gallery_gender_accuracy(eval_dir, score_thresh):
    cache_path = os.path.join(eval_dir, "inference", "_gallery_gt_inf.pt")
    saved = torch.load(cache_path, map_location="cpu")

    gts = saved["gts"]
    gt_attrs = saved.get("gt_attrs", {})
    infs = saved["infs"]
    inf_group_labels = saved.get("inf_group_labels", {})

    total_det = 0
    valid_pred = 0
    matched_det = 0
    matched_with_gt_gender = 0
    correct = 0
    unknown_pred = 0

    for image_id, labels in inf_group_labels.items():
        if labels is None or len(labels) == 0:
            continue
        if image_id not in infs or image_id not in gts:
            continue

        dets = infs[image_id]
        mask = dets[:, 4] >= score_thresh
        det_boxes = dets[mask][:, :4]
        group_labels = labels[mask]
        if det_boxes.numel() == 0 or group_labels.numel() == 0:
            continue

        gt_boxes = gts[image_id][:, :4]
        attrs_list = gt_attrs.get(image_id, [{} for _ in range(gt_boxes.shape[0])])

        for det_box, group_label in zip(det_boxes, group_labels):
            total_det += 1
            pred_gender = int(group_label[0].item())
            if pred_gender < 0:
                unknown_pred += 1
                continue
            valid_pred += 1
            gt_idx = _match_det_to_gt(det_box, gt_boxes)
            if gt_idx < 0:
                continue
            matched_det += 1
            gt_gender = _get_gender_from_attrs(attrs_list[gt_idx])
            if gt_gender is None or gt_gender < 0:
                continue
            matched_with_gt_gender += 1
            if pred_gender == gt_gender:
                correct += 1

    return {
        "score_thresh": score_thresh,
        "total_scored_dets": total_det,
        "valid_pred_dets": valid_pred,
        "unknown_pred_dets": unknown_pred,
        "matched_dets": matched_det,
        "matched_with_gt_gender": matched_with_gt_gender,
        "acc_on_matched_gt_gender": (
            correct / matched_with_gt_gender if matched_with_gt_gender > 0 else 0.0
        ),
        "acc_on_valid_pred": (correct / valid_pred) if valid_pred > 0 else 0.0,
    }


def main():
    args = parse_args()
    cfg = setup_cfg(args)
    query_dataset = args.query_dataset
    if query_dataset is None:
        query_dataset = next(name for name in cfg.DATASETS.TEST if "Query" in name)

    query_stats = measure_query_gender_accuracy(cfg, query_dataset, args.eval_dir)
    gallery_stats = measure_gallery_gender_accuracy(args.eval_dir, args.score_thresh)

    print("=== Query Gender Accuracy ===")
    for key, value in query_stats.items():
        print(f"{key}: {value}")

    print("=== Gallery Gender Accuracy ===")
    for key, value in gallery_stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
