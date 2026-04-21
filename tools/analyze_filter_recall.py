#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append("./")

import torch
from torchvision.ops import box_iou


GROUP_INDEX = {
    "gender": 0,
    "hair": 1,
    "top": 2,
    "pants": 3,
    "shoes": 4,
    "carry": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline recall analysis for label/embedding attribute filters."
    )
    parser.add_argument("--gallery-file", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--embedding-threshold", type=float, default=0.3)
    parser.add_argument("--group", default="gender")
    parser.add_argument("--max-queries", type=int, default=0, help="0 means use all cached queries")
    parser.add_argument(
        "--filter-mode",
        default="both",
        choices=["both", "label", "embedding"],
        help="Run only label filter, only embedding filter, or both",
    )
    return parser.parse_args()


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_attr_value(attrs, key):
    if isinstance(attrs, list):
        if not attrs:
            return None
        attrs = attrs[0]
    if not isinstance(attrs, dict):
        return None
    return _to_int(attrs.get(key))


def _safe_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _iou_thresh(gt_box):
    w = gt_box[2] - gt_box[0]
    h = gt_box[3] - gt_box[1]
    return min(0.5, (w * h * 1.0 / ((w + 10) * (h + 10))).item())


def _matched_positive_indices(det_boxes, gt_boxes):
    if det_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []
    matched = []
    for gt_box in gt_boxes:
        ious = box_iou(det_boxes, gt_box[None, :]).squeeze(1)
        thr = _iou_thresh(gt_box)
        keep = torch.nonzero(ious >= thr, as_tuple=False).flatten().tolist()
        matched.extend(keep)
    return sorted(set(matched))


def _flatten_queries(query_payload):
    flat = []
    for batch in query_payload:
        for item in batch:
            if len(item) == 2:
                q_gt, q_pred = item
            elif len(item) == 3:
                q_gt, _, q_pred = item
            else:
                continue
            flat.append((q_gt, q_pred))
    return flat


def analyze(args):
    gallery_saved = torch.load(args.gallery_file, map_location="cpu")
    query_saved = torch.load(args.query_file, map_location="cpu")

    gts = gallery_saved["gts"]
    infs = gallery_saved["infs"]
    inf_group_feats = gallery_saved.get("inf_group_feats", {})
    inf_group_valids = gallery_saved.get("inf_group_valids", {})
    inf_group_labels = gallery_saved.get("inf_group_labels", {})

    queries = _flatten_queries(query_saved["payload"])
    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    group_idx = GROUP_INDEX[args.group]
    attr_key = "gender" if args.group == "gender" else args.group

    run_label = args.filter_mode in {"both", "label"}
    run_embedding = args.filter_mode in {"both", "embedding"}

    label_total_gt = 0
    label_before_found = 0
    label_after_found = 0
    label_queries_all_kept = 0
    label_queries_hurt = 0
    label_avg_retain = []

    emb_total_gt = 0
    emb_before_found = 0
    emb_after_found = 0
    emb_queries_all_kept = 0
    emb_queries_hurt = 0
    emb_avg_retain = []

    evaluated_queries = 0

    for q_gt, q_pred in queries:
        q_imgid = q_gt.image_id
        q_pid = int(q_gt.gt_pids.item())
        q_gender = _get_attr_value(getattr(q_gt, "gt_attributes", None), attr_key)
        if q_gender is None or q_gender < 0:
            continue
        if run_embedding and not q_pred.has("cerberus_group_feats"):
            continue

        evaluated_queries += 1
        query_group_feat = None
        if run_embedding:
            query_group_feat = q_pred.cerberus_group_feats[0, group_idx].view(1, -1)
            query_group_feat = _safe_normalize(query_group_feat)

        label_found_before_q = 0
        label_found_after_q = 0
        emb_found_before_q = 0
        emb_found_after_q = 0
        gt_images_for_q = 0

        for image_id, gt_tensor in gts.items():
            if image_id == q_imgid:
                continue
            gt_ids = gt_tensor[:, 4].long()
            same_pid = gt_ids == q_pid
            if not same_pid.any():
                continue

            gt_images_for_q += 1
            gt_boxes = gt_tensor[same_pid][:, :4]

            if image_id not in infs:
                continue
            dets = infs[image_id]
            score_mask = dets[:, 4] >= args.score_thresh
            det_boxes = dets[score_mask][:, :4]
            if det_boxes.numel() == 0:
                continue

            positive_indices = _matched_positive_indices(det_boxes, gt_boxes)
            if not positive_indices:
                continue

            if run_label:
                label_found_before_q += 1
            if run_embedding:
                emb_found_before_q += 1

            keep_label = torch.zeros(det_boxes.shape[0], dtype=torch.bool)
            if run_label and image_id in inf_group_labels:
                labels = inf_group_labels[image_id][score_mask]
                if labels.numel() > 0 and labels.shape[1] > group_idx:
                    if args.group == "gender":
                        # Unknown/other is neutral rather than conflicting.
                        gallery_gender = labels[:, group_idx]
                        known_mask = (gallery_gender == 0) | (gallery_gender == 1)
                        keep_label = (~known_mask) | (gallery_gender == q_gender)
                    else:
                        keep_label = labels[:, group_idx] == q_gender

            keep_emb = torch.zeros(det_boxes.shape[0], dtype=torch.bool)
            if run_embedding and image_id in inf_group_feats:
                group_feats = inf_group_feats[image_id][score_mask]
                if group_feats.numel() > 0 and group_feats.shape[1] > group_idx:
                    gallery_feat = _safe_normalize(group_feats[:, group_idx])
                    sims = (gallery_feat * query_group_feat).sum(dim=1)
                    keep_emb = sims > args.embedding_threshold
                    if image_id in inf_group_valids:
                        valids = inf_group_valids[image_id][score_mask]
                        if valids.numel() > 0 and valids.shape[1] > group_idx:
                            keep_emb = keep_emb & (valids[:, group_idx] > 0)

            if run_label and any(keep_label[idx].item() for idx in positive_indices):
                label_found_after_q += 1
            if run_embedding and any(keep_emb[idx].item() for idx in positive_indices):
                emb_found_after_q += 1

        if gt_images_for_q == 0:
            continue

        if run_label:
            label_total_gt += gt_images_for_q
            label_before_found += label_found_before_q
            label_after_found += label_found_after_q
            label_avg_retain.append(label_found_after_q / max(label_found_before_q, 1))
            if label_found_after_q == label_found_before_q:
                label_queries_all_kept += 1
            if label_found_after_q < label_found_before_q:
                label_queries_hurt += 1

        if run_embedding:
            emb_total_gt += gt_images_for_q
            emb_before_found += emb_found_before_q
            emb_after_found += emb_found_after_q
            emb_avg_retain.append(emb_found_after_q / max(emb_found_before_q, 1))
            if emb_found_after_q == emb_found_before_q:
                emb_queries_all_kept += 1
            if emb_found_after_q < emb_found_before_q:
                emb_queries_hurt += 1

    print("=== Filter Recall Analysis ===")
    print("group:", args.group)
    print("score_thresh:", args.score_thresh)
    print("embedding_threshold:", args.embedding_threshold)
    print("max_queries:", args.max_queries if args.max_queries > 0 else "all")
    print("evaluated_queries:", evaluated_queries)
    if run_label:
        print("--- label filter ---")
        print("total_gt_images:", label_total_gt)
        print("before_found:", label_before_found)
        print("after_found:", label_after_found)
        print("recall_before:", label_before_found / label_total_gt if label_total_gt else 0.0)
        print("recall_after:", label_after_found / label_total_gt if label_total_gt else 0.0)
        print("positive_keep_ratio:", label_after_found / label_before_found if label_before_found else 0.0)
        print("queries_all_kept_ratio:", label_queries_all_kept / evaluated_queries if evaluated_queries else 0.0)
        print("queries_hurt_ratio:", label_queries_hurt / evaluated_queries if evaluated_queries else 0.0)
        print("avg_query_keep_ratio:", sum(label_avg_retain) / len(label_avg_retain) if label_avg_retain else 0.0)
    if run_embedding:
        print("--- embedding filter ---")
        print("total_gt_images:", emb_total_gt)
        print("before_found:", emb_before_found)
        print("after_found:", emb_after_found)
        print("recall_before:", emb_before_found / emb_total_gt if emb_total_gt else 0.0)
        print("recall_after:", emb_after_found / emb_total_gt if emb_total_gt else 0.0)
        print("positive_keep_ratio:", emb_after_found / emb_before_found if emb_before_found else 0.0)
        print("queries_all_kept_ratio:", emb_queries_all_kept / evaluated_queries if evaluated_queries else 0.0)
        print("queries_hurt_ratio:", emb_queries_hurt / evaluated_queries if evaluated_queries else 0.0)
        print("avg_query_keep_ratio:", sum(emb_avg_retain) / len(emb_avg_retain) if emb_avg_retain else 0.0)


def main():
    args = parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
