#!/usr/bin/env python
import argparse
import os
import re
import sys

sys.path.append("./")

import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops.boxes import box_iou


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate person search metrics directly from saved query/gallery cache."
    )
    parser.add_argument("--gallery-file", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--dataset", default="prw", choices=["prw", "cuhk"])
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--fusion-mode", default="global", choices=["global", "avg"])
    parser.add_argument("--avg-global-weight", type=float, default=1.0)
    parser.add_argument(
        "--avg-part-weights",
        type=float,
        nargs="*",
        default=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--cerberus-group-score-weight",
        type=float,
        default=None,
        help="Override gallery cache cerberus_group_score_weight if needed.",
    )
    parser.add_argument("--gender-label-gate", action="store_true")
    parser.add_argument(
        "--gender-filter-mode",
        default="none",
        choices=["none", "hard", "soft", "topk_hard", "selective_hard"],
    )
    parser.add_argument("--gender-soft-penalty", type=float, default=0.15)
    parser.add_argument("--gender-topk", type=int, default=200)
    parser.add_argument("--gender-selective-min-keep-ratio", type=float, default=0.45)
    parser.add_argument("--gender-selective-min-known-ratio", type=float, default=0.30)
    parser.add_argument("--query-selective-filter", action="store_true")
    parser.add_argument("--query-selective-shortlist-topk", type=int, default=300)
    parser.add_argument(
        "--query-selective-gender-min-keep-ratio", type=float, default=0.45
    )
    parser.add_argument(
        "--query-selective-gender-min-known-ratio", type=float, default=0.30
    )
    parser.add_argument("--query-selective-top-penalty", type=float, default=0.05)
    parser.add_argument("--query-selective-pants-penalty", type=float, default=0.05)
    parser.add_argument("--topk", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--full-score-matrix", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto/cpu/cuda/cuda:0. Similarity computation runs on this device.",
    )
    return parser.parse_args()


def _resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _get_img_cid(img_id):
    matched = re.match(r"c(\d+)s.*", img_id)
    if matched is None:
        return -1
    return int(matched.groups()[0])


def _match_part_weights(weights, target_dim, device):
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if target_dim <= 0:
        return weights[:0]
    if weights.numel() == target_dim:
        return weights
    if weights.numel() > target_dim:
        return weights[:target_dim]
    default_value = float(weights[-1].item()) if weights.numel() > 0 else 1.0
    padding = weights.new_full((target_dim - weights.numel(),), default_value)
    return torch.cat([weights, padding], dim=0)


def _normalize_query_feature_shapes(
    feat_q, group_feats_q=None, group_valid_q=None, group_labels_q=None
):
    if feat_q is not None and feat_q.dim() > 1:
        feat_q = feat_q[0]
    if group_feats_q is not None and group_feats_q.dim() > 2:
        group_feats_q = group_feats_q[0]
    if group_valid_q is not None and group_valid_q.dim() > 1:
        group_valid_q = group_valid_q[0]
    if group_labels_q is not None and group_labels_q.dim() > 1:
        group_labels_q = group_labels_q[0]
    return feat_q, group_feats_q, group_valid_q, group_labels_q


def _load_queries(payload, max_queries=0):
    queries = []
    for batch in payload:
        if isinstance(batch, dict):
            q_pid = batch["q_pid"]
            queries.append(
                {
                    "q_imgid": batch["q_imgid"],
                    "q_pid": int(q_pid.item() if torch.is_tensor(q_pid) else q_pid),
                    "q_box": batch["q_box"],
                    "pred_instances": batch["pred_instances"],
                    "gallery_list": None,
                }
            )
            continue
        for item in batch:
            if len(item) == 2:
                q_gt, q_pred = item
                gallery_list = None
            elif len(item) == 3:
                q_gt, gallery_list, q_pred = item
            else:
                continue
            queries.append(
                {
                    "q_imgid": q_gt.image_id,
                    "q_pid": int(q_gt.gt_pids.item()),
                    "q_box": q_gt.org_gt_boxes,
                    "pred_instances": q_pred,
                    "gallery_list": gallery_list,
                }
            )
    if max_queries > 0:
        queries = queries[:max_queries]
    return queries


def _resolve_query_features(query):
    pred_instances = query["pred_instances"]
    feat_q = pred_instances.reid_feats if pred_instances.has("reid_feats") else None
    group_feats_q = (
        pred_instances.cerberus_group_feats
        if pred_instances.has("cerberus_group_feats")
        else None
    )
    group_valid_q = (
        pred_instances.cerberus_group_valid
        if pred_instances.has("cerberus_group_valid")
        else None
    )
    group_labels_q = (
        pred_instances.cerberus_group_labels
        if pred_instances.has("cerberus_group_labels")
        else None
    )
    return _normalize_query_feature_shapes(
        feat_q, group_feats_q, group_valid_q, group_labels_q
    )


def _build_flat_gallery(gallery_saved, score_thresh, device):
    infs = gallery_saved["infs"]
    group_feats_map = gallery_saved.get("inf_group_feats", {})
    group_valids_map = gallery_saved.get("inf_group_valids", {})
    group_labels_map = gallery_saved.get("inf_group_labels", {})

    all_boxes = []
    all_feats = []
    all_img_indices = []
    all_cids = []
    all_group_feats = []
    all_group_valids = []
    all_group_labels = []
    image_names = []
    image_to_flat = {}
    image_has_group = False
    image_has_labels = False
    offset = 0

    for img_id in gallery_saved["gts"].keys():
        if img_id not in infs:
            continue
        img_save_all = infs[img_id]
        score_mask = img_save_all[:, 4] >= score_thresh
        if not score_mask.any():
            continue

        det = img_save_all[score_mask][:, :5]
        feat = img_save_all[score_mask][:, 5:]
        num_det = det.shape[0]
        if num_det == 0:
            continue

        compact_img_idx = len(image_names)
        image_names.append(img_id)
        image_to_flat[img_id] = (offset, offset + num_det)
        offset += num_det

        all_boxes.append(det[:, :4].float())
        all_feats.append(feat.float())
        all_img_indices.append(torch.full((num_det,), compact_img_idx, dtype=torch.long))
        all_cids.append(torch.full((num_det,), _get_img_cid(img_id), dtype=torch.long))

        if img_id in group_feats_map:
            all_group_feats.append(group_feats_map[img_id][score_mask].float())
            image_has_group = True
        else:
            all_group_feats.append(None)

        if img_id in group_valids_map:
            all_group_valids.append(group_valids_map[img_id][score_mask].float())
        else:
            all_group_valids.append(None)

        if img_id in group_labels_map:
            all_group_labels.append(group_labels_map[img_id][score_mask].long())
            image_has_labels = True
        else:
            all_group_labels.append(None)

    flat_boxes = torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros((0, 4))
    flat_feats = torch.cat(all_feats, dim=0) if all_feats else torch.zeros((0, 0))
    flat_img_indices = (
        torch.cat(all_img_indices, dim=0) if all_img_indices else torch.zeros((0,), dtype=torch.long)
    )
    flat_cids = torch.cat(all_cids, dim=0) if all_cids else torch.zeros((0,), dtype=torch.long)

    flat_group_feats = None
    flat_group_valids = None
    flat_group_labels = None

    if image_has_group:
        group_dim = max(
            (tensor.shape[1] for tensor in all_group_feats if tensor is not None),
            default=0,
        )
        embed_dim = max(
            (tensor.shape[2] for tensor in all_group_feats if tensor is not None),
            default=0,
        )
        feat_chunks = []
        valid_chunks = []
        for feat_chunk, valid_chunk, box_chunk in zip(all_group_feats, all_group_valids, all_boxes):
            n = box_chunk.shape[0]
            if feat_chunk is None:
                feat_chunks.append(torch.zeros((n, group_dim, embed_dim), dtype=torch.float32))
            elif feat_chunk.shape[1] < group_dim:
                pad = torch.zeros((n, group_dim - feat_chunk.shape[1], feat_chunk.shape[2]), dtype=feat_chunk.dtype)
                feat_chunks.append(torch.cat([feat_chunk, pad], dim=1))
            else:
                feat_chunks.append(feat_chunk)

            if valid_chunk is None:
                valid_chunks.append(torch.ones((n, group_dim), dtype=torch.float32))
            elif valid_chunk.shape[1] < group_dim:
                pad = torch.zeros((n, group_dim - valid_chunk.shape[1]), dtype=valid_chunk.dtype)
                valid_chunks.append(torch.cat([valid_chunk, pad], dim=1))
            else:
                valid_chunks.append(valid_chunk.float())

        flat_group_feats = torch.cat(feat_chunks, dim=0)
        flat_group_valids = torch.cat(valid_chunks, dim=0)

    if image_has_labels:
        label_dim = max(
            (tensor.shape[1] for tensor in all_group_labels if tensor is not None),
            default=0,
        )
        label_chunks = []
        for label_chunk, box_chunk in zip(all_group_labels, all_boxes):
            n = box_chunk.shape[0]
            if label_chunk is None:
                label_chunks.append(torch.full((n, label_dim), -1, dtype=torch.long))
            elif label_chunk.shape[1] < label_dim:
                pad = torch.full((n, label_dim - label_chunk.shape[1]), -1, dtype=label_chunk.dtype)
                label_chunks.append(torch.cat([label_chunk, pad], dim=1))
            else:
                label_chunks.append(label_chunk.long())
        flat_group_labels = torch.cat(label_chunks, dim=0)

    flat = {
        "boxes_cpu": flat_boxes,
        "img_indices_cpu": flat_img_indices,
        "cids_cpu": flat_cids,
        "image_names": image_names,
        "image_name_to_index": {name: idx for idx, name in enumerate(image_names)},
        "image_to_flat": image_to_flat,
        "boxes_device": flat_boxes.to(device),
        "img_indices_device": flat_img_indices.to(device),
        "cids_device": flat_cids.to(device),
        "feats_device": flat_feats.to(device),
        "group_feats_device": flat_group_feats.to(device) if flat_group_feats is not None else None,
        "group_valids_device": flat_group_valids.to(device) if flat_group_valids is not None else None,
        "group_labels_cpu": flat_group_labels,
        "group_labels_device": flat_group_labels.to(device) if flat_group_labels is not None else None,
    }
    return flat


def _compute_similarity(
    flat_gallery,
    feat_q,
    group_feats_q,
    group_valid_q,
    fusion_mode,
    avg_global_weight,
    avg_part_weights,
    cerberus_group_score_weight,
):
    feat_q = feat_q.to(flat_gallery["feats_device"].device).float().view(-1)
    scores = torch.mv(flat_gallery["feats_device"], feat_q)

    if (
        fusion_mode != "avg"
        or group_feats_q is None
        or group_valid_q is None
        or cerberus_group_score_weight <= 0
        or flat_gallery["group_feats_device"] is None
    ):
        return scores

    group_feats_q = group_feats_q.to(flat_gallery["group_feats_device"].device).float()
    group_valid_q = group_valid_q.to(flat_gallery["group_feats_device"].device).float()
    part_sims = (flat_gallery["group_feats_device"] * group_feats_q.unsqueeze(0)).sum(dim=-1)
    valid_mask = flat_gallery["group_valids_device"] * group_valid_q.unsqueeze(0)

    part_weights = _match_part_weights(
        avg_part_weights, part_sims.shape[1], part_sims.device
    )
    weighted_mask = valid_mask * part_weights.view(1, -1)
    valid_weight_sum = weighted_mask.sum(dim=1)
    nonzero = valid_weight_sum > 0
    if not nonzero.any():
        return scores

    part_sum = (part_sims * weighted_mask).sum(dim=1)
    fused = scores.clone()
    fused[nonzero] = (
        avg_global_weight * scores[nonzero]
        + cerberus_group_score_weight * part_sum[nonzero]
    ) / (avg_global_weight + cerberus_group_score_weight * valid_weight_sum[nonzero])
    return fused


def _compute_similarity_batch(
    flat_gallery,
    feat_q_batch,
    group_feats_q_batch,
    group_valid_q_batch,
    fusion_mode,
    avg_global_weight,
    avg_part_weights,
    cerberus_group_score_weight,
):
    feat_q_batch = feat_q_batch.to(flat_gallery["feats_device"].device).float()
    scores = torch.matmul(flat_gallery["feats_device"], feat_q_batch.t())

    if (
        fusion_mode != "avg"
        or group_feats_q_batch is None
        or group_valid_q_batch is None
        or cerberus_group_score_weight <= 0
        or flat_gallery["group_feats_device"] is None
    ):
        return scores

    group_feats_q_batch = group_feats_q_batch.to(flat_gallery["group_feats_device"].device).float()
    group_valid_q_batch = group_valid_q_batch.to(flat_gallery["group_feats_device"].device).float()
    part_sims = torch.einsum(
        "nge,bge->nbg", flat_gallery["group_feats_device"], group_feats_q_batch
    )
    valid_mask = (
        flat_gallery["group_valids_device"].unsqueeze(1)
        * group_valid_q_batch.unsqueeze(0)
    )

    part_weights = _match_part_weights(
        avg_part_weights, part_sims.shape[2], part_sims.device
    )
    weighted_mask = valid_mask * part_weights.view(1, 1, -1)
    valid_weight_sum = weighted_mask.sum(dim=2)
    part_sum = (part_sims * weighted_mask).sum(dim=2)

    fused = scores.clone()
    nonzero = valid_weight_sum > 0
    fused[nonzero] = (
        avg_global_weight * scores[nonzero]
        + cerberus_group_score_weight * part_sum[nonzero]
    ) / (avg_global_weight + cerberus_group_score_weight * valid_weight_sum[nonzero])
    return fused


def _apply_gender_label_gate(scores, flat_gallery, query_labels):
    gallery_labels = (
        flat_gallery["group_labels_cpu"]
        if scores.device.type == "cpu"
        else flat_gallery["group_labels_device"]
    )
    if query_labels is None or gallery_labels is None or scores.numel() == 0:
        return scores
    if gallery_labels.shape[1] < 1 or query_labels.shape[0] < 1:
        return scores

    query_gender = int(query_labels[0].item())
    if query_gender not in (0, 1):
        return scores

    gallery_gender = gallery_labels[:, 0]
    known_mask = (gallery_gender == 0) | (gallery_gender == 1)
    if not known_mask.any():
        return scores

    keep_mask = (~known_mask) | (gallery_gender == query_gender)
    if keep_mask.all() or not keep_mask.any():
        return scores

    fused = scores.clone()
    fused[~keep_mask] = -1e6
    return fused


def _build_gender_keep_mask(flat_gallery, query_labels, device):
    gallery_labels = (
        flat_gallery["group_labels_cpu"]
        if device.type == "cpu"
        else flat_gallery["group_labels_device"]
    )
    if query_labels is None or gallery_labels is None:
        return None
    if gallery_labels.shape[1] < 1 or query_labels.shape[0] < 1:
        return None

    query_gender = int(query_labels[0].item())
    if query_gender not in (0, 1):
        return None

    gallery_gender = gallery_labels[:, 0]
    known_mask = (gallery_gender == 0) | (gallery_gender == 1)
    if not known_mask.any():
        return None

    keep_mask = (~known_mask) | (gallery_gender == query_gender)
    if keep_mask.all() or not keep_mask.any():
        return None
    return keep_mask


def _resolve_gender_filter_mode(args):
    if args.gender_filter_mode != "none":
        return str(args.gender_filter_mode).lower()
    return "hard" if bool(args.gender_label_gate) else "none"


def _get_gender_label_tensor(flat_gallery, device):
    return (
        flat_gallery["group_labels_cpu"]
        if device.type == "cpu"
        else flat_gallery["group_labels_device"]
    )


def _build_group_match_masks(gallery_labels, query_labels, group_idx):
    if query_labels is None or gallery_labels is None:
        return None
    if gallery_labels.dim() != 2 or query_labels.dim() != 1:
        return None
    if gallery_labels.shape[1] <= group_idx or query_labels.shape[0] <= group_idx:
        return None

    query_label = int(query_labels[group_idx].item())
    if query_label < 0:
        return None

    gallery_group = gallery_labels[:, group_idx]
    gallery_known = gallery_group >= 0
    if not gallery_known.any():
        return None

    match_mask = (~gallery_known) | (gallery_group == query_label)
    if match_mask.all():
        return None

    return {
        "query_label": query_label,
        "gallery_group": gallery_group,
        "gallery_known": gallery_known,
        "match_mask": match_mask,
    }


def _apply_gender_filter_mode(scores, flat_gallery, query_labels, args):
    mode = _resolve_gender_filter_mode(args)
    if mode == "none":
        return scores, None

    gallery_labels = _get_gender_label_tensor(flat_gallery, scores.device)
    if query_labels is None or gallery_labels is None or scores.numel() == 0:
        return scores, None
    if gallery_labels.shape[1] < 1 or query_labels.shape[0] < 1:
        return scores, None

    query_gender = int(query_labels[0].item())
    if query_gender not in (0, 1):
        return scores, None

    gallery_gender = gallery_labels[:, 0]
    known_mask = (gallery_gender == 0) | (gallery_gender == 1)
    if not known_mask.any():
        return scores, None

    keep_mask = (~known_mask) | (gallery_gender == query_gender)
    if keep_mask.all() or not keep_mask.any():
        return scores, None

    if mode == "hard":
        return scores[keep_mask], keep_mask

    if mode == "soft":
        penalized = scores.clone()
        penalized[~keep_mask] = penalized[~keep_mask] - float(args.gender_soft_penalty)
        return penalized, None

    if mode == "topk_hard":
        topk = min(max(int(args.gender_topk), 1), int(scores.numel()))
        topk_idx = torch.argsort(scores, descending=True)[:topk]
        topk_keep = keep_mask[topk_idx]
        if topk_keep.all() or not topk_keep.any():
            return scores, None
        pruned = scores.clone()
        drop_idx = topk_idx[~topk_keep]
        pruned[drop_idx] = pruned[drop_idx].new_full((drop_idx.numel(),), -1e6)
        return pruned, None

    if mode == "selective_hard":
        keep_ratio = float(keep_mask.float().mean().item())
        known_ratio = float(known_mask.float().mean().item())
        if (
            keep_ratio >= float(args.gender_selective_min_keep_ratio)
            and known_ratio >= float(args.gender_selective_min_known_ratio)
        ):
            return scores[keep_mask], keep_mask
        return scores, None

    return scores, None


def _apply_query_selective_filter(scores, flat_gallery, query_labels, args):
    if not bool(args.query_selective_filter):
        return scores, None
    if scores.numel() == 0:
        return scores, None

    gallery_labels = _get_gender_label_tensor(flat_gallery, scores.device)
    if query_labels is None or gallery_labels is None:
        return scores, None

    topk = min(max(int(args.query_selective_shortlist_topk), 1), int(scores.numel()))
    shortlist_idx = torch.argsort(scores, descending=True)[:topk]
    shortlist_scores = scores[shortlist_idx].clone()
    shortlist_labels = gallery_labels[shortlist_idx]

    keep_mask_short = torch.ones(
        shortlist_scores.shape[0], dtype=torch.bool, device=shortlist_scores.device
    )

    gender_masks = _build_group_match_masks(shortlist_labels, query_labels, 0)
    if gender_masks is not None:
        gender_keep_ratio = float(gender_masks["match_mask"].float().mean().item())
        gender_known_ratio = float(gender_masks["gallery_known"].float().mean().item())
        if (
            gender_keep_ratio
            >= float(args.query_selective_gender_min_keep_ratio)
            and gender_known_ratio
            >= float(args.query_selective_gender_min_known_ratio)
        ):
            keep_mask_short &= gender_masks["match_mask"]

    penalty_specs = (
        (2, float(args.query_selective_top_penalty)),
        (3, float(args.query_selective_pants_penalty)),
    )
    for group_idx, penalty in penalty_specs:
        if penalty <= 0:
            continue
        group_masks = _build_group_match_masks(shortlist_labels, query_labels, group_idx)
        if group_masks is None:
            continue
        mismatch_mask = group_masks["gallery_known"] & (~group_masks["match_mask"])
        if mismatch_mask.any():
            shortlist_scores[mismatch_mask] -= penalty

    if not keep_mask_short.all():
        if keep_mask_short.any():
            keep_mask = torch.ones_like(scores, dtype=torch.bool)
            keep_mask[shortlist_idx[~keep_mask_short]] = False
            scores = scores.clone()
            scores[shortlist_idx] = shortlist_scores
            return scores, keep_mask
        return scores, None

    scores = scores.clone()
    scores[shortlist_idx] = shortlist_scores
    return scores, None


def _build_prw_query_gts(gts, q_imgid, q_pid):
    query_gts = {}
    count_gts = 0
    count_gts_mlv = 0
    q_cid = _get_img_cid(q_imgid)
    for gt_img_id, gt_img_label in gts.items():
        if gt_img_id == q_imgid:
            continue
        gt_ids = gt_img_label[:, 4].long()
        if q_pid in gt_ids.tolist():
            query_gts[gt_img_id] = gt_img_label[gt_ids == q_pid][:, :4]
            count_gts += 1
            if _get_img_cid(gt_img_id) != q_cid:
                count_gts_mlv += 1
    return query_gts, count_gts, count_gts_mlv


def _build_prw_pid_index(gts):
    pid_index = {}
    for gt_img_id, gt_img_label in gts.items():
        gt_ids = gt_img_label[:, 4].long()
        unique_pids = torch.unique(gt_ids)
        for pid_t in unique_pids:
            pid = int(pid_t.item())
            if pid < 0:
                continue
            pid_index.setdefault(pid, []).append(
                {
                    "image_id": gt_img_id,
                    "boxes": gt_img_label[gt_ids == pid][:, :4],
                    "cid": _get_img_cid(gt_img_id),
                }
            )
    return pid_index


def _build_prw_query_gts_from_index(pid_index, q_imgid, q_pid):
    query_gts = {}
    count_gts = 0
    count_gts_mlv = 0
    q_cid = _get_img_cid(q_imgid)
    for item in pid_index.get(q_pid, []):
        gt_img_id = item["image_id"]
        if gt_img_id == q_imgid:
            continue
        query_gts[gt_img_id] = item["boxes"]
        count_gts += 1
        if item["cid"] != q_cid:
            count_gts_mlv += 1
    return query_gts, count_gts, count_gts_mlv


def _build_cuhk_query_gts(gallery_list):
    query_gts = {}
    count_gts = 0
    for item in gallery_list:
        gt_boxes = item.org_gt_boxes
        query_gts[item.image_id] = gt_boxes.tensor if hasattr(gt_boxes, "tensor") else gt_boxes
        count_gts += len(gt_boxes) > 0
    return query_gts, count_gts


def _compute_iou_thresh(gt_boxes):
    gt_boxes = gt_boxes if gt_boxes.dim() > 1 else gt_boxes.view(1, 4)
    w = gt_boxes[0, 2] - gt_boxes[0, 0]
    h = gt_boxes[0, 3] - gt_boxes[0, 1]
    return float(min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10))))


def _compute_ap_and_acc(y_true, y_score, count_tps, count_gts, topks):
    if len(y_true) == 0:
        return 0.0, [0.0 for _ in topks]
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    recall_rate = count_tps * 1.0 / count_gts if count_gts > 0 else 0.0
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    if count_tps == 0:
        ap = 0.0
    else:
        tp_cum = np.cumsum(y_true)
        pos_idx = np.flatnonzero(y_true > 0)
        precision_at_pos = tp_cum[pos_idx] / (pos_idx + 1.0)
        ap = float(precision_at_pos.mean() * recall_rate) if pos_idx.size > 0 else 0.0
    acc = [float(min(1, np.sum(y_true[:k]))) for k in topks]
    return float(ap), acc


def _pad_query_group_tensors(prepared_batch):
    max_group_dim = 0
    embed_dim = 0
    for item in prepared_batch:
        gf = item["group_feats_q"]
        gv = item["group_valid_q"]
        if gf is None or gv is None:
            continue
        max_group_dim = max(max_group_dim, int(gf.shape[0]))
        embed_dim = max(embed_dim, int(gf.shape[1]))

    if max_group_dim <= 0 or embed_dim <= 0:
        return None, None

    gf_chunks = []
    gv_chunks = []
    for item in prepared_batch:
        gf = item["group_feats_q"]
        gv = item["group_valid_q"]
        if gf is None or gv is None:
            gf_chunks.append(torch.zeros((max_group_dim, embed_dim), dtype=torch.float32))
            gv_chunks.append(torch.zeros((max_group_dim,), dtype=torch.float32))
            continue
        gf = gf.float()
        gv = gv.float()
        if gf.shape[0] < max_group_dim:
            gf_pad = torch.zeros((max_group_dim - gf.shape[0], gf.shape[1]), dtype=gf.dtype)
            gv_pad = torch.zeros((max_group_dim - gv.shape[0],), dtype=gv.dtype)
            gf = torch.cat([gf, gf_pad], dim=0)
            gv = torch.cat([gv, gv_pad], dim=0)
        if gf.shape[1] < embed_dim:
            gf_pad = torch.zeros((gf.shape[0], embed_dim - gf.shape[1]), dtype=gf.dtype)
            gf = torch.cat([gf, gf_pad], dim=1)
        gf_chunks.append(gf)
        gv_chunks.append(gv)
    return torch.stack(gf_chunks, dim=0), torch.stack(gv_chunks, dim=0)


def evaluate(args):
    device = _resolve_device(args.device)
    print("Loading gallery cache...")
    gallery_saved = torch.load(args.gallery_file, map_location="cpu")
    print("Loading query cache...")
    query_saved = torch.load(args.query_file, map_location="cpu")

    queries = _load_queries(query_saved["payload"], max_queries=args.max_queries)
    gts = gallery_saved["gts"]
    print("Flattening gallery to device...")
    flat_gallery = _build_flat_gallery(gallery_saved, args.score_thresh, device)
    prw_pid_index = _build_prw_pid_index(gts) if args.dataset == "prw" else None
    cerberus_group_score_weight = (
        float(args.cerberus_group_score_weight)
        if args.cerberus_group_score_weight is not None
        else float(gallery_saved.get("cerberus_group_score_weight", 0.0))
    )

    results = {
        "aps": [],
        "accs": [],
        "aps_mlv": [],
        "accs_mlv": [],
        "num_queries": 0,
        "num_queries_mlv": 0,
    }

    img_indices_cpu = flat_gallery["img_indices_cpu"]
    boxes_cpu = flat_gallery["boxes_cpu"]
    cids_cpu = flat_gallery["cids_cpu"]
    all_names = flat_gallery["image_names"]
    need_group_batch = args.fusion_mode == "avg" and flat_gallery["group_feats_device"] is not None

    print("Preparing queries...")
    prepared_queries = []
    prep_progress = tqdm(total=len(queries), desc="Preparing queries", ncols=100)
    for query in queries:
        feat_q, group_feats_q, group_valid_q, group_labels_q = _resolve_query_features(query)
        prep_progress.update(1)
        if feat_q is None:
            continue

        q_imgid = query["q_imgid"]
        q_pid = query["q_pid"]
        q_cid = _get_img_cid(q_imgid)
        if args.dataset == "prw":
            query_gts, count_gts, count_gts_mlv = _build_prw_query_gts_from_index(
                prw_pid_index, q_imgid, q_pid
            )
        else:
            query_gts, count_gts = _build_cuhk_query_gts(query["gallery_list"])
            count_gts_mlv = 0

        prepared_queries.append(
            {
                "query": query,
                "feat_q": feat_q.float(),
                "group_feats_q": group_feats_q,
                "group_valid_q": group_valid_q,
                "group_labels_q": group_labels_q,
                "q_cid": q_cid,
                "query_gts": query_gts,
                "count_gts": count_gts,
                "count_gts_mlv": count_gts_mlv,
            }
        )
    prep_progress.close()

    query_batch_size = max(int(args.query_batch_size), 1)
    eval_progress = tqdm(total=len(prepared_queries), desc="Evaluating queries", ncols=100)

    full_scores_cpu = None
    if args.full_score_matrix and prepared_queries:
        print("Computing full score matrix on device...")
        feat_q_all = torch.stack([item["feat_q"] for item in prepared_queries], dim=0)
        group_feats_q_all, group_valid_q_all = (
            _pad_query_group_tensors(prepared_queries) if need_group_batch else (None, None)
        )
        score_cols = []
        for batch_start in range(0, feat_q_all.shape[0], query_batch_size):
            batch_end = min(batch_start + query_batch_size, feat_q_all.shape[0])
            score_cols.append(
                _compute_similarity_batch(
                    flat_gallery,
                    feat_q_all[batch_start:batch_end],
                    group_feats_q_all[batch_start:batch_end] if group_feats_q_all is not None else None,
                    group_valid_q_all[batch_start:batch_end] if group_valid_q_all is not None else None,
                    args.fusion_mode,
                    args.avg_global_weight,
                    args.avg_part_weights,
                    cerberus_group_score_weight,
                ).detach().cpu()
            )
        full_scores_cpu = torch.cat(score_cols, dim=1)

    for batch_start in range(0, len(prepared_queries), query_batch_size):
        prepared_batch = prepared_queries[batch_start : batch_start + query_batch_size]
        if not prepared_batch:
            continue

        if full_scores_cpu is None:
            feat_q_batch = torch.stack([item["feat_q"] for item in prepared_batch], dim=0)
            group_feats_q_batch, group_valid_q_batch = (
                _pad_query_group_tensors(prepared_batch) if need_group_batch else (None, None)
            )
            scores_batch_cpu = _compute_similarity_batch(
                flat_gallery,
                feat_q_batch,
                group_feats_q_batch,
                group_valid_q_batch,
                args.fusion_mode,
                args.avg_global_weight,
                args.avg_part_weights,
                cerberus_group_score_weight,
            ).detach().cpu()
        else:
            batch_end = batch_start + len(prepared_batch)
            scores_batch_cpu = full_scores_cpu[:, batch_start:batch_end]

        for batch_col, item in enumerate(prepared_batch):
            query = item["query"]
            q_imgid = query["q_imgid"]
            q_cid = item["q_cid"]
            query_gts = item["query_gts"]
            count_gts = item["count_gts"]
            count_gts_mlv = item["count_gts_mlv"]
            scores = scores_batch_cpu[:, batch_col]
            scores, keep_mask = _apply_gender_filter_mode(
                scores, flat_gallery, item["group_labels_q"], args
            )
            selective_scores, selective_keep_mask = _apply_query_selective_filter(
                scores, flat_gallery, item["group_labels_q"], args
            )
            scores = selective_scores
            if selective_keep_mask is not None:
                keep_mask = (
                    selective_keep_mask
                    if keep_mask is None
                    else (keep_mask & selective_keep_mask)
                )
            if keep_mask is not None:
                keep_mask_cpu = keep_mask.cpu()
                boxes_work = boxes_cpu[keep_mask_cpu]
                img_indices_work = img_indices_cpu[keep_mask_cpu]
                cids_work = cids_cpu[keep_mask_cpu]
            else:
                boxes_work = boxes_cpu
                img_indices_work = img_indices_cpu
                cids_work = cids_cpu
            scores_cpu = scores

            if args.dataset == "prw":
                q_img_idx = flat_gallery["image_name_to_index"].get(q_imgid, -1)
                active_mask = img_indices_work != q_img_idx
            else:
                allowed = {gallery_item.image_id for gallery_item in query["gallery_list"]}
                active_mask = torch.tensor(
                    [name in allowed for name in all_names], dtype=torch.bool
                )[img_indices_work]

            labels = torch.zeros(scores_cpu.shape[0], dtype=torch.int)
            labels_mlv = torch.zeros(scores_cpu.shape[0], dtype=torch.int)
            count_tps = 0
            count_tps_mlv = 0

            for gallery_imname, gt_boxes in query_gts.items():
                if gallery_imname not in flat_gallery["image_to_flat"]:
                    continue
                start, end = flat_gallery["image_to_flat"][gallery_imname]
                gallery_mask = (img_indices_work == flat_gallery["image_name_to_index"][gallery_imname])
                if not gallery_mask.any():
                    continue
                det_boxes = boxes_work[gallery_mask]

                gt_boxes = gt_boxes if gt_boxes.dim() > 1 else gt_boxes.view(1, 4)
                iou_thresh = _compute_iou_thresh(gt_boxes)
                ious = box_iou(det_boxes, gt_boxes)
                positive_local = torch.nonzero(
                    ious.max(dim=1).values >= iou_thresh, as_tuple=False
                ).flatten()
                if positive_local.numel() == 0:
                    continue

                local_scores = scores_cpu[gallery_mask]
                best_local = positive_local[torch.argmax(local_scores[positive_local])].item()
                local_indices = torch.nonzero(gallery_mask, as_tuple=False).flatten()
                global_idx = local_indices[best_local].item()
                labels[global_idx] = 1
                count_tps += 1
                if args.dataset == "prw" and cids_work[global_idx].item() != q_cid:
                    labels_mlv[global_idx] = 1
                    count_tps_mlv += 1

            y_true = labels[active_mask].numpy()
            y_score = scores_cpu[active_mask].numpy()
            ap, acc = _compute_ap_and_acc(y_true, y_score, count_tps, count_gts, args.topk)
            results["aps"].append(ap)
            results["accs"].append(acc)
            results["num_queries"] += 1

            if args.dataset == "prw":
                mlv_mask = active_mask & (cids_work != q_cid)
                y_true_mlv = labels_mlv[mlv_mask].numpy()
                y_score_mlv = scores_cpu[mlv_mask].numpy()
                ap_mlv, acc_mlv = _compute_ap_and_acc(
                    y_true_mlv, y_score_mlv, count_tps_mlv, count_gts_mlv, args.topk
                )
                results["aps_mlv"].append(ap_mlv)
                results["accs_mlv"].append(acc_mlv)
                results["num_queries_mlv"] += 1
            eval_progress.update(1)

    eval_progress.close()

    print("=== Search Eval From Cache ===")
    print("device:", str(device))
    print("dataset:", args.dataset)
    print("fusion_mode:", args.fusion_mode)
    print("score_thresh:", args.score_thresh)
    print("query_batch_size:", query_batch_size)
    print("full_score_matrix:", args.full_score_matrix)
    print("gender_label_gate:", args.gender_label_gate)
    print("query_selective_filter:", args.query_selective_filter)
    print("query_selective_shortlist_topk:", int(args.query_selective_shortlist_topk))
    print("gallery_detections:", int(flat_gallery["boxes_cpu"].shape[0]))
    print("feat_device:", str(flat_gallery["feats_device"].device))
    print(
        "group_feat_device:",
        str(flat_gallery["group_feats_device"].device)
        if flat_gallery["group_feats_device"] is not None
        else "None",
    )
    print("queries_evaluated:", results["num_queries"])
    if results["num_queries"] > 0:
        print("mAP:", float(np.mean(results["aps"])))
        mean_acc = np.mean(np.asarray(results["accs"]), axis=0)
        for idx, k in enumerate(args.topk):
            print(f"top{k}:", float(mean_acc[idx]))
    if args.dataset == "prw" and results["num_queries_mlv"] > 0:
        print("mAP_mlv:", float(np.mean(results["aps_mlv"])))
        mean_acc_mlv = np.mean(np.asarray(results["accs_mlv"]), axis=0)
        for idx, k in enumerate(args.topk):
            print(f"top{k}_mlv:", float(mean_acc_mlv[idx]))


def main():
    args = parse_args()
    if not os.path.exists(args.gallery_file):
        raise FileNotFoundError(args.gallery_file)
    if not os.path.exists(args.query_file):
        raise FileNotFoundError(args.query_file)
    evaluate(args)


if __name__ == "__main__":
    main()
