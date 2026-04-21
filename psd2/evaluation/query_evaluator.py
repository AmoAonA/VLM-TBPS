from copy import copy
from typing import List, OrderedDict
import psd2.utils.comm as comm
import torch
import logging
from .evaluator import DatasetEvaluator
import itertools
import os
import numpy as np
import copy
import logging
import psd2.utils.comm as comm
from psd2.structures import Boxes, BoxMode, pairwise_iou


logger = logging.getLogger(__name__)  # setup_logger()


class QueryEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
        search_fusion_mode="auto",
        search_shortlist_size=200,
        search_rrf_k=20.0,
        search_max_parts=2,
        search_global_vote_weight=1.0,
        search_part_vote_weights=None,
        search_avg_global_weight=1.0,
        search_avg_part_weights=None,
        search_bp_pos_threshold=0.55,
        search_bp_neg_threshold=0.15,
        search_bp_pos_weight=0.12,
        search_bp_neg_weight=0.08,
        search_kde_bandwidth=0.0,
        search_kde_density_weight=0.3,
        search_kde_top_k=50,
        search_gender_label_shortlist_enabled=False,
        search_gender_label_shortlist_size=0,
        search_top_pants_soft_rerank_enabled=False,
        search_top_pants_soft_rerank_size=0,
        search_top_soft_weight=0.08,
        search_pants_soft_weight=0.08,
        search_top_soft_center=0.35,
        search_pants_soft_center=0.35,
        search_top_pants_soft_rerank_uncertain_margin=0.0,
        search_top_pants_soft_rerank_score_band=0.0,
        search_top_pants_soft_rerank_max_delta=0.0,
        cuhk_search_top_pants_positive_only=False,
        coarse_topk=0,
        reuse_eval_cache=False,
        save_query_cache=True,
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.reuse_eval_cache = bool(reuse_eval_cache)
        self.save_query_cache = bool(save_query_cache)
        self._cached_query_ready = False
        # {image name: torch concatenated [boxes, scores, reid features]
        inf_rts = torch.load(
            os.path.join(
                self._output_dir,
                "_gallery_gt_inf.pt"
                if "GT" not in self.dataset_name
                else "_gallery_gt_infgt.pt",
            ),
            map_location=self._cpu_device,
        )
        self.gts = inf_rts["gts"]
        self.infs = inf_rts["infs"]
        self.gtfs = inf_rts["gt_fnames"]
        self.gt_attrs = inf_rts.get("gt_attrs", {})
        self.inf_group_feats = inf_rts.get("inf_group_feats", {})
        self.inf_group_valids = inf_rts.get("inf_group_valids", {})
        self.inf_group_labels = inf_rts.get("inf_group_labels", {})
        self.cerberus_group_score_weight = float(
            inf_rts.get(
                "cerberus_group_score_weight",
                1.0 if len(self.inf_group_feats) > 0 else 0.0,
            )
        )
        self.topks = [1, 5, 10]
        self.search_fusion_mode = self._resolve_search_fusion_mode(
            search_fusion_mode,
            len(self.inf_group_feats) > 0 and self.cerberus_group_score_weight > 0,
        )
        self.search_shortlist_size = max(int(search_shortlist_size), 1)
        self.search_rrf_k = float(search_rrf_k)
        self.search_max_parts = max(int(search_max_parts), 0)
        self.search_global_vote_weight = float(search_global_vote_weight)
        if search_part_vote_weights is None:
            search_part_vote_weights = [0.05, 0.20, 0.35, 0.30, 0.10]
        self.search_part_vote_weights = torch.tensor(
            search_part_vote_weights, dtype=torch.float32
        )
        if self.search_part_vote_weights.dim() != 1:
            raise ValueError("TEST.SEARCH_PART_VOTE_WEIGHTS must be a 1D sequence.")
        self.search_avg_global_weight = float(search_avg_global_weight)
        if search_avg_part_weights is None:
            search_avg_part_weights = [1.0] * int(self.search_part_vote_weights.numel())
        self.search_avg_part_weights = torch.tensor(
            search_avg_part_weights, dtype=torch.float32
        )
        if self.search_avg_part_weights.dim() != 1:
            raise ValueError("TEST.SEARCH_AVG_PART_WEIGHTS must be a 1D sequence.")
        self.search_bp_pos_threshold = float(search_bp_pos_threshold)
        self.search_bp_neg_threshold = float(search_bp_neg_threshold)
        self.search_bp_pos_weight = float(search_bp_pos_weight)
        self.search_bp_neg_weight = float(search_bp_neg_weight)
        self.search_kde_bandwidth = float(search_kde_bandwidth)
        self.search_kde_density_weight = float(search_kde_density_weight)
        self.search_kde_top_k = max(int(search_kde_top_k), 1)
        self.search_gender_label_shortlist_enabled = bool(
            search_gender_label_shortlist_enabled
        )
        self.search_gender_label_shortlist_size = int(search_gender_label_shortlist_size)
        self.search_top_pants_soft_rerank_enabled = bool(
            search_top_pants_soft_rerank_enabled
        )
        self.search_top_pants_soft_rerank_size = int(search_top_pants_soft_rerank_size)
        self.search_top_soft_weight = float(search_top_soft_weight)
        self.search_pants_soft_weight = float(search_pants_soft_weight)
        self.search_top_soft_center = float(search_top_soft_center)
        self.search_pants_soft_center = float(search_pants_soft_center)
        self.search_top_pants_soft_rerank_uncertain_margin = float(
            search_top_pants_soft_rerank_uncertain_margin
        )
        self.search_top_pants_soft_rerank_score_band = float(
            search_top_pants_soft_rerank_score_band
        )
        self.search_top_pants_soft_rerank_max_delta = float(
            search_top_pants_soft_rerank_max_delta
        )
        self.cuhk_search_top_pants_positive_only = bool(
            cuhk_search_top_pants_positive_only
        ) and ("CUHK-SYSU" in str(self.dataset_name))
        self.coarse_topk = int(coarse_topk) if coarse_topk > 0 else 0
        # statistics
        self.det_score_thresh = s_threds
        self.ignore_cam_id = False
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}

    @staticmethod
    def _resolve_search_fusion_mode(requested_mode, has_group_feats):
        mode = "auto" if requested_mode is None else str(requested_mode).lower()
        if mode == "auto":
            return "rrf" if has_group_feats else "global"
        if mode not in {"global", "avg", "rrf", "kde", "shortlist", "bonus_penalty"}:
            raise ValueError(
                "TEST.SEARCH_FUSION_MODE must be one of auto/global/avg/rrf/kde/shortlist/bonus_penalty."
            )
        if not has_group_feats and mode != "global":
            return "global"
        return mode

    @staticmethod
    def _match_part_weights(weights, target_dim, default_value=None):
        if target_dim <= 0:
            return weights[:0]
        if weights.numel() == target_dim:
            return weights
        if weights.numel() > target_dim:
            return weights[:target_dim]
        if default_value is None:
            default_value = float(weights[-1].item()) if weights.numel() > 0 else 1.0
        padding = weights.new_full((target_dim - weights.numel(),), float(default_value))
        return torch.cat([weights, padding], dim=0)


    def reset(self):
        self._cached_query_ready = False
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}

    def supports_cached_inference(self):
        return bool(self._cached_query_ready)

    def _query_cache_path(self):
        return os.path.join(
            self._output_dir,
            "_query_inf.pt" if "GT" not in self.dataset_name else "_query_infgt.pt",
        )

    def _load_query_cache_payload(self, payload):
        return False

    def _get_query_cache_payload(self):
        return None

    def _maybe_load_query_cache(self):
        if not self.reuse_eval_cache:
            return False
        cache_path = self._query_cache_path()
        if not os.path.exists(cache_path):
            return False
        cached = torch.load(cache_path, map_location=self._cpu_device)
        payload = cached.get("payload")
        if payload is None:
            return False
        loaded = self._load_query_cache_payload(payload)
        self._cached_query_ready = bool(loaded)
        return self._cached_query_ready

    def _maybe_save_query_cache(self):
        if not self.save_query_cache:
            return
        payload = self._get_query_cache_payload()
        if payload is None:
            return
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        torch.save({"payload": payload}, self._query_cache_path())

    def _get_gallery_dets(self, img_id, score_thred):
        img_dets_all = self.infs[img_id][:, :5]
        return img_dets_all[img_dets_all[:, 4] >= score_thred]

    def _get_gallery_feats(self, img_id, score_thred):
        img_save_all = self.infs[img_id]
        return img_save_all[img_save_all[:, 4] >= score_thred][:, 5:]

    def _get_gallery_group_feats(self, img_id, score_thred):
        if img_id not in self.inf_group_feats:
            return None
        img_save_all = self.infs[img_id]
        return self.inf_group_feats[img_id][img_save_all[:, 4] >= score_thred]

    def _get_gallery_group_valids(self, img_id, score_thred):
        if img_id not in self.inf_group_valids:
            return None
        img_save_all = self.infs[img_id]
        return self.inf_group_valids[img_id][img_save_all[:, 4] >= score_thred]

    def _get_gallery_group_labels(self, img_id, score_thred):
        if img_id not in self.inf_group_labels:
            return None
        img_save_all = self.infs[img_id]
        return self.inf_group_labels[img_id][img_save_all[:, 4] >= score_thred]

    def _get_gt_boxs(self, img_id):
        return self.gts[img_id][:, :4]

    def _get_gt_ids(self, img_id):
        return self.gts[img_id][:, 4].long()

    def process(self, inputs, outputs):
        raise NotImplementedError

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps, dst=0)
            accs_all = comm.gather(self.accs, dst=0)

            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))

        else:
            aps = self.aps
            accs = self.accs
        search_result = OrderedDict()
        search_result["search"] = {}

        for dst in self.det_score_thresh:
            logger.info("Search eval_{:.2f} on {} queries. ".format(dst, len(aps[dst])))
            mAP = np.mean(aps[dst])
            search_result["search"].update({"mAP_{:.2f}".format(dst): mAP})
            acc = np.mean(np.array(accs[dst]), axis=0)
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                search_result["search"].update({"top{:2d}_{:.2f}".format(k, dst): v})


        return copy.deepcopy(search_result)

    def _get_query_gallery_gts(self, query_dict):
        raise NotImplementedError

    def _normalize_query_feature_shapes(
        self, feat_q, group_feats_q=None, group_valid_q=None, group_labels_q=None
    ):
        if feat_q.dim() > 1:
            feat_q = feat_q[0]
        if group_feats_q is not None and group_feats_q.dim() > 2:
            group_feats_q = group_feats_q[0]
        if group_valid_q is not None and group_valid_q.dim() > 1:
            group_valid_q = group_valid_q[0]
        if group_labels_q is not None and group_labels_q.dim() > 1:
            group_labels_q = group_labels_q[0]
        return feat_q, group_feats_q, group_valid_q, group_labels_q

    def _resolve_query_features(self, pred_instances, q_imgid, q_box, score_thred):
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

        if feat_q is not None:
            return self._normalize_query_feature_shapes(
                feat_q, group_feats_q, group_valid_q, group_labels_q
            )

        query_img_boxes_t = self._get_gallery_dets(q_imgid, score_thred)[:, :4]
        query_img_feats = self._get_gallery_feats(q_imgid, score_thred)
        if query_img_boxes_t.shape[0] == 0:
            return None, None, None, None

        ious = pairwise_iou(q_box, Boxes(query_img_boxes_t, BoxMode.XYXY_ABS))
        max_iou, nmax = torch.max(ious, dim=1)
        if max_iou.item() < 0.4:
            logger.warning(
                "Low-quality {} query person detected in {} !".format(
                    max_iou.item(), q_imgid
                )
            )
        match_idx = nmax.item()
        feat_q = query_img_feats[match_idx]
        group_feats_all = self._get_gallery_group_feats(q_imgid, score_thred)
        group_valid_all = self._get_gallery_group_valids(q_imgid, score_thred)
        if group_feats_all is not None and match_idx < len(group_feats_all):
            group_feats_q = group_feats_all[match_idx]
        else:
            group_feats_q = None
        if group_valid_all is not None and match_idx < len(group_valid_all):
            group_valid_q = group_valid_all[match_idx]
        else:
            group_valid_q = None
        group_labels_all = self._get_gallery_group_labels(q_imgid, score_thred)
        if group_labels_all is not None and match_idx < len(group_labels_all):
            group_labels_q = group_labels_all[match_idx]
        else:
            group_labels_q = None
        return self._normalize_query_feature_shapes(
            feat_q, group_feats_q, group_valid_q, group_labels_q
        )

    def _compute_similarity(
        self,
        feat_g,
        feat_q,
        gallery_imgid,
        score_thred,
        query_group_feats=None,
        query_group_valid=None,
    ):
        global_sim, part_sims, valid_mask = self._compute_similarity_components(
            feat_g,
            feat_q,
            gallery_imgid,
            score_thred,
            query_group_feats,
            query_group_valid,
        )
        return self._fuse_average_scores(global_sim, part_sims, valid_mask)

    def _compute_similarity_components(
        self,
        feat_g,
        feat_q,
        gallery_imgid,
        score_thred,
        query_group_feats=None,
        query_group_valid=None,
    ):
        global_sim = torch.mm(feat_g, feat_q.view(-1)[:, None]).squeeze(1)

        if (
            query_group_feats is None
            or query_group_valid is None
            or self.cerberus_group_score_weight <= 0
        ):
            return global_sim, None, None

        gallery_group_feats = self._get_gallery_group_feats(gallery_imgid, score_thred)
        if gallery_group_feats is None or gallery_group_feats.shape[0] != feat_g.shape[0]:
            return global_sim, None, None

        gallery_group_valid = self._get_gallery_group_valids(gallery_imgid, score_thred)
        if gallery_group_valid is None:
            gallery_group_valid = torch.ones(
                gallery_group_feats.shape[:2], dtype=torch.float32, device=gallery_group_feats.device
            )
        else:
            gallery_group_valid = gallery_group_valid.to(gallery_group_feats.device).float()

        query_group_feats = query_group_feats.to(gallery_group_feats.device)
        query_group_valid = query_group_valid.to(gallery_group_feats.device).float()
        part_sims = (gallery_group_feats * query_group_feats.unsqueeze(0)).sum(dim=-1)
        valid_mask = gallery_group_valid * query_group_valid.unsqueeze(0)
        return global_sim, part_sims, valid_mask

    def _fuse_average_scores(self, global_sim, part_sims=None, valid_mask=None):
        if part_sims is None or valid_mask is None:
            return global_sim

        part_weights = self._match_part_weights(
            self.search_avg_part_weights.to(part_sims.device),
            part_sims.shape[1],
            default_value=1.0,
        )

        weighted_mask = valid_mask.float() * part_weights.view(1, -1)
        valid_weight_sum = weighted_mask.sum(dim=1)
        if torch.all(valid_weight_sum == 0):
            return global_sim

        part_sum = (part_sims * weighted_mask).sum(dim=1)
        weight = float(self.cerberus_group_score_weight)
        global_weight = self.search_avg_global_weight
        return (global_weight * global_sim + weight * part_sum) / (
            global_weight + weight * valid_weight_sum
        )

    def _rank_positions(self, scores):
        order = torch.argsort(scores, descending=True)
        ranks = torch.empty(
            scores.shape[0], dtype=torch.float32, device=scores.device
        )
        ranks[order] = torch.arange(
            1, scores.shape[0] + 1, dtype=torch.float32, device=scores.device
        )
        return ranks

    @staticmethod
    def _group_part_index(group_name):
        order = {
            "gender": 0,
            "hair": 1,
            "top": 2,
            "pants": 3,
            "shoes": 4,
            "carry": 5,
        }
        return order.get(group_name, -1)

    def _apply_gender_label_shortlist_gate(
        self, fused_scores, global_sim, group_labels, query_group_labels
    ):
        if (
            not self.search_gender_label_shortlist_enabled
            or group_labels is None
            or query_group_labels is None
            or fused_scores.numel() == 0
        ):
            return fused_scores

        gender_idx = self._group_part_index("gender")
        if gender_idx < 0 or gender_idx >= group_labels.shape[1]:
            return fused_scores
        query_gender = int(query_group_labels[gender_idx].item())
        # Treat unknown/other as neutral: only male/female can trigger a hard conflict.
        if query_gender not in (0, 1):
            return fused_scores

        gallery_gender = group_labels[:, gender_idx]
        known_mask = (gallery_gender == 0) | (gallery_gender == 1)
        if not known_mask.any():
            return fused_scores

        keep_mask = (~known_mask) | (gallery_gender == query_gender)
        if keep_mask.all() or not keep_mask.any():
            return fused_scores

        fused = fused_scores.clone()
        drop_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()
        fused[drop_indices] = fused[drop_indices].new_full((drop_indices.numel(),), -1e6)
        return fused

    def _resolve_post_shortlist_size(self, override_size, total_size):
        shortlist_size = (
            self.search_shortlist_size if int(override_size) <= 0 else int(override_size)
        )
        return min(max(shortlist_size, 1), int(total_size))

    def _apply_top_pants_soft_rerank(self, fused_scores, global_sim, part_sims, valid_mask):
        if (
            not self.search_top_pants_soft_rerank_enabled
            or part_sims is None
            or valid_mask is None
            or fused_scores.numel() == 0
        ):
            return fused_scores

        if (
            self.search_top_pants_soft_rerank_uncertain_margin > 0
            and global_sim.numel() >= 2
        ):
            top2 = torch.topk(global_sim, k=2, largest=True).values
            global_margin = float((top2[0] - top2[1]).item())
            if global_margin > self.search_top_pants_soft_rerank_uncertain_margin:
                return fused_scores

        shortlist_size = self._resolve_post_shortlist_size(
            self.search_top_pants_soft_rerank_size, global_sim.numel()
        )
        shortlist = torch.argsort(global_sim, descending=True)[:shortlist_size]
        if (
            self.search_top_pants_soft_rerank_score_band > 0
            and shortlist.numel() > 0
        ):
            top_score = global_sim[shortlist[0]]
            band_mask = (top_score - global_sim[shortlist]) <= self.search_top_pants_soft_rerank_score_band
            shortlist = shortlist[band_mask]
            if shortlist.numel() == 0:
                return fused_scores
        fused = fused_scores.clone()

        for group_name, weight, center in (
            ("top", self.search_top_soft_weight, self.search_top_soft_center),
            ("pants", self.search_pants_soft_weight, self.search_pants_soft_center),
        ):
            if weight == 0:
                continue
            part_idx = self._group_part_index(group_name)
            if part_idx < 0 or part_idx >= part_sims.shape[1]:
                continue
            shortlist_valid = valid_mask[shortlist, part_idx] > 0
            if not shortlist_valid.any():
                continue
            active = shortlist[shortlist_valid]
            part_scale = valid_mask[active, part_idx].float()
            if self.cuhk_search_top_pants_positive_only:
                delta = weight * part_scale * torch.relu(part_sims[active, part_idx] - center)
            else:
                delta = weight * part_scale * (part_sims[active, part_idx] - center)
            if self.search_top_pants_soft_rerank_max_delta > 0:
                delta = torch.clamp(
                    delta,
                    min=-self.search_top_pants_soft_rerank_max_delta,
                    max=self.search_top_pants_soft_rerank_max_delta,
                )
            fused[active] = fused[active] + delta
        return fused

    def _select_vote_parts(self, part_sims, valid_mask):
        if part_sims is None or valid_mask is None:
            return []
        candidate_parts = torch.nonzero(valid_mask.gt(0).any(dim=0), as_tuple=False).flatten()
        if candidate_parts.numel() == 0:
            return []

        weights = self._match_part_weights(
            self.search_part_vote_weights.to(part_sims.device),
            part_sims.shape[1],
        )
        scored_parts = []
        for part_idx in candidate_parts.tolist():
            if weights[part_idx] <= 0:
                continue
            valid_vals = part_sims[:, part_idx][valid_mask[:, part_idx] > 0]
            if valid_vals.numel() == 0:
                continue
            dispersion = valid_vals.std(unbiased=False).item()
            scored_parts.append((dispersion * float(weights[part_idx]), part_idx))

        scored_parts.sort(reverse=True)
        if self.search_max_parts > 0:
            scored_parts = scored_parts[: self.search_max_parts]
        return [part_idx for _, part_idx in scored_parts]

    def _fuse_vote_scores(self, global_sim, part_sims=None, valid_mask=None):
        fused = self.search_global_vote_weight / (
            self.search_rrf_k + self._rank_positions(global_sim)
        )
        fused = fused + global_sim.float() * 1e-6

        if part_sims is None or valid_mask is None or global_sim.numel() == 0:
            return fused

        shortlist_size = min(self.search_shortlist_size, global_sim.numel())
        shortlist = torch.argsort(global_sim, descending=True)[:shortlist_size]
        selected_parts = self._select_vote_parts(part_sims[shortlist], valid_mask[shortlist])
        if not selected_parts:
            return fused

        weights = self._match_part_weights(
            self.search_part_vote_weights.to(global_sim.device),
            part_sims.shape[1],
        )
        for part_idx in selected_parts:
            shortlist_valid = valid_mask[shortlist, part_idx] > 0
            if not shortlist_valid.any():
                continue
            active_indices = shortlist[shortlist_valid]
            part_scores = part_sims[active_indices, part_idx]
            part_scale = valid_mask[active_indices, part_idx].float()
            fused[active_indices] += part_scale * weights[part_idx] / (
                self.search_rrf_k + self._rank_positions(part_scores)
            )
        return fused

    def _fuse_shortlist_scores(self, global_sim, part_sims=None, valid_mask=None):
        if (
            part_sims is None
            or valid_mask is None
            or global_sim.numel() == 0
            or self.search_shortlist_size <= 0
        ):
            return global_sim

        shortlist_size = min(self.search_shortlist_size, global_sim.numel())
        shortlist = torch.argsort(global_sim, descending=True)[:shortlist_size]
        fused = global_sim.clone()
        fused_short = self._fuse_average_scores(
            global_sim[shortlist], part_sims[shortlist], valid_mask[shortlist]
        )
        # Preserve the candidate pool selected by global retrieval and only rerank within it.
        fused[shortlist] = fused_short
        return fused

    def _fuse_bonus_penalty_scores(self, global_sim, part_sims=None, valid_mask=None):
        if part_sims is None or valid_mask is None:
            return global_sim

        part_weights = self._match_part_weights(
            self.search_avg_part_weights.to(part_sims.device),
            part_sims.shape[1],
            default_value=1.0,
        )
        weighted_mask = valid_mask.float() * part_weights.view(1, -1)
        if torch.all(weighted_mask == 0):
            return global_sim

        pos_component = torch.relu(part_sims - self.search_bp_pos_threshold)
        neg_component = torch.relu(self.search_bp_neg_threshold - part_sims)

        pos_bonus = (pos_component * weighted_mask).sum(dim=1)
        neg_penalty = (neg_component * weighted_mask).sum(dim=1)

        return (
            self.search_avg_global_weight * global_sim
            + self.search_bp_pos_weight * pos_bonus
            - self.search_bp_neg_weight * neg_penalty
        )

    def _fuse_query_candidates(self, candidate_chunks, query_group_labels=None):
        if not candidate_chunks:
            return []

        global_sims = [chunk["global_sim"] for chunk in candidate_chunks]
        split_sizes = [sim.shape[0] for sim in global_sims]
        global_cat = torch.cat(global_sims, dim=0)
        group_label_chunks = [chunk.get("group_labels") for chunk in candidate_chunks]

        # Coarse-to-Fine: 先用全局特征筛选 top-K
        if hasattr(self, 'coarse_topk') and self.coarse_topk > 0 and global_cat.shape[0] > self.coarse_topk:
            topk_indices = torch.topk(global_cat, k=min(self.coarse_topk, global_cat.shape[0]), largest=True).indices
            topk_mask = torch.zeros_like(global_cat, dtype=torch.bool)
            topk_mask[topk_indices] = True

            # 只对 top-K 计算属性得分
            filtered_chunks = []
            offset = 0
            for i, chunk in enumerate(candidate_chunks):
                chunk_size = split_sizes[i]
                chunk_mask = topk_mask[offset:offset+chunk_size]
                if chunk_mask.any():
                    filtered_chunks.append({
                        "global_sim": chunk["global_sim"][chunk_mask],
                        "part_sims": chunk["part_sims"][chunk_mask] if chunk["part_sims"] is not None else None,
                        "valid_mask": chunk["valid_mask"][chunk_mask] if chunk["valid_mask"] is not None else None,
                        "group_labels": chunk["group_labels"][chunk_mask] if chunk.get("group_labels") is not None else None,
                    })
                offset += chunk_size

            # 重新计算
            global_sims = [chunk["global_sim"] for chunk in filtered_chunks]
            split_sizes = [sim.shape[0] for sim in global_sims]
            global_cat = torch.cat(global_sims, dim=0)
            candidate_chunks = filtered_chunks
            group_label_chunks = [chunk.get("group_labels") for chunk in candidate_chunks]

        part_dim = max(
            [
                chunk["part_sims"].shape[1]
                for chunk in candidate_chunks
                if chunk["part_sims"] is not None
            ]
            or [self.search_part_vote_weights.numel()]
        )
        part_cat = None
        valid_cat = None
        if part_dim > 0:
            part_chunks = []
            valid_chunks = []
            for chunk in candidate_chunks:
                if chunk["part_sims"] is None or chunk["valid_mask"] is None:
                    part_chunks.append(
                        global_cat.new_zeros((chunk["global_sim"].shape[0], part_dim))
                    )
                    valid_chunks.append(
                        global_cat.new_zeros((chunk["global_sim"].shape[0], part_dim))
                    )
                elif chunk["part_sims"].shape[1] < part_dim:
                    pad_dim = part_dim - chunk["part_sims"].shape[1]
                    part_pad = chunk["part_sims"].new_zeros((chunk["part_sims"].shape[0], pad_dim))
                    valid_pad = chunk["valid_mask"].new_zeros((chunk["valid_mask"].shape[0], pad_dim))
                    part_chunks.append(torch.cat([chunk["part_sims"], part_pad], dim=1))
                    valid_chunks.append(torch.cat([chunk["valid_mask"], valid_pad], dim=1))
                else:
                    part_chunks.append(chunk["part_sims"])
                    valid_chunks.append(chunk["valid_mask"])
            part_cat = torch.cat(part_chunks, dim=0)
            valid_cat = torch.cat(valid_chunks, dim=0)

        group_label_cat = None
        if any(label is not None for label in group_label_chunks):
            label_dim = max(
                [labels.shape[1] for labels in group_label_chunks if labels is not None] or [0]
            )
            if label_dim > 0:
                filled = []
                for chunk in candidate_chunks:
                    labels = chunk.get("group_labels")
                    if labels is None:
                        filled.append(
                            global_cat.new_full(
                                (chunk["global_sim"].shape[0], label_dim), -1
                            ).long()
                        )
                    elif labels.shape[1] < label_dim:
                        pad = labels.new_full((labels.shape[0], label_dim - labels.shape[1]), -1)
                        filled.append(torch.cat([labels, pad], dim=1))
                    else:
                        filled.append(labels)
                group_label_cat = torch.cat(filled, dim=0)

        if self.search_fusion_mode == "global":
            fused_cat = global_cat
        else:
            if self.search_fusion_mode == "avg":
                fused_cat = self._fuse_average_scores(global_cat, part_cat, valid_cat)
            elif self.search_fusion_mode == "shortlist":
                fused_cat = self._fuse_shortlist_scores(global_cat, part_cat, valid_cat)
            elif self.search_fusion_mode == "bonus_penalty":
                fused_cat = self._fuse_bonus_penalty_scores(global_cat, part_cat, valid_cat)
            elif self.search_fusion_mode == "kde":
                fused_cat = self._fuse_kde_scores(global_cat, part_cat, valid_cat)
            else:
                fused_cat = self._fuse_vote_scores(global_cat, part_cat, valid_cat)

        fused_cat = self._apply_gender_label_shortlist_gate(
            fused_cat, global_cat, group_label_cat, query_group_labels
        )
        fused_cat = self._apply_top_pants_soft_rerank(
            fused_cat, global_cat, part_cat, valid_cat
        )

        return list(torch.split(fused_cat, split_sizes))

    def _fuse_kde_scores(self, global_sim, part_sims=None, valid_mask=None):
        """KDE density-boosted fusion: estimate per-group local density via KNN
        Gaussian KDE, then multiplicatively boost part scores in high-density
        regions before weighted averaging with global similarity."""
        N = global_sim.shape[0]
        G = part_sims.shape[1] if part_sims is not None else 0

        # Fallback: too few detections or no part info
        if N < 3 or part_sims is None or valid_mask is None or G == 0:
            return global_sim

        part_weights = self._match_part_weights(
            self.search_avg_part_weights.to(part_sims.device),
            G,
            default_value=1.0,
        )
        kde_w = self.search_kde_density_weight
        top_k = min(self.search_kde_top_k, N - 1)

        # Density matrix: [N, G]
        density = torch.zeros_like(part_sims)

        for g in range(G):
            valid_g = valid_mask[:, g] > 0  # [N] bool
            n_valid = int(valid_g.sum().item())
            if n_valid < 3:
                continue

            scores_g = part_sims[valid_g, g]  # [n_valid]

            # Bandwidth: Silverman's rule or fixed
            if self.search_kde_bandwidth > 0:
                h = self.search_kde_bandwidth
            else:
                std_val = scores_g.std(unbiased=False).item()
                sorted_g = scores_g.sort().values
                q75_idx = min(int(n_valid * 0.75), n_valid - 1)
                q25_idx = min(int(n_valid * 0.25), n_valid - 1)
                iqr = (sorted_g[q75_idx] - sorted_g[q25_idx]).item()
                spread = min(std_val, iqr / 1.34) if iqr > 0 else std_val
                h = 0.9 * spread * (n_valid ** (-0.2))
                if h < 1e-8:
                    h = 1e-8

            # Pairwise 1D distances: [n_valid, n_valid]
            dists = (scores_g.unsqueeze(1) - scores_g.unsqueeze(0)).abs()

            # KNN: for each point, take top_k nearest (exclude self via +inf diagonal)
            k_actual = min(top_k, n_valid - 1)
            dists.fill_diagonal_(float("inf"))
            knn_dists, _ = dists.topk(k_actual, dim=1, largest=False)  # [n_valid, k_actual]

            # Gaussian kernel density
            knn_density = torch.exp(-0.5 * (knn_dists / h) ** 2).mean(dim=1)  # [n_valid]

            # Normalize to [0, 1]
            d_min = knn_density.min()
            d_max = knn_density.max()
            if d_max - d_min > 1e-8:
                knn_density = (knn_density - d_min) / (d_max - d_min)
            else:
                knn_density.zero_()

            density[valid_g, g] = knn_density

        # Adjusted part scores: multiplicative boost
        adjusted_parts = part_sims * (1.0 + kde_w * density) * valid_mask.float()

        # Weighted fusion (reuse avg config)
        weighted_mask = valid_mask.float() * part_weights.view(1, -1)
        # Scale weights by density boost for normalization consistency
        adjusted_weighted = (1.0 + kde_w * density) * weighted_mask
        valid_weight_sum = adjusted_weighted.sum(dim=1)

        if torch.all(valid_weight_sum == 0):
            return global_sim

        part_sum = (adjusted_parts * part_weights.view(1, -1)).sum(dim=1)
        cerb_w = float(self.cerberus_group_score_weight)
        global_w = self.search_avg_global_weight
        return (global_w * global_sim + cerb_w * part_sum) / (
            global_w + cerb_w * valid_weight_sum
        )
