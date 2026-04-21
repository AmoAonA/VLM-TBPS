import copy
import itertools
import logging
import re

import numpy as np
import psd2.utils.comm as comm
import torch
from sklearn.metrics import average_precision_score

from .aps_query_eval import APSQueryEvaluator

logger = logging.getLogger(__name__)


def _get_img_cid(img_id):
    cn = re.match(r"c(\d+)s.*", img_id)
    cn = int(cn.groups()[0])
    return cn


class PrwAPSQueryEvaluator(APSQueryEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}

    def reset(self):
        super().reset()
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}

    def process(self, inputs, outputs):
        for bi, in_dict in enumerate(inputs):
            q_instances = in_dict["query"]["instances"].to(self._cpu_device)
            q_imgid = q_instances.image_id
            q_box = q_instances.org_gt_boxes
            q_cid = _get_img_cid(q_imgid)
            q_attributes = self._extract_query_attributes(q_instances)
            if self._count_valid_groups(q_attributes) == 0:
                logger.warning("Skip APS query without valid attributes in %s", q_imgid)
                continue

            for dst in self.det_score_thresh:
                pred_instances = outputs[bi].to(self._cpu_device)
                feat_q, query_group_feats, query_group_valid, query_group_labels = self._resolve_query_features(
                    pred_instances, q_imgid, q_box, dst
                )
                if feat_q is None:
                    logger.warning("Undetected APS query in %s", q_imgid)
                    continue

                y_trues = []
                y_scores = []
                count_gts = 0
                count_tps = 0
                y_trues_mlv = []
                y_scores_mlv = []
                count_gts_mlv = 0
                count_tps_mlv = 0
                candidate_chunks = []

                for gallery_imname in self.gts.keys():
                    if gallery_imname == q_imgid:
                        continue

                    g_cid = _get_img_cid(gallery_imname)
                    gt_boxes = self._get_positive_gt_boxes(gallery_imname, q_attributes)
                    if len(gt_boxes) > 0:
                        count_gts += 1
                        if g_cid != q_cid:
                            count_gts_mlv += 1

                    if gallery_imname not in self.infs:
                        continue
                    det = self._get_gallery_dets(gallery_imname, dst)
                    if det.shape[0] == 0:
                        continue
                    feat_g = self._get_gallery_feats(gallery_imname, dst)
                    global_sim, part_sims, valid_mask = self._compute_similarity_components(
                        feat_g,
                        feat_q,
                        gallery_imname,
                        dst,
                        query_group_feats,
                        query_group_valid,
                    )
                    candidate_chunks.append(
                        {
                            "gallery_imname": gallery_imname,
                            "gallery_cid": g_cid,
                            "det": det,
                            "global_sim": global_sim,
                            "part_sims": part_sims,
                            "valid_mask": valid_mask,
                            "group_labels": self._get_gallery_group_labels(gallery_imname, dst),
                            "gt_boxes": gt_boxes,
                        }
                    )

                fused_scores = self._fuse_query_candidates(
                    candidate_chunks, query_group_labels=query_group_labels
                )
                for chunk, sim in zip(candidate_chunks, fused_scores):
                    det = chunk["det"]
                    gt_boxes = chunk["gt_boxes"]
                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    inds = torch.argsort(sim, descending=True)
                    sim = sim[inds]
                    det_sorted = det[inds]
                    if len(gt_boxes) > 0:
                        for j, roi in enumerate(det_sorted[:, :4]):
                            if self._match_detection_to_gt(roi, gt_boxes):
                                label[j] = 1
                                count_tps += 1
                                if chunk["gallery_cid"] != q_cid:
                                    count_tps_mlv += 1
                                break
                    y_trues.extend(label.tolist())
                    y_scores.extend(sim.tolist())
                    if chunk["gallery_cid"] != q_cid:
                        y_trues_mlv.extend(label.tolist())
                        y_scores_mlv.extend(sim.tolist())

                y_score = np.asarray(y_scores)
                y_true = np.asarray(y_trues)
                recall_rate = count_tps * 1.0 / count_gts if count_gts > 0 else 0.0
                ap = (
                    0.0
                    if count_tps == 0 or y_score.size == 0
                    else average_precision_score(y_true, y_score) * recall_rate
                )
                self.aps[dst].append(ap)
                if y_score.size == 0:
                    self.accs[dst].append([0.0 for _ in self.topks])
                else:
                    inds = np.argsort(y_score)[::-1]
                    y_true_sorted = y_true[inds]
                    self.accs[dst].append([min(1, sum(y_true_sorted[:k])) for k in self.topks])

                y_score_mlv = np.asarray(y_scores_mlv)
                y_true_mlv = np.asarray(y_trues_mlv)
                recall_rate_mlv = count_tps_mlv * 1.0 / count_gts_mlv if count_gts_mlv > 0 else 0.0
                ap_mlv = (
                    0.0
                    if count_tps_mlv == 0 or y_score_mlv.size == 0
                    else average_precision_score(y_true_mlv, y_score_mlv) * recall_rate_mlv
                )
                self.aps_mlv[dst].append(ap_mlv)
                if y_score_mlv.size == 0:
                    self.accs_mlv[dst].append([0.0 for _ in self.topks])
                else:
                    inds_mlv = np.argsort(y_score_mlv)[::-1]
                    y_true_mlv_sorted = y_true_mlv[inds_mlv]
                    self.accs_mlv[dst].append(
                        [min(1, sum(y_true_mlv_sorted[:k])) for k in self.topks]
                    )

    def evaluate(self):
        mix_eval_results = super().evaluate()
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps_mlv, dst=0)
            accs_all = comm.gather(self.accs_mlv, dst=0)
            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
        else:
            aps = self.aps_mlv
            accs = self.accs_mlv

        for dst in self.det_score_thresh:
            mAP = np.mean(aps[dst]) if len(aps[dst]) > 0 else 0.0
            mix_eval_results["search"].update({"mAP_{:.2f}_mlv".format(dst): mAP})
            if len(accs[dst]) == 0:
                acc = np.zeros(len(self.topks), dtype=np.float32)
            else:
                acc = np.mean(np.array(accs[dst]), axis=0)
            for i, v in enumerate(acc.tolist()):
                k = self.topks[i]
                mix_eval_results["search"].update({"top{:2d}_{:.2f}_mlv".format(k, dst): v})
        return copy.deepcopy(mix_eval_results)


class PrwAPSQueryEvaluatorP(PrwAPSQueryEvaluator):
    pass
