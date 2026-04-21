import copy
import logging

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .aps_query_eval import APSQueryEvaluator

logger = logging.getLogger(__name__)


class CuhkAPSQueryEvaluator(APSQueryEvaluator):
    def process(self, inputs, outputs):
        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_instances = query_dict["instances"].to(self._cpu_device)
            g_instances_list = in_dict["gallery"]
            q_imgid = q_instances.image_id
            q_box = q_instances.org_gt_boxes
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
                candidate_chunks = []

                for item in g_instances_list:
                    gallery_imname = item.image_id
                    gt_boxes = self._get_positive_gt_boxes(gallery_imname, q_attributes)
                    if len(gt_boxes) > 0:
                        count_gts += 1
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
                                break
                    y_trues.extend(label.tolist())
                    y_scores.extend(sim.tolist())

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

    def evaluate(self):
        return copy.deepcopy(super().evaluate())


class CuhkAPSQueryEvaluatorP(CuhkAPSQueryEvaluator):
    pass
