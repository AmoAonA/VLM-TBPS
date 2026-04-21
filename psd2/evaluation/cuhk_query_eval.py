import torch
import logging

import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score

import logging
from .query_evaluator import QueryEvaluator
from psd2.structures import Boxes, BoxMode, pairwise_iou

logger = logging.getLogger(__name__)  # setup_logger()


class CuhkQueryEvaluator(QueryEvaluator):
    def process(self, inputs, outputs):
        """
        Args:
            inputs:
                a batch of
                {
                    "query":
                    {
                        "image": augmented image tensor
                        "instances": an Instances object with attrs
                            {
                                image_size: hw (int, int)
                                file_name: full path string
                                image_id: filename string
                                gt_boxes: Boxes (1 , 4)
                                gt_classes: tensor full with 0s
                                gt_pids: person identity tensor (1,)
                                org_img_size: hw before augmentation (int, int)
                                org_gt_boxes: Boxes before augmentation
                            }
                    }
                    "gallery":
                    {
                        "instances_list": a gallery of Instances objects
                        [
                            Instances object with attr:
                                file_name,
                                image_id,
                                gt_boxes: (1,4) box of the true positive
                                gt_pids: (1,) query pid
                            ...
                        ]
                    }
                }
            outputs:
                a batch of instances with attrs
                {
                    reid_feats: tensor (1,pfeat_dim)
                }
                or
                a batch of empty instances
        """
        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_instances = query_dict["instances"].to(self._cpu_device)
            g_instances_list = in_dict["gallery"]
            q_imgid = q_instances.image_id
            q_pid = q_instances.gt_pids
            q_box = q_instances.org_gt_boxes
            y_trues = {dst: [] for dst in self.det_score_thresh}
            y_scores = {dst: [] for dst in self.det_score_thresh}
            count_gts = {dst: 0 for dst in self.det_score_thresh}
            count_tps = {dst: 0 for dst in self.det_score_thresh}
            for dst in self.det_score_thresh:
                pred_instances = outputs[bi].to(self._cpu_device)
                feat_q, query_group_feats, query_group_valid, query_group_labels = self._resolve_query_features(
                    pred_instances, q_imgid, q_box, dst
                )
                if feat_q is None:
                    logger.warning("Undetected query person in {} !".format(q_imgid))
                    continue

                candidate_chunks = []


                # 1. Go through the gallery samples defined by the protocol
                for item in g_instances_list:
                    gallery_imname = item.image_id
                    # some contain the query (gt not empty), some not
                    gt_boxes = item.org_gt_boxes
                    count_gts[dst] += len(gt_boxes) > 0
                    # compute distance between query and gallery dets
                    if gallery_imname not in self.infs:
                        continue
                    det, feat_g = self._get_gallery_dets(
                        gallery_imname, dst
                    ), self._get_gallery_feats(gallery_imname, dst)
                    # no detection in this gallery, skip it
                    if det.shape[0] == 0:
                        continue
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
                    if len(gt_boxes) > 0:
                        hw = gt_boxes.get_sizes().squeeze(0)
                        iou_thresh = min(
                            0.5, (hw[1] * hw[0] * 1.0) / ((hw[1] + 10) * (hw[0] + 10))
                        )
                        inds = torch.argsort(sim, descending=True)
                        sim = sim[inds]
                        det = det[inds]
                        for j, roi in enumerate(det[:, :4]):
                            if box_iou(roi[None, :], gt_boxes.tensor).squeeze().item() >= iou_thresh:
                                label[j] = 1
                                count_tps[dst] += 1
                                break

                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())

                # 2. Compute AP for this probe (need to scale by recall rate)

                y_score = np.asarray(y_scores[dst])
                y_true = np.asarray(y_trues[dst])
                assert count_tps[dst] <= count_gts[dst]
                recall_rate = count_tps[dst] * 1.0 / count_gts[dst]
                ap = (
                    0
                    if count_tps[dst] == 0
                    else average_precision_score(y_true, y_score) * recall_rate
                )
                self.aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true_o = y_true[inds]
                self.accs[dst].append([min(1, sum(y_true_o[:k])) for k in self.topks])
