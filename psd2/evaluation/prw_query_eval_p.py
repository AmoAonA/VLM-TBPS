from .query_evaluator import QueryEvaluator
from psd2.structures import Boxes, BoxMode, pairwise_iou
import torch
import logging
import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score
import copy
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_img_cid(img_id):
    cn = re.match(r"c(\d+)s.*", img_id)
    cn = int(cn.groups()[0])
    return cn


class PrwQueryEvaluatorP(QueryEvaluator):
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}
        self.inf_results = []

    def reset(self):
        super().reset()
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}
        self.inf_results = []
        self._maybe_load_query_cache()

    def _load_query_cache_payload(self, payload):
        self.inf_results = payload
        return True

    def _get_query_cache_payload(self):
        return self.inf_results

    def process(self, inputs, outputs):
        if self._cached_query_ready:
            return
        """
        Args:
            inputs:
                a batch of
                { "query":
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
                }
            outputs:
                a batch of instances with attrs
                {
                    reid_feats: tensor (1,pfeat_dim)
                }
                or
                a batch of empty instances
        """
        # NOTE save pairs (query gt instances, query pred instances)
        save_inputs = []
        for item_in, inst_pred in zip(inputs, outputs):
            q_gt_instances = item_in["query"]["instances"].to(self._cpu_device)
            q_pred_instances = inst_pred.to(self._cpu_device)
            save_inputs.append((q_gt_instances, q_pred_instances))
        self.inf_results.append(save_inputs)

    def evaluate(self):
        eval_dataset = EvaluatorDataset(self.inf_results, self)
        eval_worker = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x,
        )
        logger.info("Parallel evaluating on {}:".format(self.dataset_name))
        with tqdm(total=len(eval_worker)) as pbar:
            for b_rst in eval_worker:
                (
                    rst_aps,
                    rst_accs,
                    rst_aps_mlv,
                    rst_accs_mlv,
                ) = b_rst[0]
                for st in self.det_score_thresh:
                    self.aps[st].extend(rst_aps[st])
                    self.accs[st].extend(rst_accs[st])
                    self.aps_mlv[st].extend(rst_aps_mlv[st])
                    self.accs_mlv[st].extend(rst_accs_mlv[st])
                pbar.update(1)
        self._maybe_save_query_cache()
        mix_eval_results = super().evaluate()
        aps = self.aps_mlv
        accs = self.accs_mlv

        for dst in self.det_score_thresh:
            logger.info(
                "Multi-view Search eval_{:.2f} on {} queries. ".format(
                    dst, len(aps[dst])
                )
            )
            mAP = np.mean(aps[dst])
            mix_eval_results["search"].update({"mAP_{:.2f}_mlv".format(dst): mAP})
            acc = np.mean(accs[dst], axis=0)
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                mix_eval_results["search"].update(
                    {"top{:2d}_{:.2f}_mlv".format(k, dst): v}
                )
        return copy.deepcopy(mix_eval_results)



class EvaluatorDataset(Dataset):
    def __init__(self, eval_inputs, eval_ref):
        self.eval_inputs = eval_inputs
        self.eval_ref = eval_ref

    def __len__(self):
        return len(self.eval_inputs)

    def __getitem__(self, idx):
        inputs = self.eval_inputs[idx]
        rst_aps_mlv = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_accs_mlv = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_aps = {st: [] for st in self.eval_ref.det_score_thresh}
        rst_accs = {st: [] for st in self.eval_ref.det_score_thresh}
        for bi, in_dict in enumerate(inputs):
            q_gt_instances, q_pred_instances = in_dict
            q_imgid = q_gt_instances.image_id
            q_pid = q_gt_instances.gt_pids
            q_pid_value = int(q_pid.item())
            q_box = q_gt_instances.org_gt_boxes
            q_cid = _get_img_cid(q_imgid)
            y_trues = {dst: [] for dst in self.eval_ref.det_score_thresh}
            y_scores = {dst: [] for dst in self.eval_ref.det_score_thresh}
            count_gts = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            count_tps = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            y_trues_mlv = {dst: [] for dst in self.eval_ref.det_score_thresh}
            y_scores_mlv = {dst: [] for dst in self.eval_ref.det_score_thresh}
            count_gts_mlv = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            count_tps_mlv = {dst: 0 for dst in self.eval_ref.det_score_thresh}
            # Find all occurence of this query
            # Construct gallery set for this query
            gallery_imgs = []
            query_gts = {}
            for gt_img_id, gt_img_label in self.eval_ref.gts.items():
                if gt_img_id != q_imgid:
                    gt_boxes_t = gt_img_label[:, :4]
                    gt_ids = gt_img_label[:, 4].long()
                    gallery_imgs.append(
                        {
                            "image_id": gt_img_id,
                            "boxes_t": gt_boxes_t,
                            "ids": gt_ids,
                        }
                    )
                    if q_pid_value in gt_ids.tolist():
                        query_gts[gt_img_id] = gt_boxes_t[gt_ids == q_pid_value].squeeze(0)

            for dst in self.eval_ref.det_score_thresh:
                feat_q, query_group_feats, query_group_valid, query_group_labels = self.eval_ref._resolve_query_features(
                    q_pred_instances, q_imgid, q_box, dst
                )
                if feat_q is None:
                    logger.warning("Undetected query person in {} !".format(q_imgid))
                    continue

                candidate_chunks = []
                # 1. Go through the gallery samples defined by the protocol
                for item in gallery_imgs:
                    gallery_imname = item["image_id"]
                    g_cid = _get_img_cid(gallery_imname)
                    # some contain the query (gt not empty), some not
                    count_gts[dst] += gallery_imname in query_gts
                    if g_cid != q_cid:
                        # some contain the query (gt not empty), some not
                        count_gts_mlv[dst] += gallery_imname in query_gts
                    # compute distance between query and gallery dets
                    if gallery_imname not in self.eval_ref.infs:
                        continue
                    det, feat_g = self.eval_ref._get_gallery_dets(
                        gallery_imname, dst
                    ), self.eval_ref._get_gallery_feats(gallery_imname, dst)
                    # no detection in this gallery, skip it
                    if det.shape[0] == 0:
                        continue
                    global_sim, part_sims, valid_mask = self.eval_ref._compute_similarity_components(
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
                            "group_labels": self.eval_ref._get_gallery_group_labels(gallery_imname, dst),
                            "gt": query_gts.get(gallery_imname),
                        }
                    )

                fused_scores = self.eval_ref._fuse_query_candidates(
                    candidate_chunks, query_group_labels=query_group_labels
                )
                for chunk, sim in zip(candidate_chunks, fused_scores):
                    det = chunk["det"]
                    gallery_imname = chunk["gallery_imname"]
                    g_cid = chunk["gallery_cid"]
                    gt = chunk["gt"]

                    label = torch.zeros(sim.shape[0], dtype=torch.int)
                    if gt is not None:
                        w, h = gt[2] - gt[0], gt[3] - gt[1]
                        iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                        inds = torch.argsort(sim, descending=True)
                        sim = sim[inds]
                        det = det[inds]
                        for j, roi in enumerate(det[:, :4]):
                            if box_iou(roi[None, :], gt[None, :]).squeeze().item() >= iou_thresh:
                                label[j] = 1
                                count_tps[dst] += 1
                                if q_cid != g_cid:
                                    count_tps_mlv[dst] += 1
                                break
                    y_trues[dst].extend(label.tolist())
                    y_scores[dst].extend(sim.tolist())
                    if g_cid != q_cid:
                        y_trues_mlv[dst].extend(label.tolist())
                        y_scores_mlv[dst].extend(sim.tolist())

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
                rst_aps[dst].append(ap)
                inds = np.argsort(y_score)[::-1]
                y_score = y_score[inds]
                y_true = y_true[inds]
                rst_accs[dst].append(
                    [min(1, sum(y_true[:k])) for k in self.eval_ref.topks]
                )
                # mlv
                y_score_mlv = np.asarray(y_scores_mlv[dst])
                y_true_mlv = np.asarray(y_trues_mlv[dst])
                assert count_tps_mlv[dst] <= count_gts_mlv[dst]
                recall_rate_mlv = count_tps_mlv[dst] * 1.0 / count_gts_mlv[dst]
                ap_mlv = (
                    0
                    if count_tps_mlv[dst] == 0
                    else average_precision_score(y_true_mlv, y_score_mlv)
                    * recall_rate_mlv
                )

                rst_aps_mlv[dst].append(ap_mlv)
                inds_mlv = np.argsort(y_score_mlv)[::-1]
                y_score_mlv = y_score_mlv[inds_mlv]
                y_true_mlv = y_true_mlv[inds_mlv]
                rst_accs_mlv[dst].append(
                    [min(1, sum(y_true_mlv[:k])) for k in self.eval_ref.topks]
                )
        return (
            rst_aps,
            rst_accs,
            rst_aps_mlv,
            rst_accs_mlv,

        )
