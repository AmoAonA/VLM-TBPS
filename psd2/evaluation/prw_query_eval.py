from .query_evaluator import QueryEvaluator
from copy import copy
import psd2.utils.comm as comm
import torch
import logging
import itertools
import numpy as np
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score
import copy
import re
from psd2.structures import Boxes, BoxMode, pairwise_iou

logger = logging.getLogger(__name__)


def _get_img_cid(img_id):
    cn = re.match(r"c(\d+)s.*", img_id)
    cn = int(cn.groups()[0])
    return cn


class PrwQueryEvaluator(QueryEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
        **kwargs,
    ):
        super().__init__(
            dataset_name, distributed, output_dir, s_threds, **kwargs
        )
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}
        # 【新增】用于缓存 Query 数据的列表
        self.queries = []

    def reset(self):
        super().reset()
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}
        # 【新增】清空缓存
        self.queries = []
        self._maybe_load_query_cache()

    def _load_query_cache_payload(self, payload):
        self.queries = payload
        return True

    def _get_query_cache_payload(self):
        return self.queries

    def process(self, inputs, outputs):
        if self._cached_query_ready:
            return
        """
        Args:
            inputs: a batch of query dicts
            outputs: a batch of instances (predictions)
        """
        # 【关键修改】Process 阶段只收集数据，不做计算，防止阻塞主进程
        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_instances = query_dict["instances"].to(self._cpu_device)

            # 提取并缓存 Query 元数据
            q_data = {
                "q_imgid": q_instances.image_id,
                "q_pid": q_instances.gt_pids,
                "q_box": q_instances.org_gt_boxes,
                "q_cid": _get_img_cid(q_instances.image_id),
                "pred_instances": outputs[bi].to(self._cpu_device)
            }
            self.queries.append(q_data)

    def evaluate(self):
        # 初始化结果容器
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}
        self.aps_mlv = {st: [] for st in self.det_score_thresh}
        self.accs_mlv = {st: [] for st in self.det_score_thresh}

        # 【新增】构建 PID -> ImageIDs 的倒排索引，加速 GT 统计
        logger.info("Building PID index for fast retrieval...")
        pid_to_imgids = {}
        for gt_img_id, gt_img_label in self.gts.items():
            gt_ids = gt_img_label[:, 4].long().tolist()
            for pid in gt_ids:
                if pid not in pid_to_imgids:
                    pid_to_imgids[pid] = []
                pid_to_imgids[pid].append(gt_img_id)

        # 针对每个阈值进行集中评估
        for dst in self.det_score_thresh:
            logger.info(f"Preparing Gallery Cache for threshold {dst}...")

            # 【关键修改】预构建 Gallery 缓存
            # 避免在 N 个 Query 循环中重复进行 N * M 次的 Tensor 切片和索引查找
            gallery_cache = []
            for gt_img_id, gt_img_label in self.gts.items():
                if gt_img_id not in self.infs:
                    continue

                det = self._get_gallery_dets(gt_img_id, dst)
                if det.shape[0] == 0:
                    continue

                feat_g = self._get_gallery_feats(gt_img_id, dst)
                gt_ids = gt_img_label[:, 4].long()

                gallery_cache.append({
                    "image_id": gt_img_id,
                    "cid": _get_img_cid(gt_img_id),
                    "det": det,
                    "feat_g": feat_g,
                    "gt_boxes_t": gt_img_label[:, :4],
                    "gt_ids": gt_ids
                })

            logger.info(
                f"Evaluating {len(self.queries)} queries against {len(gallery_cache)} gallery images (thresh={dst})...")

            # 批量处理 Query
            for q_data in self.queries:
                q_imgid = q_data["q_imgid"]
                q_pid = q_data["q_pid"]
                q_pid_value = int(q_pid.item())
                q_box = q_data["q_box"]
                q_cid = q_data["q_cid"]
                pred_instances = q_data["pred_instances"]

                y_trues = []
                y_scores = []
                count_tps = 0
                y_trues_mlv = []
                y_scores_mlv = []
                count_tps_mlv = 0

                # 提取 Query 特征
                feat_q, query_group_feats, query_group_valid, query_group_labels = self._resolve_query_features(
                    pred_instances, q_imgid, q_box, dst
                )
                if feat_q is None:
                    continue

                candidate_chunks = []
                # 快速计算 count_gts (Total Positives)
                # 使用倒排索引瞬间完成，替代原来的全库遍历
                all_gt_imgs = pid_to_imgids.get(q_pid_value, [])
                count_gts = len([img for img in all_gt_imgs if img != q_imgid])

                # 计算 MLV count_gts
                count_gts_mlv = 0
                for img in all_gt_imgs:
                    if img != q_imgid and _get_img_cid(img) != q_cid:
                        count_gts_mlv += 1

                # 遍历缓存的 Gallery 进行匹配
                for item in gallery_cache:
                    gallery_imname = item["image_id"]
                    if gallery_imname == q_imgid:
                        continue

                    feat_g = item["feat_g"]
                    global_sim, part_sims, valid_mask = self._compute_similarity_components(
                        feat_g,
                        feat_q,
                        gallery_imname,
                        dst,
                        query_group_feats,
                        query_group_valid,
                    )
                    gt_ids = item["gt_ids"]
                    gt_box_for_q = (
                        item["gt_boxes_t"][gt_ids == q_pid_value].squeeze(0)
                        if q_pid_value in gt_ids.tolist()
                        else None
                    )
                    candidate_chunks.append(
                        {
                            "gallery_imname": gallery_imname,
                            "gallery_cid": item["cid"],
                            "det": item["det"],
                            "global_sim": global_sim,
                            "part_sims": part_sims,
                            "valid_mask": valid_mask,
                            "group_labels": self._get_gallery_group_labels(gallery_imname, dst),
                            "gt": gt_box_for_q,
                        }
                    )

                fused_scores = self._fuse_query_candidates(
                    candidate_chunks, query_group_labels=query_group_labels
                )
                for chunk, sim in zip(candidate_chunks, fused_scores):
                    det = chunk["det"]
                    gt = chunk["gt"]
                    label = torch.zeros(sim.shape[0], dtype=torch.int)

                    if gt is not None:
                        w, h = gt[2] - gt[0], gt[3] - gt[1]
                        iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                        inds = torch.argsort(sim, descending=True)
                        sim = sim[inds]
                        det_sorted = det[inds]
                        for j, roi in enumerate(det_sorted[:, :4]):
                            if box_iou(roi[None, :], gt[None, :]).squeeze().item() >= iou_thresh:
                                label[j] = 1
                                count_tps += 1
                                if q_cid != chunk["gallery_cid"]:
                                    count_tps_mlv += 1
                                break

                    y_trues.extend(label.tolist())
                    y_scores.extend(sim.tolist())

                    if chunk["gallery_cid"] != q_cid:
                        y_trues_mlv.extend(label.tolist())
                        y_scores_mlv.extend(sim.tolist())

                # 计算 AP
                y_score = np.asarray(y_scores)
                y_true = np.asarray(y_trues)
                recall_rate = count_tps * 1.0 / count_gts if count_gts > 0 else 0
                ap = 0 if count_tps == 0 else average_precision_score(y_true, y_score) * recall_rate
                self.aps[dst].append(ap)

                inds = np.argsort(y_score)[::-1]
                y_true_sorted = y_true[inds]
                self.accs[dst].append([min(1, sum(y_true_sorted[:k])) for k in self.topks])

                # 计算 MLV AP
                y_score_mlv = np.asarray(y_scores_mlv)
                y_true_mlv = np.asarray(y_trues_mlv)
                recall_rate_mlv = count_tps_mlv * 1.0 / count_gts_mlv if count_gts_mlv > 0 else 0
                ap_mlv = 0 if count_tps_mlv == 0 else average_precision_score(y_true_mlv, y_score_mlv) * recall_rate_mlv
                self.aps_mlv[dst].append(ap_mlv)

                inds_mlv = np.argsort(y_score_mlv)[::-1]
                y_true_mlv_sorted = y_true_mlv[inds_mlv]
                self.accs_mlv[dst].append([min(1, sum(y_true_mlv_sorted[:k])) for k in self.topks])

        # 3. 收集分布式结果 (保持原逻辑)
        mix_eval_results = {}  # 原 super().evaluate() 也是空的，这里直接构造
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

        mix_eval_results["search"] = {}
        for dst in self.det_score_thresh:
            logger.info(
                "Multi-view Search eval_{:.2f} on {} queries. ".format(
                    dst, len(aps[dst])
                )
            )
            mAP = np.mean(aps[dst])
            mix_eval_results["search"].update({"mAP_{:.2f}_mlv".format(dst): mAP})
            acc = np.mean(accs[dst], axis=0)
            for i, v in enumerate(acc.tolist()):
                k = self.topks[i]
                mix_eval_results["search"].update(
                    {"top{:2d}_{:.2f}_mlv".format(k, dst): v}
                )
        self._maybe_save_query_cache()
        return copy.deepcopy(mix_eval_results)
