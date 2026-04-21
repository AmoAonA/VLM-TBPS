#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in psd2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use psd2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys
import numpy as np
import random

sys.path.append("./")
import logging
import os
from collections import OrderedDict
import torch
import torch.multiprocessing
# 解决 received 0 items of ancdata 报错
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import psd2.utils.comm as comm
from psd2.checkpoint import DetectionCheckpointer
from psd2.config import get_cfg
from psd2.data import MetadataCatalog
from psd2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from psd2.evaluation import (
    InfDetEvaluator,
    PrwQueryEvaluator,
    CuhkQueryEvaluator,
    DatasetEvaluators,
    verify_results,
    PrwQueryEvaluatorP,
    CuhkQueryEvaluatorP,
)
import re
from psd2.modeling import GeneralizedRCNNWithTTA

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    single_gpu = comm.get_world_size() == 1
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    query_eval_kwargs = dict(
        search_fusion_mode=cfg.TEST.SEARCH_FUSION_MODE,
        search_shortlist_size=cfg.TEST.SEARCH_SHORTLIST_SIZE,
        search_rrf_k=cfg.TEST.SEARCH_RRF_K,
        search_max_parts=cfg.TEST.SEARCH_MAX_PARTS,
        search_global_vote_weight=cfg.TEST.SEARCH_GLOBAL_VOTE_WEIGHT,
        search_part_vote_weights=cfg.TEST.SEARCH_PART_VOTE_WEIGHTS,
        search_avg_global_weight=cfg.TEST.SEARCH_AVG_GLOBAL_WEIGHT,
        search_avg_part_weights=cfg.TEST.SEARCH_AVG_PART_WEIGHTS,
        search_bp_pos_threshold=cfg.TEST.SEARCH_BP_POS_THRESHOLD,
        search_bp_neg_threshold=cfg.TEST.SEARCH_BP_NEG_THRESHOLD,
        search_bp_pos_weight=cfg.TEST.SEARCH_BP_POS_WEIGHT,
        search_bp_neg_weight=cfg.TEST.SEARCH_BP_NEG_WEIGHT,
        search_kde_bandwidth=cfg.TEST.SEARCH_KDE_BANDWIDTH,
        search_kde_density_weight=cfg.TEST.SEARCH_KDE_DENSITY_WEIGHT,
        search_kde_top_k=cfg.TEST.SEARCH_KDE_TOP_K,
        search_gender_label_shortlist_enabled=cfg.TEST.SEARCH_GENDER_LABEL_SHORTLIST_ENABLED,
        search_gender_label_shortlist_size=cfg.TEST.SEARCH_GENDER_LABEL_SHORTLIST_SIZE,
        search_top_pants_soft_rerank_enabled=cfg.TEST.SEARCH_TOP_PANTS_SOFT_RERANK_ENABLED,
        search_top_pants_soft_rerank_size=cfg.TEST.SEARCH_TOP_PANTS_SOFT_RERANK_SIZE,
        search_top_soft_weight=cfg.TEST.SEARCH_TOP_SOFT_WEIGHT,
        search_pants_soft_weight=cfg.TEST.SEARCH_PANTS_SOFT_WEIGHT,
        search_top_soft_center=cfg.TEST.SEARCH_TOP_SOFT_CENTER,
        search_pants_soft_center=cfg.TEST.SEARCH_PANTS_SOFT_CENTER,
        coarse_topk=cfg.TEST.COARSE_TOPK,
        reuse_eval_cache=cfg.TEST.REUSE_EVAL_CACHE,
        save_query_cache=cfg.TEST.SAVE_QUERY_CACHE,
    )
    if evaluator_type == "det":
        evaluator_list.append(
            InfDetEvaluator(
                dataset_name,
                distributed=not single_gpu,
                output_dir=output_folder,
                s_threds=cfg.TEST.DETECTION_SCORE_TS,
                topk=cfg.TEST.DETECTIONS_PER_IMAGE,
                reuse_eval_cache=cfg.TEST.REUSE_EVAL_CACHE,
            )
        )
    elif evaluator_type == "query":
        if "CUHK-SYSU" in dataset_name:
            evaluator_list.append(
                CuhkQueryEvaluatorP(
                    dataset_name,
                    distributed=False,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                    **query_eval_kwargs,
                )
                if single_gpu
                else CuhkQueryEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                    **query_eval_kwargs,
                )
            )
        elif "PRW" in dataset_name:
            evaluator_list.append(
                PrwQueryEvaluatorP(
                    dataset_name,
                    distributed=False,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                    **query_eval_kwargs,
                )
                if single_gpu
                else PrwQueryEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    s_threds=cfg.TEST.DETECTION_SCORE_TS,
                    **query_eval_kwargs,
                )
            )
        else:
            raise ValueError("Unknown dataset {}".format(dataset_name))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        # NOTE not tested
        logger = logging.getLogger("psd2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        from psd2.data.catalog import MapperCatalog
        from psd2.data.build import build_detection_train_loader

        mapper = MapperCatalog.get(cfg.DATASETS.TRAIN[0])(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        from psd2.data.catalog import MapperCatalog
        from psd2.data.build import get_detection_dataset_dicts, trivial_batch_collator
        from psd2.data.common import DatasetFromList, MapDataset
        from psd2.data.samplers import InferenceSampler
        import torch.utils.data as torchdata

        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)]
                for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        mapper = MapperCatalog.get(dataset_name)(cfg, is_train=False)
        # batched test
        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False)
        dataset = MapDataset(dataset, mapper)
        sampler = InferenceSampler(len(dataset))

        batch_sampler = torchdata.sampler.BatchSampler(
            sampler, cfg.TEST.IMS_PER_PROC, drop_last=False
        )
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader

    @classmethod
    def build_optimizer(cls, cfg, model):
        from psd2.solver.build import maybe_add_gradient_clipping

        logger = logging.getLogger("psd2.trainer")
        frozen_params = []
        learn_param_keys = []
        param_groups = [{"params": [], "lr": cfg.SOLVER.BASE_LR}] + [
            {"params": [], "lr": cfg.SOLVER.BASE_LR * lf}
            for lf in cfg.SOLVER.LR_FACTORS
        ]
        bias_param_groups = [{"params": [], "lr": cfg.SOLVER.BASE_LR,"weight_decay":cfg.SOLVER.WEIGHT_DECAY_BIAS}] + [
            {"params": [], "lr": cfg.SOLVER.BASE_LR * lf,"weight_decay":cfg.SOLVER.WEIGHT_DECAY_BIAS}
            for lf in cfg.SOLVER.LR_FACTORS
        ]
        freeze_regex = [re.compile(reg) for reg in cfg.SOLVER.FREEZE_PARAM_REGEX]
        lr_group_regex = [re.compile(reg) for reg in cfg.SOLVER.LR_GROUP_REGEX]

        module_params={}

        for mn,md in model.named_children():
            if mn == "clip_model":
                for mmn,mmd in md.named_children():
                    if mmn=="visual":
                        for mmmn,mmmd in mmd.named_children():
                            n_params=0
                            for param in mmmd.parameters():
                                n_params+=param.numel()
                            module_params["clip_model.visual."+mmmn]=n_params
                    else:
                        n_params=0
                        for param in mmd.parameters():
                            n_params+=param.numel()
                        module_params["clip_model."+mmn]=n_params
            else:
                n_params=0
                for param in md.parameters():
                    n_params+=param.numel()
                module_params[mn]=n_params

        def _find_match(pkey, prob_regs):
            match_idx = -1
            for mi, mreg in enumerate(prob_regs):
                if re.match(mreg, pkey):
                    assert match_idx == -1, "Ambiguous matching of {}".format(pkey)
                    match_idx = mi
            return match_idx

        for key, value in model.named_parameters(recurse=True):
            match_freeze = _find_match(key, freeze_regex)
            if match_freeze > -1:
                value.requires_grad = False
            if not value.requires_grad:
                frozen_params.append(key)
                continue
            match_learn = _find_match(key, lr_group_regex)
            if match_learn > -1:
                if key.endswith("bias" ):
                    bias_param_groups[match_learn+1]["params"].append(value)
                else:
                    param_groups[match_learn+1]["params"].append(value)
            else:
                if key.endswith("bias" ):
                    bias_param_groups[0]["params"].append(value)
                else:
                    param_groups[0]["params"].append(value)
            learn_param_keys.append(key)
        bias_param_groups = [groups for groups in bias_param_groups if len(groups["params"])>0]
        param_groups=[groups for groups in param_groups if len(groups["params"])>0]+bias_param_groups
        logger.info("Frozen parameters:\n{}".format("\n".join(frozen_params)))
        logger.info("Training parameters:\n{}".format("\n".join(learn_param_keys)))
        logger.info("Total parameters:{}'.format(module_params)'")
        optim = cfg.SOLVER.OPTIM
        if optim == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optim == "Adam":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optim == "AdamW":
            return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError("Unsupported optimizer {}".format(optim))

    def _freeze_backbone(self):
        """冻结backbone和roi_heads的参数"""
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'roi_heads' in name:
                param.requires_grad = False
        logger = logging.getLogger("psd2.trainer")
        logger.info("Frozen backbone and roi_heads parameters")

    def _unfreeze_backbone(self):
        """解冻backbone和roi_heads的参数"""
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'roi_heads' in name:
                param.requires_grad = True
        logger = logging.getLogger("psd2.trainer")
        logger.info("Unfrozen backbone and roi_heads parameters")

    def _setup_freeze_hooks(self):
        """设置冻结/解冻钩子"""
        if hasattr(self.cfg, 'SOLVER') and self.cfg.SOLVER.get('FREEZE_EPOCHS', 0) > 0:
            freeze_epochs = self.cfg.SOLVER.FREEZE_EPOCHS
            self.register_hook(FreezeBackboneHook(self.model, freeze_epochs))

    def resume_or_load(self, resume=True):
        """
        如果 `resume==True` 并且 `cfg.OUTPUT_DIR` 包含最后一个检查点（由
        一个 `last_checkpoint` 文件定义），则从该文件恢复。恢复意味着加载所有
        可用状态（例如优化器和调度器）并从检查点更新迭代计数器
        从 `cfg.MODEL.WEIGHTS` 文件（但不会加载其他状态）加载模型
        权重并从迭代 0 开始。
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # 检查点存储刚刚完成的训练迭代，因此我们从下一个迭代开始
            self.start_iter = self.iter + 1
        
        # 设置冻结/解冻钩子
        self._setup_freeze_hooks()


class FreezeBackboneHook:
    """冻结backbone参数的钩子"""
    
    def __init__(self, model, freeze_epochs):
        self.model = model
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        logger = logging.getLogger("psd2.trainer")
        logger.info(f"FreezeBackboneHook initialized: freeze_epochs={freeze_epochs}")
    
    def before_step(self):
        """在每个step前调用"""
        # 检查是否需要解冻
        if self.current_epoch == self.freeze_epochs:
            # 解冻backbone和roi_heads
            for name, param in self.model.named_parameters():
                if 'backbone' in name or 'roi_heads' in name:
                    param.requires_grad = True
            logger = logging.getLogger("psd2.trainer")
            logger.info(f"Epoch {self.current_epoch}: Unfrozen backbone and roi_heads")
        self.current_epoch += 1


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    set_seed(cfg.SEED)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg, model_find_unused_parameters=not args.n_find_unparams)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
