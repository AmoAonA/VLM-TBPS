"""
Evaluate gallery filtering upper bounds using gallery GT labels or predicted group embeddings.

This script is intended for person search / re-ID style analysis:
- statistics are computed by query pid, not by "same image contains same attribute"
- positives are gallery GT instances with the same pid
- supports both single-attribute and multi-attribute combinations

Example:
python tools/test_attribute_recall.py ^
  --gallery-file outputs/xxx/inference/_gallery_gt_inf.pt ^
  --query-file /data/lzj/dataset/PRW/prw_Final_Complete_cleaned_test.json ^
  --dataset prw
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psd2.data.datasets.prw_tbps import get_resort_id
from psd2.modeling.extend.cerberus_semantic_id import CerberusPrototypeBank


DEFAULT_FILTER_SPECS = [
    "gender",
    "top",
    "pants",
    "gender+top",
    "gender+pants",
    "gender+top+pants",
    "hair",
    "shoes",
]

GROUP_TO_ATTRS = {
    "gender": ["gender"],
    "hair": ["hair_color", "hair_length"],
    "top": ["top_color", "top_style"],
    "pants": ["pants_color", "pants_style"],
    "shoes": ["shoes_color", "shoes_style"],
    "carry": ["bag", "umbrella"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure attribute-based gallery filtering upper bounds with pid-aware recall."
    )
    parser.add_argument("--gallery-file", required=True, help="Path to _gallery_gt_inf.pt")
    parser.add_argument(
        "--query-file",
        required=True,
        help="Path to query/test attribute JSON (e.g. prw_Final_Complete_cleaned_test.json)",
    )
    parser.add_argument(
        "--gallery-attrs-json",
        default="",
        help="Optional gallery/test JSON used when _gallery_gt_inf.pt lacks gt_attrs.",
    )
    parser.add_argument(
        "--gallery-attr-source",
        choices=["gt", "pred_gtbox"],
        default="gt",
        help=(
            "Use GT attributes or predicted attributes decoded from inf_group_feats. "
            "pred_gtbox expects a GT-box cache such as _gallery_gt_infgt.pt."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path used to load Cerberus prototype bank for pred_gtbox mode.",
    )
    parser.add_argument(
        "--schema-path",
        default="",
        help="Schema path used to decode prototype predictions for pred_gtbox mode.",
    )
    parser.add_argument("--semantic-dim", type=int, default=512)
    parser.add_argument(
        "--dataset",
        choices=["auto", "prw", "cuhk"],
        default="auto",
        help="Controls pid remapping. Use prw for PRW_TBPS-style remapped ids.",
    )
    parser.add_argument(
        "--filter-specs",
        nargs="*",
        default=DEFAULT_FILTER_SPECS,
        help=(
            "Filter specs to test. Supports raw attrs like gender/top_color and "
            "group names like top/pants/hair/shoes, combined with '+'."
        ),
    )
    parser.add_argument(
        "--include-unlabeled-gallery",
        action="store_true",
        help="Also count unlabeled gallery GT boxes in the before/after candidate pool.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["label", "embedding"],
        default="label",
        help=(
            "label: decode gallery prediction to discrete attrs and exact-match them; "
            "embedding: compare query prototype embedding with gallery group embeddings."
        ),
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.35,
        help="Default cosine threshold for embedding-mode filtering.",
    )
    parser.add_argument(
        "--embedding-group-thresholds",
        default="",
        help="Optional per-group thresholds like gender=0.55,top=0.35,pants=0.35",
    )
    return parser.parse_args()


def infer_dataset_kind(dataset_arg: str, query_file: str) -> str:
    if dataset_arg != "auto":
        return dataset_arg
    query_name = Path(query_file).name.lower()
    if "prw" in query_name:
        return "prw"
    if "cuhk" in query_name:
        return "cuhk"
    return "cuhk"


def normalize_image_id(pic_path: str) -> str:
    pic_path = str(pic_path)
    return pic_path if pic_path.endswith(".jpg") else pic_path + ".jpg"


def remap_pid(pid: int, dataset_kind: str) -> int:
    pid = int(pid)
    if dataset_kind == "prw":
        return int(get_resort_id(pid))
    return pid


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        return checkpoint
    raise TypeError(f"Unsupported checkpoint structure: {type(checkpoint)}")


def infer_use_carry_group(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        key.endswith("prototype_bank.prototypes_carry") or key.endswith("prototypes_carry")
        for key in state_dict
    )


def load_prototype_bank(
    checkpoint_path: str,
    schema_path: str,
    semantic_dim: int,
) -> CerberusPrototypeBank:
    state_dict = load_checkpoint_state(Path(checkpoint_path))
    bank = CerberusPrototypeBank(
        schema_path=schema_path,
        semantic_dim=semantic_dim,
        use_carry_group=infer_use_carry_group(state_dict),
    )
    missing = []
    for group_name in bank.group_names:
        target = f"prototypes_{group_name}"
        candidates = [
            f"cerberus_branch.prototype_bank.{target}",
            f"module.cerberus_branch.prototype_bank.{target}",
            f"prototype_bank.{target}",
            target,
        ]
        tensor = None
        for key in candidates:
            if key in state_dict:
                tensor = state_dict[key]
                break
        if tensor is None:
            missing.append(target)
            continue
        getattr(bank, target).data.copy_(tensor.float())
    if missing:
        raise KeyError(f"Missing prototype tensors in checkpoint: {missing}")
    return bank


def load_queries(query_file: str, dataset_kind: str) -> List[Dict[str, object]]:
    raw_queries = load_json(query_file)
    queries = []
    for item in raw_queries:
        if "id" not in item:
            continue
        queries.append(
            {
                "pid": remap_pid(item["id"], dataset_kind),
                "orig_pid": int(item["id"]),
                "image_id": normalize_image_id(item.get("pic_path", "")),
                "attributes": item.get("attributes", {}) or {},
            }
        )
    return queries


def build_gallery_attr_lookup(json_path: str, dataset_kind: str) -> Dict[str, Dict[int, Dict[str, int]]]:
    raw_items = load_json(json_path)
    lookup: Dict[str, Dict[int, Dict[str, int]]] = {}
    for item in raw_items:
        if "id" not in item:
            continue
        image_id = normalize_image_id(item.get("pic_path", ""))
        pid = remap_pid(item["id"], dataset_kind)
        lookup.setdefault(image_id, {})[pid] = item.get("attributes", {}) or {}
    return lookup


def maybe_recover_gt_attrs(
    saved: Dict[str, object],
    gallery_attrs_json: str,
    dataset_kind: str,
) -> Dict[str, List[Dict[str, int]]]:
    gt_attrs = saved.get("gt_attrs")
    if isinstance(gt_attrs, dict) and len(gt_attrs) > 0:
        return gt_attrs
    if not gallery_attrs_json:
        return {}

    lookup = build_gallery_attr_lookup(gallery_attrs_json, dataset_kind)
    recovered: Dict[str, List[Dict[str, int]]] = {}
    for img_id, gt_tensor in saved["gts"].items():
        img_lookup = lookup.get(img_id, {})
        per_image_attrs = []
        for row in gt_tensor:
            pid = int(row[4].item())
            per_image_attrs.append(img_lookup.get(pid, {}))
        recovered[img_id] = per_image_attrs
    return recovered


def expand_filter_spec(spec: str) -> List[str]:
    attrs = []
    for token in str(spec).split("+"):
        token = token.strip()
        if not token:
            continue
        if token in GROUP_TO_ATTRS:
            attrs.extend(GROUP_TO_ATTRS[token])
        else:
            attrs.append(token)
    deduped = []
    for attr_name in attrs:
        if attr_name not in deduped:
            deduped.append(attr_name)
    return deduped


def expand_filter_spec_to_groups(spec: str) -> List[str]:
    groups = []
    for token in str(spec).split("+"):
        token = token.strip()
        if not token:
            continue
        if token in GROUP_TO_ATTRS:
            if token not in groups:
                groups.append(token)
            continue
        for group_name, group_attrs in GROUP_TO_ATTRS.items():
            if token in group_attrs and group_name not in groups:
                groups.append(group_name)
                break
    return groups


def parse_group_thresholds(raw_value: str) -> Dict[str, float]:
    if not raw_value:
        return {}
    resolved = {}
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "--embedding-group-thresholds must look like gender=0.55,top=0.35"
            )
        lhs, rhs = token.split("=", 1)
        resolved[lhs.strip()] = float(rhs.strip())
    return resolved


def is_attr_known(attributes: Dict[str, int], attr_name: str) -> bool:
    value = attributes.get(attr_name, None)
    if value is None:
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def attributes_match(query_attrs: Dict[str, int], gallery_attrs: Dict[str, int], attr_names: Iterable[str]) -> bool:
    for attr_name in attr_names:
        if not is_attr_known(query_attrs, attr_name):
            return False
        if not is_attr_known(gallery_attrs, attr_name):
            return False
        if int(query_attrs[attr_name]) != int(gallery_attrs[attr_name]):
            return False
    return True


def build_attr_key(attributes: Dict[str, int], attr_names: Iterable[str]) -> Optional[Tuple[int, ...]]:
    values = []
    for attr_name in attr_names:
        if not is_attr_known(attributes, attr_name):
            return None
        values.append(int(attributes[attr_name]))
    return tuple(values)


def decode_group_index(bank: CerberusPrototypeBank, group_name: str, index: int) -> Dict[str, int]:
    attrs = bank.attribute_groups[group_name]
    dims = bank.group_value_sizes[group_name]
    values = [0] * len(attrs)
    temp = int(index)
    for pos in range(len(attrs) - 1, -1, -1):
        count = dims[pos]
        values[pos] = temp % count
        temp //= count
    return {attr_name: value for attr_name, value in zip(attrs, values)}


def decode_group_predictions(
    bank: CerberusPrototypeBank,
    group_feats: torch.Tensor,
    group_valid: Optional[torch.Tensor] = None,
) -> Dict[str, int]:
    decoded = {}
    if group_feats is None:
        return decoded
    for group_offset, group_name in enumerate(bank.group_names):
        if group_offset >= group_feats.shape[0]:
            continue
        if group_valid is not None and group_offset < group_valid.numel():
            if float(group_valid[group_offset].item()) <= 0:
                continue
        prototypes = F.normalize(bank.get_group_prototypes(group_name).float(), dim=-1)
        pred_embedding = F.normalize(group_feats[group_offset].float(), dim=-1)
        pred_group_idx = int(torch.argmax(pred_embedding @ prototypes.t()).item())
        decoded.update(decode_group_index(bank, group_name, pred_group_idx))
    return decoded


def build_query_group_embeddings(
    bank: CerberusPrototypeBank,
    query_attrs: Dict[str, int],
    group_names: List[str],
) -> Optional[Dict[str, torch.Tensor]]:
    embeddings = {}
    for group_name in group_names:
        group_index = bank.get_group_index(group_name, query_attrs)
        if group_index < 0:
            return None
        prototypes = bank.get_group_prototypes(group_name)
        embeddings[group_name] = F.normalize(prototypes[group_index].float(), dim=-1)
    return embeddings


def build_gallery_instances(
    saved: Dict[str, object],
    gt_attrs: Dict[str, List[Dict[str, int]]],
    include_unlabeled_gallery: bool,
) -> List[Dict[str, object]]:
    instances = []
    for img_id, gt_tensor in saved["gts"].items():
        attrs_list = gt_attrs.get(img_id, [{} for _ in range(len(gt_tensor))])
        for idx, row in enumerate(gt_tensor):
            pid = int(row[4].item())
            if not include_unlabeled_gallery and pid < 0:
                continue
            instances.append(
                {
                    "image_id": img_id,
                    "pid": pid,
                    "attrs": attrs_list[idx] if idx < len(attrs_list) else {},
                    "group_feats": None,
                    "group_valid": None,
                }
            )
    return instances


def build_gallery_instances_from_pred_gtbox(
    saved: Dict[str, object],
    bank: CerberusPrototypeBank,
    include_unlabeled_gallery: bool,
) -> List[Dict[str, object]]:
    instances = []
    inf_group_feats = saved.get("inf_group_feats", {})
    inf_group_valids = saved.get("inf_group_valids", {})
    for img_id, gt_tensor in saved["gts"].items():
        group_feats = inf_group_feats.get(img_id)
        group_valids = inf_group_valids.get(img_id)
        for idx, row in enumerate(gt_tensor):
            pid = int(row[4].item())
            if not include_unlabeled_gallery and pid < 0:
                continue
            attrs = {}
            per_group_feats = None
            per_group_valid = None
            if group_feats is not None and idx < len(group_feats):
                valid = group_valids[idx] if group_valids is not None and idx < len(group_valids) else None
                attrs = decode_group_predictions(bank, group_feats[idx], valid)
                per_group_feats = group_feats[idx].float()
                per_group_valid = valid.float() if valid is not None else None
            instances.append(
                {
                    "image_id": img_id,
                    "pid": pid,
                    "attrs": attrs,
                    "group_feats": per_group_feats,
                    "group_valid": per_group_valid,
                }
            )
    return instances


def build_global_indexes(gallery_instances: List[Dict[str, object]]):
    total_count = len(gallery_instances)
    image_counts: Dict[str, int] = {}
    pid_counts: Dict[int, int] = {}
    pid_image_counts: Dict[Tuple[int, str], int] = {}

    for inst in gallery_instances:
        image_id = str(inst["image_id"])
        pid = int(inst["pid"])
        image_counts[image_id] = image_counts.get(image_id, 0) + 1
        pid_counts[pid] = pid_counts.get(pid, 0) + 1
        pid_image_counts[(pid, image_id)] = pid_image_counts.get((pid, image_id), 0) + 1

    return {
        "total_count": total_count,
        "image_counts": image_counts,
        "pid_counts": pid_counts,
        "pid_image_counts": pid_image_counts,
    }


def build_filter_indexes(
    gallery_instances: List[Dict[str, object]], attr_names: List[str]
):
    known_count = 0
    known_image_counts: Dict[str, int] = {}
    key_counts: Dict[Tuple[int, ...], int] = {}
    key_image_counts: Dict[Tuple[Tuple[int, ...], str], int] = {}
    key_pid_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
    key_pid_image_counts: Dict[Tuple[Tuple[int, ...], int, str], int] = {}

    for inst in gallery_instances:
        image_id = str(inst["image_id"])
        pid = int(inst["pid"])
        key = build_attr_key(inst["attrs"], attr_names)
        if key is None:
            continue
        known_count += 1
        known_image_counts[image_id] = known_image_counts.get(image_id, 0) + 1
        key_counts[key] = key_counts.get(key, 0) + 1
        key_image_counts[(key, image_id)] = key_image_counts.get((key, image_id), 0) + 1
        key_pid_counts[(key, pid)] = key_pid_counts.get((key, pid), 0) + 1
        key_pid_image_counts[(key, pid, image_id)] = key_pid_image_counts.get(
            (key, pid, image_id), 0
        ) + 1

    return {
        "known_count": known_count,
        "known_image_counts": known_image_counts,
        "key_counts": key_counts,
        "key_image_counts": key_image_counts,
        "key_pid_counts": key_pid_counts,
        "key_pid_image_counts": key_pid_image_counts,
    }


def build_embedding_indexes(
    gallery_instances: List[Dict[str, object]],
    group_names: List[str],
    query_group_embeddings_by_key: Dict[Tuple[int, ...], Dict[str, torch.Tensor]],
    group_thresholds: Dict[str, float],
):
    key_counts: Dict[Tuple[int, ...], int] = {}
    key_image_counts: Dict[Tuple[Tuple[int, ...], str], int] = {}
    key_pid_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
    key_pid_image_counts: Dict[Tuple[Tuple[int, ...], int, str], int] = {}
    known_count = 0
    known_image_counts: Dict[str, int] = {}

    for inst in gallery_instances:
        image_id = str(inst["image_id"])
        pid = int(inst["pid"])
        group_feats = inst.get("group_feats")
        group_valid = inst.get("group_valid")
        if group_feats is None:
            continue

        candidate_keys = []
        for query_key, query_group_embeddings in query_group_embeddings_by_key.items():
            passed = True
            for group_name in group_names:
                group_offset = query_group_embeddings["__group_offsets__"][group_name]
                if group_offset >= group_feats.shape[0]:
                    passed = False
                    break
                if group_valid is not None and group_offset < group_valid.numel():
                    if float(group_valid[group_offset].item()) <= 0:
                        passed = False
                        break
                gallery_embed = F.normalize(group_feats[group_offset].float(), dim=-1)
                query_embed = query_group_embeddings[group_name]
                similarity = float(torch.dot(gallery_embed, query_embed).item())
                if similarity < float(group_thresholds[group_name]):
                    passed = False
                    break
            if passed:
                candidate_keys.append(query_key)

        has_any_valid = False
        for group_name in group_names:
            group_offset = query_group_embeddings_by_key[next(iter(query_group_embeddings_by_key))]["__group_offsets__"][group_name]
            if group_offset >= group_feats.shape[0]:
                has_any_valid = False
                break
            if group_valid is not None and group_offset < group_valid.numel():
                if float(group_valid[group_offset].item()) <= 0:
                    has_any_valid = False
                    break
            has_any_valid = True
        if has_any_valid:
            known_count += 1
            known_image_counts[image_id] = known_image_counts.get(image_id, 0) + 1

        for query_key in candidate_keys:
            key_counts[query_key] = key_counts.get(query_key, 0) + 1
            key_image_counts[(query_key, image_id)] = key_image_counts.get((query_key, image_id), 0) + 1
            key_pid_counts[(query_key, pid)] = key_pid_counts.get((query_key, pid), 0) + 1
            key_pid_image_counts[(query_key, pid, image_id)] = key_pid_image_counts.get(
                (query_key, pid, image_id), 0
            ) + 1

    return {
        "known_count": known_count,
        "known_image_counts": known_image_counts,
        "key_counts": key_counts,
        "key_image_counts": key_image_counts,
        "key_pid_counts": key_pid_counts,
        "key_pid_image_counts": key_pid_image_counts,
    }


def evaluate_filter(
    gallery_instances: List[Dict[str, object]],
    queries: List[Dict[str, object]],
    filter_spec: str,
    global_indexes: Dict[str, object],
    match_mode: str = "label",
    prototype_bank: Optional[CerberusPrototypeBank] = None,
    group_thresholds: Optional[Dict[str, float]] = None,
    embedding_threshold: float = 0.35,
) -> Optional[Dict[str, float]]:
    attr_names = expand_filter_spec(filter_spec)
    if not attr_names:
        return None

    if match_mode == "embedding":
        if prototype_bank is None:
            raise ValueError("prototype_bank is required for embedding-mode filtering.")
        group_names = expand_filter_spec_to_groups(filter_spec)
        if not group_names:
            return None
        query_group_embeddings_by_key = {}
        for query in queries:
            query_key = build_attr_key(query["attributes"], attr_names)
            if query_key is None or query_key in query_group_embeddings_by_key:
                continue
            group_embeddings = build_query_group_embeddings(
                prototype_bank, query["attributes"], group_names
            )
            if group_embeddings is None:
                continue
            group_embeddings["__group_offsets__"] = {
                group_name: prototype_bank.group_names.index(group_name)
                for group_name in group_names
            }
            query_group_embeddings_by_key[query_key] = group_embeddings
        if not query_group_embeddings_by_key:
            return None
        resolved_thresholds = {}
        for group_name in group_names:
            if group_thresholds and group_name in group_thresholds:
                resolved_thresholds[group_name] = float(group_thresholds[group_name])
            else:
                resolved_thresholds[group_name] = float(embedding_threshold)
        filter_indexes = build_embedding_indexes(
            gallery_instances,
            group_names,
            query_group_embeddings_by_key,
            resolved_thresholds,
        )
    else:
        filter_indexes = build_filter_indexes(gallery_instances, attr_names)
    total_gallery = int(global_indexes["total_count"])
    image_counts = global_indexes["image_counts"]
    pid_counts = global_indexes["pid_counts"]
    pid_image_counts = global_indexes["pid_image_counts"]
    key_counts = filter_indexes["key_counts"]
    known_count = int(filter_indexes["known_count"])
    known_image_counts = filter_indexes["known_image_counts"]
    key_image_counts = filter_indexes["key_image_counts"]
    key_pid_counts = filter_indexes["key_pid_counts"]
    key_pid_image_counts = filter_indexes["key_pid_image_counts"]

    total_queries = 0
    survival_count = 0
    total_before = 0.0
    total_after = 0.0
    total_retained_ratio = 0.0
    total_known_before = 0.0
    total_known_retained_ratio = 0.0
    total_positive_ratio = 0.0
    total_positive_before = 0.0
    total_positive_after = 0.0

    for query in queries:
        query_attrs = query["attributes"]
        query_key = build_attr_key(query_attrs, attr_names)
        if query_key is None:
            continue

        query_pid = int(query["pid"])
        query_image_id = str(query["image_id"])
        candidates_before = total_gallery - int(image_counts.get(query_image_id, 0))
        positives_before = int(pid_counts.get(query_pid, 0)) - int(
            pid_image_counts.get((query_pid, query_image_id), 0)
        )
        if candidates_before <= 0 or positives_before <= 0:
            continue

        retained = int(key_counts.get(query_key, 0)) - int(
            key_image_counts.get((query_key, query_image_id), 0)
        )
        known_before = known_count - int(known_image_counts.get(query_image_id, 0))
        positives_after = int(key_pid_counts.get((query_key, query_pid), 0)) - int(
            key_pid_image_counts.get((query_key, query_pid, query_image_id), 0)
        )
        retained = max(retained, 0)
        known_before = max(known_before, 0)
        positives_after = max(positives_after, 0)

        total_queries += 1
        total_before += float(candidates_before)
        total_after += float(retained)
        total_retained_ratio += float(retained) / float(candidates_before)
        total_known_before += float(known_before)
        if known_before > 0:
            total_known_retained_ratio += float(retained) / float(known_before)
        total_positive_before += float(positives_before)
        total_positive_after += float(positives_after)
        total_positive_ratio += float(positives_after) / float(positives_before)
        if positives_after > 0:
            survival_count += 1

    if total_queries == 0:
        return None

    avg_before = total_before / total_queries
    avg_after = total_after / total_queries
    retained_ratio = total_retained_ratio / total_queries
    avg_known_before = total_known_before / total_queries
    known_retained_ratio = total_known_retained_ratio / total_queries
    positive_keep_ratio = total_positive_ratio / total_queries
    return {
        "filter_spec": filter_spec,
        "num_queries": total_queries,
        "match_mode": match_mode,
        "avg_candidates_before": avg_before,
        "avg_candidates_after": avg_after,
        "avg_retained_ratio": retained_ratio,
        "avg_known_candidates_before": avg_known_before,
        "avg_known_retained_ratio": known_retained_ratio,
        "compression_ratio": retained_ratio,
        "positive_survival": float(survival_count) / float(total_queries),
        "avg_positive_keep_ratio": positive_keep_ratio,
        "avg_positive_before": total_positive_before / total_queries,
        "avg_positive_after": total_positive_after / total_queries,
        "num_attrs": len(attr_names),
    }


def print_results(results: List[Dict[str, float]], dataset_kind: str, gallery_instances: List[Dict[str, object]], queries: List[Dict[str, object]]):
    print(f"Dataset: {dataset_kind}")
    print(f"Gallery instances: {len(gallery_instances)}")
    print(f"Queries loaded: {len(queries)}")
    print("")
    print(
        f"{'Filter':<22} {'#Q':>6} {'Before':>10} {'After':>10} "
        f"{'Retain':>9} {'KnownB':>10} {'KnownR':>9} {'PosSurv':>9} {'PosKeep':>9}"
    )
    print("-" * 108)
    for row in results:
        print(
            f"{row['filter_spec']:<22} "
            f"{int(row['num_queries']):>6d} "
            f"{row['avg_candidates_before']:>10.1f} "
            f"{row['avg_candidates_after']:>10.1f} "
            f"{row['avg_retained_ratio'] * 100:>8.2f}% "
            f"{row['avg_known_candidates_before']:>10.1f} "
            f"{row['avg_known_retained_ratio'] * 100:>8.2f}% "
            f"{row['positive_survival'] * 100:>8.2f}% "
            f"{row['avg_positive_keep_ratio'] * 100:>8.2f}%"
        )


def main():
    args = parse_args()
    dataset_kind = infer_dataset_kind(args.dataset, args.query_file)
    group_thresholds = parse_group_thresholds(args.embedding_group_thresholds)
    prototype_bank = None

    print("Loading gallery cache...")
    saved = torch.load(args.gallery_file, map_location="cpu")
    if args.gallery_attr_source == "gt":
        gt_attrs = maybe_recover_gt_attrs(saved, args.gallery_attrs_json, dataset_kind)
        gallery_instances = build_gallery_instances(
            saved, gt_attrs, include_unlabeled_gallery=args.include_unlabeled_gallery
        )
    else:
        if not args.checkpoint or not args.schema_path:
            raise ValueError(
                "--checkpoint and --schema-path are required when --gallery-attr-source=pred_gtbox."
            )
        prototype_bank = load_prototype_bank(args.checkpoint, args.schema_path, args.semantic_dim)
        gallery_instances = build_gallery_instances_from_pred_gtbox(
            saved,
            prototype_bank,
            include_unlabeled_gallery=args.include_unlabeled_gallery,
        )

    print("Loading queries...")
    queries = load_queries(args.query_file, dataset_kind)
    global_indexes = build_global_indexes(gallery_instances)

    results = []
    for filter_spec in args.filter_specs:
        result = evaluate_filter(
            gallery_instances,
            queries,
            filter_spec,
            global_indexes,
            match_mode=args.match_mode,
            prototype_bank=prototype_bank,
            group_thresholds=group_thresholds,
            embedding_threshold=args.embedding_threshold,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No valid queries/filter specs found. Check dataset/json paths and filter specs.")

    print(f"Gallery attribute source: {args.gallery_attr_source}")
    print(f"Match mode: {args.match_mode}")
    if args.match_mode == "embedding":
        print(f"Embedding default threshold: {args.embedding_threshold:.3f}")
        if group_thresholds:
            print(f"Embedding group thresholds: {group_thresholds}")
    print_results(results, dataset_kind, gallery_instances, queries)


if __name__ == "__main__":
    main()
