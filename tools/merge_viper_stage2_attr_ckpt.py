#!/usr/bin/env python
import argparse
import copy
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge a standard ViPer checkpoint with a stage2 attribute checkpoint. "
            "The base/global weights come from --base-ckpt, and selected prefixes "
            "from --attr-ckpt override or extend the merged model state."
        )
    )
    parser.add_argument("--base-ckpt", required=True, help="Official/base ViPer checkpoint.")
    parser.add_argument("--attr-ckpt", required=True, help="Stage2 attribute checkpoint.")
    parser.add_argument("--output", required=True, help="Path to save merged checkpoint.")
    parser.add_argument(
        "--take-prefix",
        action="append",
        default=["cerberus_branch."],
        help=(
            "Parameter prefix to copy from attr checkpoint. "
            "Can be given multiple times. Default: cerberus_branch."
        ),
    )
    parser.add_argument(
        "--drop-training-state",
        action="store_true",
        help="Drop optimizer/scheduler metadata and only save model + lightweight merge metadata.",
    )
    return parser.parse_args()


def get_model_state(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def set_model_state(container, model_state):
    if isinstance(container, dict) and "model" in container:
        container["model"] = model_state
        return container
    return model_state


def should_take(key, prefixes):
    return any(key.startswith(prefix) for prefix in prefixes)


def main():
    args = parse_args()

    base_path = Path(args.base_ckpt)
    attr_path = Path(args.attr_ckpt)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_ckpt = torch.load(base_path, map_location="cpu")
    attr_ckpt = torch.load(attr_path, map_location="cpu")

    base_state = get_model_state(base_ckpt)
    attr_state = get_model_state(attr_ckpt)

    merged_state = copy.deepcopy(base_state)

    copied_keys = []
    added_keys = []
    skipped_keys = []
    for key, value in attr_state.items():
        if not should_take(key, args.take_prefix):
            skipped_keys.append(key)
            continue
        if key in merged_state:
            copied_keys.append(key)
        else:
            added_keys.append(key)
        merged_state[key] = value

    if args.drop_training_state:
        merged_ckpt = {
            "model": merged_state,
            "__author__": "openai_codex_merge",
            "matching_heuristics": False,
            "merge_info": {
                "base_ckpt": str(base_path),
                "attr_ckpt": str(attr_path),
                "take_prefixes": list(args.take_prefix),
                "copied_keys": len(copied_keys),
                "added_keys": len(added_keys),
            },
        }
    else:
        merged_ckpt = copy.deepcopy(base_ckpt)
        merged_ckpt = set_model_state(merged_ckpt, merged_state)
        if isinstance(merged_ckpt, dict):
            merged_ckpt["merge_info"] = {
                "base_ckpt": str(base_path),
                "attr_ckpt": str(attr_path),
                "take_prefixes": list(args.take_prefix),
                "copied_keys": len(copied_keys),
                "added_keys": len(added_keys),
            }

    torch.save(merged_ckpt, out_path)

    print("merged checkpoint saved:", out_path)
    print("base ckpt:", base_path)
    print("attr ckpt:", attr_path)
    print("take prefixes:", list(args.take_prefix))
    print("copied keys:", len(copied_keys))
    print("added keys:", len(added_keys))
    print("skipped attr keys:", len(skipped_keys))
    if copied_keys:
        print("sample copied keys:")
        for key in copied_keys[:10]:
            print("  ", key)
    if added_keys:
        print("sample added keys:")
        for key in added_keys[:10]:
            print("  ", key)


if __name__ == "__main__":
    main()
