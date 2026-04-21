import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_WEIGHT_PRESETS = [
    "0.1,1.0,1.2,1.2,1.0",
    "0.1,1.0,1.3,1.3,1.0",
    "0.1,0.9,1.4,1.4,0.9",
    "0.1,1.1,1.3,1.3,1.1",
]


def build_overnight_querydep_presets() -> List[str]:
    presets = []
    for gender in (0.05, 0.10, 0.20):
        for hair in (1.0, 1.1):
            for top_pants in (1.25, 1.30, 1.35, 1.40):
                for shoes in (1.0, 1.1):
                    presets.append(
                        f"{gender:.2f},{hair:.1f},{top_pants:.2f},{top_pants:.2f},{shoes:.1f}"
                    )
    return presets


def build_overnight_querydep_wide_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    seen = set()
    weight_presets = [
        "0.05,1.0,1.25,1.25,1.0",
        "0.05,1.1,1.30,1.30,1.0",
        "0.05,1.1,1.35,1.35,1.1",
        "0.10,1.0,1.20,1.20,1.0",
        "0.10,1.0,1.30,1.30,1.0",
        "0.10,1.1,1.30,1.30,1.1",
        "0.10,1.1,1.35,1.35,1.1",
        "0.10,1.2,1.30,1.30,1.0",
        "0.15,1.0,1.30,1.30,1.0",
        "0.15,1.1,1.30,1.30,1.1",
        "0.20,1.0,1.30,1.30,1.0",
        "0.20,1.1,1.25,1.25,1.1",
    ]
    for weights in weight_presets:
        for unknown in (0.25, 0.35, 0.45):
            for text_boost in (0.10, 0.20, 0.30):
                for text_max in (1.25, 1.35):
                    key = (weights, unknown, text_boost, text_max)
                    if key in seen:
                        continue
                    seen.add(key)
                    runs.append(
                        {
                            "label": (
                                f"uw_{unknown:.2f}_tb_{text_boost:.2f}_tm_{text_max:.2f}_"
                                f"pw_{sanitize_weights(weights)}"
                            ),
                            "weights": weights,
                            "extra_opts": [
                                f"PERSON_SEARCH.REID.CERBERUS_QUERY_UNKNOWN_WEIGHT {unknown:.2f}",
                                "PERSON_SEARCH.REID.CERBERUS_QUERY_GROUP_WEIGHT_MODE avg",
                                "PERSON_SEARCH.REID.CERBERUS_QUERY_TEXT_BOOST_ENABLED True",
                                f"PERSON_SEARCH.REID.CERBERUS_QUERY_TEXT_BOOST {text_boost:.2f}",
                                f"PERSON_SEARCH.REID.CERBERUS_QUERY_TEXT_MAX_WEIGHT {text_max:.2f}",
                            ],
                        }
                    )
    return runs


def build_overnight_querydep_focus_runs() -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    seen = set()
    weight_presets = [
        "0.05,1.1,1.30,1.30,1.1",
        "0.10,1.0,1.30,1.30,1.0",
        "0.10,1.1,1.25,1.25,1.1",
        "0.10,1.1,1.30,1.30,1.1",
        "0.10,1.1,1.35,1.35,1.1",
        "0.10,1.15,1.30,1.30,1.05",
        "0.15,1.1,1.30,1.30,1.1",
    ]
    for weights in weight_presets:
        for unknown in (0.30, 0.35, 0.40):
            for text_boost in (0.15, 0.20, 0.25):
                for text_max in (1.30, 1.35):
                    key = (weights, unknown, text_boost, text_max)
                    if key in seen:
                        continue
                    seen.add(key)
                    runs.append(
                        {
                            "label": (
                                f"focus_uw_{unknown:.2f}_tb_{text_boost:.2f}_tm_{text_max:.2f}_"
                                f"pw_{sanitize_weights(weights)}"
                            ),
                            "weights": weights,
                            "extra_opts": [
                                f"PERSON_SEARCH.REID.CERBERUS_QUERY_UNKNOWN_WEIGHT {unknown:.2f}",
                                "PERSON_SEARCH.REID.CERBERUS_QUERY_GROUP_WEIGHT_MODE avg",
                                "PERSON_SEARCH.REID.CERBERUS_QUERY_TEXT_BOOST_ENABLED True",
                                f"PERSON_SEARCH.REID.CERBERUS_QUERY_TEXT_BOOST {text_boost:.2f}",
                                f"PERSON_SEARCH.REID.CERBERUS_QUERY_TEXT_MAX_WEIGHT {text_max:.2f}",
                            ],
                        }
                    )
    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eval-only ablations over multiple SEARCH_AVG_PART_WEIGHTS presets."
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch tools/train_ps_net.py",
    )
    parser.add_argument(
        "--config-file",
        default="configs/person_search/viper/viper_semantic_cerberus.yaml",
    )
    parser.add_argument("--weights", nargs="*", default=None)
    parser.add_argument(
        "--preset",
        choices=[
            "default",
            "overnight_querydep",
            "overnight_querydep_focus",
            "overnight_querydep_wide",
        ],
        default="default",
        help="Named weight preset collection.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--schema-path")
    parser.add_argument("--train-attrs-path")
    parser.add_argument("--test-attrs-path")
    parser.add_argument("--fusion-mode", default="avg")
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--extra-opt", action="append", default=[])
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a run if OUTPUT_DIR/log.txt already contains final copypaste metrics.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the preset list into N shards for multi-GPU or multi-machine sweeps.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run only the shard with this 0-based index.",
    )
    return parser.parse_args()


def sanitize_weights(weights: str) -> str:
    return weights.replace(" ", "").replace(".", "p").replace(",", "_")


def resolve_weights(args) -> List[str]:
    if args.weights:
        return args.weights
    if args.preset == "overnight_querydep":
        return build_overnight_querydep_presets()
    return DEFAULT_WEIGHT_PRESETS


def resolve_runs(args) -> List[Dict[str, object]]:
    if args.preset == "overnight_querydep_focus":
        return build_overnight_querydep_focus_runs()
    if args.preset == "overnight_querydep_wide":
        return build_overnight_querydep_wide_runs()
    return [{"label": None, "weights": w, "extra_opts": []} for w in resolve_weights(args)]


def is_completed(output_dir: Path) -> bool:
    log_path = output_dir / "log.txt"
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "copypaste: mAP_0.50" in text


def merge_extra_opts(base_opts: List[str], run_opts: Optional[List[str]] = None) -> List[str]:
    opts: List[str] = []
    if base_opts:
        opts.extend(base_opts)
    if run_opts:
        opts.extend(run_opts)
    return opts


def resolve_checkpoint_path(checkpoint: str) -> str:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        model_final = checkpoint_path / "model_final.pth"
        if model_final.exists():
            return str(model_final)
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint} does not contain model_final.pth."
        )

    if checkpoint_path.exists():
        return str(checkpoint_path)

    parent = checkpoint_path.parent
    suggestions: List[str] = []
    if parent.exists():
        suggestions = sorted(str(path) for path in parent.glob("*/model_final.pth"))[:20]
    raise FileNotFoundError(
        "Checkpoint not found: {}.\nAvailable nearby model_final.pth candidates:\n{}".format(
            checkpoint,
            "\n".join(suggestions) if suggestions else "(none found under parent directory)",
        )
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    runs = resolve_runs(args)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    if args.num_shards > 1:
        runs = [run for idx, run in enumerate(runs) if idx % args.num_shards == args.shard_index]
    data_root = os.environ.get("PSD2_DATA_ROOT", "")

    schema_path = args.schema_path or (
        str(Path(data_root) / "PRW" / "generated_schema.json") if data_root else None
    )
    train_attrs_path = args.train_attrs_path or (
        str(Path(data_root) / "PRW" / "prw_Final_Complete_cleaned_train.json")
        if data_root
        else None
    )
    test_attrs_path = args.test_attrs_path or (
        str(Path(data_root) / "PRW" / "prw_Final_Complete_cleaned_test.json")
        if data_root
        else None
    )

    if not schema_path or not train_attrs_path or not test_attrs_path:
        raise ValueError(
            "schema/train/test attribute paths must be provided, or PSD2_DATA_ROOT must be set."
        )

    print(f"Total presets: {len(runs)}")
    for run in runs:
        weights = str(run["weights"])
        run_label = run.get("label")
        run_extra_opts = run.get("extra_opts", [])
        weights_clean = weights.replace(" ", "")
        run_name = f"partw_{sanitize_weights(weights_clean)}"
        if run_label:
            run_name = f"{run_label}__{run_name}"
        output_dir_path = Path(args.output_root) / run_name
        output_dir = str(output_dir_path)
        if args.skip_completed and is_completed(output_dir_path):
            print(f"Skipping completed preset: {run_name}")
            continue
        cmd = [
            args.python_bin,
            "tools/train_ps_net.py",
            "--config-file",
            args.config_file,
            "--eval-only",
            "MODEL.WEIGHTS",
            checkpoint_path,
            "TEST.SEARCH_FUSION_MODE",
            args.fusion_mode,
            "TEST.SEARCH_AVG_GLOBAL_WEIGHT",
            str(args.global_weight),
            "TEST.SEARCH_AVG_PART_WEIGHTS",
            f"[{weights_clean}]",
            "PERSON_SEARCH.REID.SCHEMA_PATH",
            schema_path,
            "INPUT.ATTRIBUTE_ANNOTATIONS_PATH",
            train_attrs_path,
            "INPUT.TEST_ATTRIBUTE_ANNOTATIONS_PATH",
            test_attrs_path,
            "OUTPUT_DIR",
            output_dir,
        ]
        for opt in merge_extra_opts(args.extra_opt, run_extra_opts):
            cmd.extend(opt.split(" ", 1) if " " in opt else [opt])

        print("=" * 80)
        print("Running:", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
