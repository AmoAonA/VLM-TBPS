#!/usr/bin/env python
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run overnight cache-based gender filtering sweeps."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gallery-file", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="prw", choices=["prw", "cuhk"])
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--fusion-mode", default="global", choices=["global", "avg"])
    parser.add_argument("--avg-global-weight", type=float, default=1.0)
    parser.add_argument("--avg-part-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--full-score-matrix", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--preset",
        default="overnight",
        choices=["quick", "overnight"],
    )
    return parser.parse_args()


def build_runs(preset):
    runs = [{"label": "baseline_none", "opts": ["--gender-filter-mode", "none"]}]
    if preset == "quick":
        runs.extend(
            [
                {
                    "label": "soft_p010",
                    "opts": ["--gender-filter-mode", "soft", "--gender-soft-penalty", "0.10"],
                },
                {
                    "label": "soft_p015",
                    "opts": ["--gender-filter-mode", "soft", "--gender-soft-penalty", "0.15"],
                },
                {
                    "label": "topk_hard_200",
                    "opts": ["--gender-filter-mode", "topk_hard", "--gender-topk", "200"],
                },
                {
                    "label": "selective_k045_r030",
                    "opts": [
                        "--gender-filter-mode",
                        "selective_hard",
                        "--gender-selective-min-keep-ratio",
                        "0.45",
                        "--gender-selective-min-known-ratio",
                        "0.30",
                    ],
                },
            ]
        )
        return runs

    for penalty in (0.05, 0.10, 0.15, 0.20, 0.30):
        runs.append(
            {
                "label": f"soft_p{penalty:.2f}".replace(".", "p"),
                "opts": ["--gender-filter-mode", "soft", "--gender-soft-penalty", f"{penalty:.2f}"],
            }
        )
    for topk in (50, 100, 200, 500):
        runs.append(
            {
                "label": f"topk_hard_{topk}",
                "opts": ["--gender-filter-mode", "topk_hard", "--gender-topk", str(topk)],
            }
        )
    for keep_ratio in (0.35, 0.45, 0.55):
        for known_ratio in (0.20, 0.30, 0.40):
            runs.append(
                {
                    "label": f"selective_k{keep_ratio:.2f}_r{known_ratio:.2f}".replace(".", "p"),
                    "opts": [
                        "--gender-filter-mode",
                        "selective_hard",
                        "--gender-selective-min-keep-ratio",
                        f"{keep_ratio:.2f}",
                        "--gender-selective-min-known-ratio",
                        f"{known_ratio:.2f}",
                    ],
                }
            )
    return runs


def parse_metrics(text):
    metrics = {}
    for key in ("mAP", "top1", "top5", "top10", "mAP_mlv", "top1_mlv", "top5_mlv", "top10_mlv"):
        matched = re.search(rf"^{re.escape(key)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text, re.M)
        if matched:
            metrics[key] = float(matched.group(1))
    return metrics


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_jsonl = output_root / "summary.jsonl"
    summary_tsv = output_root / "summary.tsv"

    runs = build_runs(args.preset)
    eval_script = repo_root / "tools" / "eval_search_from_cache.py"

    header = "label\tmAP\ttop1\ttop5\ttop10\tmAP_mlv\ttop1_mlv\ttop5_mlv\ttop10_mlv\n"
    if not summary_tsv.exists():
        summary_tsv.write_text(header, encoding="utf-8")

    for idx, run in enumerate(runs, start=1):
        run_dir = output_root / run["label"]
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "stdout.log"

        if args.skip_completed and log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if "mAP:" in text:
                print(f"[{idx}/{len(runs)}] skip {run['label']}")
                continue

        cmd = [
            args.python_bin,
            str(eval_script),
            "--dataset",
            args.dataset,
            "--gallery-file",
            args.gallery_file,
            "--query-file",
            args.query_file,
            "--score-thresh",
            str(args.score_thresh),
            "--fusion-mode",
            args.fusion_mode,
            "--device",
            args.device,
            "--query-batch-size",
            str(args.query_batch_size),
        ]
        if args.max_queries > 0:
            cmd.extend(["--max-queries", str(args.max_queries)])
        if args.full_score_matrix:
            cmd.append("--full-score-matrix")
        if args.fusion_mode == "avg":
            cmd.extend(["--avg-global-weight", str(args.avg_global_weight)])
            if args.avg_part_weights:
                cmd.append("--avg-part-weights")
                cmd.extend([str(x) for x in args.avg_part_weights])
        cmd.extend(run["opts"])

        print(f"[{idx}/{len(runs)}] {run['label']}")
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        text = result.stdout + ("\n" + result.stderr if result.stderr else "")
        log_path.write_text(text, encoding="utf-8")

        metrics = parse_metrics(text)
        record = {
            "label": run["label"],
            "returncode": result.returncode,
            "cmd": cmd,
            "metrics": metrics,
        }
        with summary_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with summary_tsv.open("a", encoding="utf-8") as f:
            f.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    run["label"],
                    metrics.get("mAP", ""),
                    metrics.get("top1", ""),
                    metrics.get("top5", ""),
                    metrics.get("top10", ""),
                    metrics.get("mAP_mlv", ""),
                    metrics.get("top1_mlv", ""),
                    metrics.get("top5_mlv", ""),
                    metrics.get("top10_mlv", ""),
                )
            )

        if result.returncode != 0:
            print(text)
            print(f"run failed: {run['label']}")


if __name__ == "__main__":
    main()
