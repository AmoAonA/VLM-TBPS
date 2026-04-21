import argparse
import datetime as dt
import subprocess
from pathlib import Path


def run_command(command):
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"[command failed] {' '.join(command)} :: {exc}"

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return stdout or "(empty)"
    if stderr and stdout:
        return f"{stdout}\n{stderr}"
    return stderr or stdout or f"[exit {completed.returncode}]"


def read_tail(log_path, tail_lines):
    path = Path(log_path)
    if not path.exists():
        return f"[missing] {path}"
    return run_command(["tail", "-n", str(tail_lines), str(path)])


def file_age_minutes(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    modified = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return (dt.datetime.now() - modified).total_seconds() / 60.0


def main():
    parser = argparse.ArgumentParser(description="Append a lightweight job health snapshot.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--process-pattern", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tail-lines", type=int, default=5)
    parser.add_argument("--heartbeat-file", default=None)
    parser.add_argument("--max-stale-minutes", type=int, default=90)
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    process_snapshot = run_command(
        [
            "bash",
            "-lc",
            (
                f"ps -ef | grep -F -- \"{args.process_pattern}\" "
                "| grep -v grep | grep -v hourly_job_healthcheck.py || true"
            ),
        ]
    )
    gpu_snapshot = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_snapshot = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    log_tail = read_tail(args.training_log, args.tail_lines)
    heartbeat_file = args.heartbeat_file or args.training_log
    heartbeat_age = file_age_minutes(heartbeat_file)

    has_process = bool(process_snapshot and process_snapshot != "(empty)")
    if has_process:
        status = "running"
    elif heartbeat_age is not None and heartbeat_age <= args.max_stale_minutes:
        status = f"fresh_log({heartbeat_age:.1f}m)"
    elif heartbeat_age is not None:
        status = f"stale_log({heartbeat_age:.1f}m)"
    else:
        status = "missing"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {args.label} status={status}\n")
        handle.write(f"[heartbeat] {heartbeat_file}\n")
        handle.write("[process]\n")
        handle.write(f"{process_snapshot}\n")
        handle.write("[gpus]\n")
        handle.write(f"{gpu_snapshot}\n")
        handle.write("[compute]\n")
        handle.write(f"{compute_snapshot}\n")
        handle.write("[log_tail]\n")
        handle.write(f"{log_tail}\n")
        handle.write("=" * 80 + "\n")


if __name__ == "__main__":
    main()
