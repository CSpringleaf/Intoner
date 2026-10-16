#!/usr/bin/env python3
"""Train PIS: acoustic warmup (120k steps), then shallow diffusion (160k steps). Re-run to resume."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from intoner.paths import CHECKPOINT_DIR, PIS_DATA_DIR, REPO_ROOT, require_file, resolve

STAGES = {"warmup": ("configs/pis/warmup.yaml", 120000), "diffusion": ("configs/pis/diffusion.yaml", 160000)}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, state):
    state["updated_at"] = now()
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("warmup", "diffusion", "all"), default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--exp-prefix", default="pis", help="checkpoints go to checkpoints/<prefix>_<stage>")
    parser.add_argument("--data-dir", type=Path, default=PIS_DATA_DIR / "binary")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.exp_prefix):
        parser.error("--exp-prefix may contain letters, digits, '_' and '-' only")
    data_dir = resolve(args.data_dir)
    require_file(data_dir / "phone_set.json", "PIS binary data")
    os.chdir(REPO_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PIS training requires a CUDA GPU")
    log_dir = REPO_ROOT / "logs" / args.exp_prefix
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another training run already uses --exp-prefix {args.exp_prefix}") from exc
    state_path = log_dir / "status.json"
    state = {"started_at": now(), "driver_pid": os.getpid(), "gpu": args.gpu,
             "gpu_name": torch.cuda.get_device_name(0), "git_commit": git_commit(),
             "stage_requested": args.stage, "data_dir": str(data_dir), "events": []}
    for stage in (("warmup", "diffusion") if args.stage == "all" else (args.stage,)):
        config, steps = STAGES[stage]
        experiment = f"{args.exp_prefix}_{stage}"
        target = CHECKPOINT_DIR / experiment / f"model_ckpt_steps_{steps}.ckpt"
        if target.exists():
            checkpoint = torch.load(target, map_location="cpu")
            if checkpoint.get("global_step", -1) < steps or not checkpoint.get("optimizer_states"):
                raise RuntimeError(f"incomplete training checkpoint: {target}")
            del checkpoint
            state["events"].append({"stage": stage, "status": "already_complete", "checkpoint": str(target), "time": now()})
            write_status(state_path, state)
            continue
        overrides = f"max_updates={steps},binary_data_dir={data_dir}"
        if stage == "diffusion":
            warmup = CHECKPOINT_DIR / f"{args.exp_prefix}_warmup" / "model_ckpt_steps_120000.ckpt"
            require_file(warmup, "warmup checkpoint", hint="Run --stage warmup first.")
            overrides += f",fs2_ckpt={warmup}"
        command = [sys.executable, "-u", "-m", "intoner.pis.run", "--config", config,
                   "--exp_name", experiment, "--hparams", overrides]
        log_path = log_dir / f"{stage}.log"
        print(f"{now()} starting {stage}; log: {log_path}", flush=True)
        with log_path.open("a", buffering=1) as log:
            log.write("\n" + json.dumps({"started_at": now(), "command": command, "git_commit": state["git_commit"]}) + "\n")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="running", active_stage=stage, training_pid=process.pid, log=str(log_path))
            write_status(state_path, state)
            returncode = process.wait()
        state["events"].append({"stage": stage, "returncode": returncode, "checkpoint": str(target), "time": now()})
        if returncode != 0 or not target.exists():
            state.update(status="failed", failure=f"{stage} exited with {returncode}; see {log_path}")
            write_status(state_path, state)
            print(state["failure"], file=sys.stderr, flush=True)
            return returncode or 1
    state.update(status="complete", active_stage=None, training_pid=None)
    write_status(state_path, state)
    print(f"{now()} requested PIS stages complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
