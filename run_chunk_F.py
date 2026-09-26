from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BACKBONES = ("efficientnet_b4", "densenet121")
SEEDS = (42, 123, 2026)

ARMS = {
    "F1": ("proposed_km", "kermany", "focal_smoothing", BACKBONES),
    "F2": ("stage1none", "none", "focal_smoothing", BACKBONES),
    "F3": ("loss_ce", "kermany", "ce_smoothing", BACKBONES),
    "R": ("baseline_resnet50", "none", "ce_smoothing", ("resnet50",)),
    "F4": ("proposed_ce", "none", "ce_smoothing", BACKBONES),
}


def hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def jobs_for(chunk: str) -> list[tuple]:
    if chunk == "S":
        return [("stage1", bb, None) for bb in BACKBONES]
    return [("train", bb, seed) for bb in ARMS[chunk][3] for seed in SEEDS]


def build_command(job, gpu: int, a) -> list[str]:
    kind, backbone, seed = job
    base = [sys.executable, "leverF_pipeline.py",
            "--backbone", backbone,
            "--cache-dir", a.cache_dir,
            "--manifest-csv", a.manifest_csv,
            "--out-root", a.out_root,
            "--num-workers", str(a.num_workers)]
    if a.rsna_root:
        base += ["--rsna-root", a.rsna_root]

    if kind == "stage1":
        cmd = base + ["--stage", "stage1", "--stage1", "kermany"]
        if a.kermany_root:
            cmd += ["--kermany-root", a.kermany_root]
        return cmd

    arm, stage1, loss, _ = ARMS[a.chunk]
    cmd = base + ["--stage", "train", "--arm", arm, "--seed", str(seed),
                  "--stage1", stage1, "--loss", loss,
                  "--patience", str(a.patience)]
    if not a.refit:
        cmd += ["--no-refit"]
    if stage1 == "kermany" and a.kermany_root:
        cmd += ["--kermany-root", a.kermany_root]
    if a.full_epochs:
        cmd += ["--full-epochs", str(a.full_epochs)]
    if a.batch_size:
        cmd += ["--batch-size", str(a.batch_size)]
    return cmd


def label(job, chunk: str) -> str:
    kind, backbone, seed = job
    if kind == "stage1":
        return f"stage1_kermany__{backbone}"
    return f"{ARMS[chunk][0]}__{backbone}__seed{seed}"


def launch(job, gpu: int, a):
    name = label(job, a.chunk)
    log_path = Path(a.log_dir) / f"{name}.log"
    log = open(log_path, "w", buffering=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = build_command(job, gpu, a)
    log.write(f"# GPU {gpu}\n# {subprocess.list2cmdline(cmd)}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"proc": proc, "name": name, "log": log, "path": log_path,
            "start": time.time()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True, choices=["S", "F1", "F2", "F3", "R", "F4"])
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="use a separate folder (leverF caches at 256 px)")
    ap.add_argument("--kermany-root", default="",
                    help="required for chunks S, F1 and F3")
    ap.add_argument("--rsna-root", default="",
                    help="optional; leverF_pipeline.py finds it otherwise")
    ap.add_argument("--out-root", default="/kaggle/working/rsna_leverF_results")
    ap.add_argument("--log-dir", default="/kaggle/working/logs_F")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2,
                    help="per process (Kaggle has 4 CPUs for both GPUs)")
    ap.add_argument("--patience", type=int, default=6,
                    help="early stopping on calibration AUC (0 = off). Use the same value for all arms.")
    ap.add_argument("--refit", action="store_true",
                    help="refit on train + calibration (off by default: the paper trains on the 70%% split only)")
    ap.add_argument("--full-epochs", type=int, default=0, help="0 = pipeline default")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = pipeline default")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.chunk in ("S", "F1", "F3") and not a.kermany_root:
        sys.exit(f"chunk {a.chunk} uses Kermany stage-1 - pass --kermany-root")
    if "png_cache_f" not in a.cache_dir and a.cache_dir.rstrip("/").endswith("png_cache"):
        print("WARNING: --cache-dir may be the old 544 px cache; leverF needs its own 256 px\n"
              "         cache (e.g. /tmp/png_cache_f).\n", flush=True)

    for d in (a.out_root, a.log_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    jobs = jobs_for(a.chunk)
    rounds = [jobs[i:i + a.gpus] for i in range(0, len(jobs), a.gpus)]

    if a.chunk == "S":
        print(f"chunk S: Kermany stage-1 for {len(jobs)} backbones\n")
    else:
        arm, stage1, loss, _ = ARMS[a.chunk]
        print(f"chunk {a.chunk}: arm={arm}  stage1={stage1}  loss={loss}  "
              f"refit={'on' if a.refit else 'OFF'}  patience={a.patience}")
        print(f"{len(jobs)} members in {len(rounds)} rounds across {a.gpus} GPUs\n")
    for r, group in enumerate(rounds, 1):
        for g, job in enumerate(group):
            print(f"  round {r}  gpu {g}  {label(job, a.chunk)}")
    print()

    if a.dry_run:
        print("first job would run:")
        print("   ", subprocess.list2cmdline(build_command(jobs[0], 0, a)))
        return

    t0 = time.time()
    failed = []
    for r, group in enumerate(rounds, 1):
        print(f"=== round {r}/{len(rounds)} starting ===", flush=True)
        running = [launch(job, g, a) for g, job in enumerate(group)]
        for w in running:
            code = w["proc"].wait()
            w["log"].close()
            print(f"  [{hms(time.time() - w['start'])}] {w['name']}: "
                  f"{'ok' if code == 0 else f'FAILED (exit {code})'}", flush=True)
            if code:
                failed.append(w)
        print(f"=== round {r} done, elapsed {hms(time.time() - t0)} ===\n", flush=True)

    print(f"chunk {a.chunk} finished in {hms(time.time() - t0)}")
    if failed:
        print("\nFAILED - inspect these logs before evaluating:")
        for w in failed:
            print(f"  {w['name']}  ->  {w['path']}")
        sys.exit(1)

    if a.chunk == "S":
        print("\nStage-1 weights written. Next: --chunk F1")
    else:
        arm = ARMS[a.chunk][0]
        print(f"\nAll members done. Now evaluate the arm:")
        print(f"  !python leverF_pipeline.py --stage evaluate --arm {arm} "
              f"--out-root {a.out_root} --cache-dir {a.cache_dir} "
              f"--manifest-csv {a.manifest_csv}")
        print(f"\n(The no-TTA scores are saved as well, no retraining needed.)")


if __name__ == "__main__":
    main()
