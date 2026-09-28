"""Run one C1 benchmark cell under resource monitoring and append one CSV row.

Usage:
  python monitor.py --method M --mode MODE --scale N --rep R --device cpu|gpu \
      [--timeout-hours 24] -- <runner command ...>

The runner receives C1_RESULT_JSON in its environment and should write a JSON with
fit_seconds, load_seconds, n_bins, n_genes, n_types, version, notes, status
("OK" or "OOM"/"ERROR" when it catches the failure itself) and optionally
peak_gpu_gb (framework-reported peak allocation).

Memory: the process tree (runner and all children, e.g. forked RCTD workers) is
sampled every second; peak_rss_gb is the peak of the summed PSS (proportional set
size, so copy-on-write pages shared by forked workers are not double counted; for a
single process PSS ~= RSS). The summed-RSS peak and the cgroup memory.peak (which
also counts page cache) are reported in notes.
GPU: nvidia-smi memory.used is polled every 2 s (the GPU is exclusively allocated by
SLURM); peak_gpu_gb is max(polled, framework-reported).
Status: OK, OOM (cgroup oom_kill event, or runner-reported allocation failure),
DNF (wall-clock cap reached), ERROR (anything else).
"""
import argparse
import csv
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

COLUMNS = [
    "method", "mode", "scale", "n_bins", "n_genes", "n_types", "device", "n_cores",
    "hostname", "cpu_model", "gpu_model", "fit_seconds", "load_seconds", "peak_rss_gb",
    "peak_gpu_gb", "status", "version", "notes",
    "rep", "n_bins_fit", "total_wall_seconds", "mem_limit_gb", "slurm_job_id", "timestamp",
]
FINAL_STATUSES = {"OK", "OOM", "DNF"}


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def gpu_query(field):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            return [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def own_cgroup_dir():
    try:
        rel = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
        return Path("/sys/fs/cgroup" + rel)
    except (OSError, IndexError):
        return None


def read_oom_kills(cg):
    total = 0
    for d in [cg, cg.parent if cg else None]:
        if d is None:
            continue
        try:
            for line in open(d / "memory.events"):
                k, v = line.split()
                if k == "oom_kill":
                    total += int(v)
        except OSError:
            pass
    return total


def read_cgroup_peak_gb(cg):
    vals = []
    for d in [cg, cg.parent if cg else None]:
        if d is None:
            continue
        try:
            vals.append(int(open(d / "memory.peak").read().strip()) / 1e9)
        except (OSError, ValueError):
            pass
    return max(vals) if vals else float("nan")


def done_keys(csv_path):
    keys = set()
    if not csv_path.exists():
        return keys
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") in FINAL_STATUSES:
                keys.add((row["method"], row["mode"], str(row["scale"]), str(row["rep"])))
    return keys


def append_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        empty = f.read(1) == ""
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if empty:
            w.writeheader()
        w.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


class Sampler(threading.Thread):
    def __init__(self, pid, poll_gpu):
        super().__init__(daemon=True)
        self.pid = pid
        self.poll_gpu = poll_gpu
        self.peak_pss = 0.0
        self.peak_rss = 0.0
        self.peak_gpu = 0.0
        self.stop_evt = threading.Event()

    def run(self):
        last_gpu = 0.0
        while not self.stop_evt.is_set():
            try:
                root = psutil.Process(self.pid)
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                procs = []
            rss = pss = 0
            for p in procs:
                try:
                    mi = p.memory_full_info()
                    rss += mi.rss
                    pss += getattr(mi, "pss", mi.rss)
                except psutil.Error:
                    try:
                        m = p.memory_info().rss
                        rss += m
                        pss += m
                    except psutil.Error:
                        pass
            self.peak_rss = max(self.peak_rss, rss / 1e9)
            self.peak_pss = max(self.peak_pss, pss / 1e9)
            now = time.time()
            if self.poll_gpu and now - last_gpu >= 2.0:
                used = gpu_query("memory.used")
                if used:
                    self.peak_gpu = max(self.peak_gpu, max(float(u) for u in used) / 1024.0)
                last_gpu = now
            self.stop_evt.wait(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--scale", required=True, type=int)
    ap.add_argument("--rep", default=1, type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout-hours", default=24.0, type=float)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    csv_path = Path(a.csv)

    key = (a.method, a.mode, str(a.scale), str(a.rep))
    if not a.force and key in done_keys(csv_path):
        print(f"[monitor] {key} already recorded with a final status; skipping")
        return 0

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.method}_{a.mode}_{a.scale}_r{a.rep}"
    result_json = workdir / f"{tag}.json"
    if result_json.exists():
        result_json.unlink()
    env = dict(os.environ, C1_RESULT_JSON=str(result_json))

    cg = own_cgroup_dir()
    oom_before = read_oom_kills(cg)
    gpu_model = ";".join(gpu_query("name")) if a.device == "gpu" else ""
    t0 = time.time()
    print(f"[monitor] start {tag} on {socket.gethostname()}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    sampler = Sampler(proc.pid, poll_gpu=(a.device == "gpu"))
    sampler.start()
    timed_out = False
    try:
        rc = proc.wait(timeout=a.timeout_hours * 3600)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        rc = proc.returncode
    total = time.time() - t0
    sampler.stop_evt.set()
    sampler.join(timeout=30)
    oom_after = read_oom_kills(cg)

    res = {}
    if result_json.exists():
        try:
            res = json.loads(result_json.read_text())
        except json.JSONDecodeError:
            res = {}

    notes = [res.get("notes", "")] if res.get("notes") else []
    if timed_out:
        status = "DNF"
        notes.append(f"killed at {a.timeout_hours} h cap")
    elif oom_after > oom_before:
        status = "OOM"
        notes.append(f"cgroup oom_kill events={oom_after - oom_before}; rc={rc}")
    elif res.get("status") in ("OOM", "ERROR"):
        status = res["status"]
    elif rc == 0 and res.get("status") == "OK":
        status = "OK"
    else:
        status = "ERROR"
        notes.append(f"rc={rc}")
        if rc is not None and rc < 0:
            notes.append(f"signal={-rc}")

    peak_gpu = max(sampler.peak_gpu, float(res.get("peak_gpu_gb") or 0.0))
    cg_peak = read_cgroup_peak_gb(cg)
    notes.append(
        f"peak_rss_sum_gb={sampler.peak_rss:.2f}; cgroup_peak_gb={cg_peak:.2f}"
        + (f"; torch_peak_alloc_gb={float(res['peak_gpu_gb']):.2f}" if res.get("peak_gpu_gb") else "")
    )
    mem_limit = ""
    d = cg
    while d is not None and str(d) != "/sys/fs/cgroup":
        try:
            mem_limit = f"{int(open(d / 'memory.max').read()) / 2**30:.0f}"
            break
        except (OSError, ValueError):
            d = d.parent

    row = {
        "method": a.method, "mode": a.mode, "scale": a.scale,
        "n_bins": res.get("n_bins", a.scale), "n_genes": res.get("n_genes", ""),
        "n_types": res.get("n_types", ""), "device": a.device,
        "n_cores": os.environ.get("SLURM_CPUS_PER_TASK", str(len(os.sched_getaffinity(0)))),
        "hostname": socket.gethostname(), "cpu_model": cpu_model(), "gpu_model": gpu_model,
        "fit_seconds": res.get("fit_seconds", "") if status == "OK" else res.get("fit_seconds", ""),
        "load_seconds": res.get("load_seconds", ""),
        "peak_rss_gb": round(sampler.peak_pss, 3),
        "peak_gpu_gb": round(peak_gpu, 3) if a.device == "gpu" else "",
        "status": status, "version": res.get("version", ""),
        "notes": " | ".join(n for n in notes if n),
        "rep": a.rep, "n_bins_fit": res.get("n_bins_fit", ""),
        "total_wall_seconds": round(total, 1), "mem_limit_gb": mem_limit,
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
        + (f"_{os.environ['SLURM_ARRAY_TASK_ID']}" if os.environ.get("SLURM_ARRAY_TASK_ID") else ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_row(csv_path, row)
    print(f"[monitor] {tag}: status={status} fit={row['fit_seconds']} total={total:.1f}s "
          f"peak_pss={sampler.peak_pss:.2f}GB peak_gpu={peak_gpu:.2f}GB", flush=True)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
