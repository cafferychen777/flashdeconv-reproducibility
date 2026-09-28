"""Shared helpers for S1 Python runners (copied from runtime_benchmark_c1)."""
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

SAVE_PROPS_MAX_SCALE = 10_000


def write_result(**kw):
    path = os.environ.get("C1_RESULT_JSON")
    if path:
        with open(path, "w") as f:
            json.dump(kw, f, default=str)
    print("[runner] result:", json.dumps(kw, default=str), flush=True)


def is_oom(exc):
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(s in msg for s in ("out of memory", "memoryerror", "cannot allocate",
                                  "unable to allocate", "cuda error: out of memory"))


def run_guarded(fn):
    """Run fn() -> dict; on exception, record OOM/ERROR with a short message."""
    try:
        res = fn()
        res.setdefault("status", "OK")
        write_result(**res)
        return 0
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        status = "OOM" if is_oom(exc) else "ERROR"
        write_result(status=status, notes=f"{type(exc).__name__}: {str(exc)[:300]}",
                     **getattr(exc, "c1_partial", {}))
        return 1


def save_props(props, obs_names, type_names, out_path):
    df = pd.DataFrame(np.asarray(props, dtype=np.float32), index=obs_names, columns=type_names)
    df.to_csv(out_path, float_format="%.5g", compression="gzip")


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
