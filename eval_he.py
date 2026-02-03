"""
Evaluator for Euclidean TSP on NPZ/NPY train dataset.

Candidate API (required):
    from typing import List, Tuple, Dict, Any
    Coord = Tuple[float, float]

    def solve_tsp_bounds(coords: List[Coord]) -> Dict[str, Any]:
        return {
            "tour": List[int],         # permutation 0..n-1 (tour)
            "lower_bound": float,      # ignored here (for compatibility)
            "witness": { "pi": List[float] }  # ignored here
        }

Dataset layout (hard-coded default):
    C:\\Users\\psgpe\\Downloads\\openevolve\\openevolve-main\\examples\\tsp\\data

There we expect files like:
    train_tsp_instance_100.npy   # shape: (M, 100, 2)
    train_tsp_sol_100.npy        # shape: (M, 101)  (closed tour, last = first)
    train_tsp_instance_200.npy   # shape: (M, 200, 2)
    train_tsp_sol_200.npy        # shape: (M, 201)
    ...
We only use *train* files, *test* игнорируем.

ИЗ КАЖДОГО train_tsp_instance_*.npy берём ТОЛЬКО ОДИН инстанс (k = 0).

Scoring:
    For each instance with optimal tour length OPT:
        rel_error = max(0, (L - OPT) / OPT),  L – candidate tour length.

    avg_rel_error = mean(rel_error_i) over all instances
    avg_time      = mean(time_per_instance)

    score_length = 1 / (1 + beta * avg_rel_error)
    score_time   = 1 / (1 + log1p(avg_time))

    combined_score = alpha * score_length + (1-alpha) * score_time
        with alpha = 0.7, beta = 3.0

Environment (optional):
    TSP_NPY_DIR        - override path to directory with NPY files
    TSP_SAVE_JSONL     - if set, save per-instance rows to this JSONL
    TSP_SAVE_DIR       - if set, save each instance row as separate JSON

Returned artifacts:
    "per_file_metrics" – list of dicts, по одному на каждый инстанс.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
from openevolve.evaluation_result import EvaluationResult

import importlib.util
import os
import time
import json
import math

import numpy as np


Coord = Tuple[float, float]


# ---------------------- geometry helpers ----------------------

def _dist(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tour_len_perm(tour: List[int], coords: List[Coord]) -> float:
    n = len(tour)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        s += _dist(coords[tour[i]], coords[tour[(i + 1) % n]])
    return s


def _valid_perm(tour: Any, n: int) -> bool:
    return (
        isinstance(tour, list)
        and len(tour) == n
        and all(isinstance(v, int) for v in tour)
        and set(tour) == set(range(n))
    )


# ---------------------- IO helpers ----------------------

def _ensure_dir_for_file(p: str):
    if not p:
        return
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def _save_jsonl(path: str, rows: List[Dict[str, Any]]):
    if not path:
        return
    _ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_per_instance_dir(dirpath: str, rows: List[Dict[str, Any]]):
    if not dirpath:
        return
    os.makedirs(dirpath, exist_ok=True)
    for r in rows:
        idx = int(r.get("dataset_idx", r.get("idx", 0)))
        n = int(r.get("N", -1))
        filename = f"idx_{idx:05d}_N{n}.json"
        with open(os.path.join(dirpath, filename), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


# ---------------------- dataset iterator ----------------------

def _parse_n_from_name(fname: str) -> int:
    """
    train_tsp_instance_100.npy -> 100
    train_tsp_instance_1000.npy -> 1000
    """
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot parse N from file name: {fname}")


def _iter_train_instances(data_dir: str):
    """
    Генератор по train_tsp_instance_*.npy.

    ИЗ КАЖДОГО файла берём ТОЛЬКО ОДИН инстанс: k = 0.

    Yield:
        (global_idx, N, coords_list, opt_perm, file_tag, local_idx)
    """
    files = os.listdir(data_dir)
    inst_files = sorted(
        f for f in files
        if f.startswith("train_tsp_instance_") and f.endswith(".npy")
    )

    global_idx = 0
    for inst_name in inst_files:
        n = _parse_n_from_name(inst_name)
        sol_name = f"train_tsp_sol_{n}.npy"
        if sol_name not in files:
            # пропускаем, если нет солюшенов
            continue

        inst_path = os.path.join(data_dir, inst_name)
        sol_path = os.path.join(data_dir, sol_name)

        instances = np.load(inst_path)   # shape: (M, N, 2)
        sols = np.load(sol_path)         # shape: (M, N+1), замкнутый тур

        if instances.shape[0] != sols.shape[0]:
            raise ValueError(
                f"Mismatch in count for {inst_name} and {sol_name}: "
                f"{instances.shape[0]} vs {sols.shape[0]}"
            )

        M = instances.shape[0]
        if M == 0:
            continue

        if instances.shape[1] != n:
            raise ValueError(
                f"Unexpected N in {inst_name}: parsed {n}, array has {instances.shape[1]}"
            )

        # Берём только первый инстанс из файла
        k = 0
        coords_arr = instances[k]          # (N, 2)
        sol_arr = sols[k]                 # (N+1,)

        coords_list: List[Coord] = [(float(x), float(y)) for x, y in coords_arr]

        opt_seq = sol_arr.astype(int).tolist()
        if len(opt_seq) == n + 1 and opt_seq[0] == opt_seq[-1]:
            opt_perm = opt_seq[:-1]
        else:
            opt_perm = opt_seq[:n]

        file_tag = inst_name
        yield (
            global_idx,
            n,
            coords_list,
            opt_perm,
            file_tag,
            k,
        )
        global_idx += 1


# ---------------------- safe import ----------------------

def _safe_import(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------- main evaluate ----------------------

def evaluate(program_path: str) -> EvaluationResult:
    t0_eval = time.time()

    # --- путь к данным ---
    DATA_DIR_DEFAULT = r"C:\Users\psgpe\Downloads\openevolve\openevolve-main\examples\tsp\data"
    DATA_DIR = os.getenv("TSP_NPY_DIR", DATA_DIR_DEFAULT).strip()

    SAVE_JSONL = os.getenv("TSP_SAVE_JSONL", "").strip()
    SAVE_DIR = os.getenv("TSP_SAVE_DIR", "").strip()

    if not DATA_DIR or not os.path.isdir(DATA_DIR):
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error_code": "data_dir_missing",
                "error_message": f"Data directory not found: {DATA_DIR}",
                "eval_time": 0.0,
            },
            artifacts={"per_file_metrics": []},
        )

    # --- load candidate ---
    try:
        mod = _safe_import(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error_code": "import_error",
                "error_message": f"{type(e).__name__}: {e}",
            },
            artifacts={"per_file_metrics": []},
        )

    if not hasattr(mod, "solve_tsp_bounds"):
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error_code": "missing_api",
                "error_message": "solve_tsp_bounds not found",
            },
            artifacts={"per_file_metrics": []},
        )

    per_rows: List[Dict[str, Any]] = []
    rel_errors: List[float] = []
    times: List[float] = []

    instances_total = 0
    instances_valid = 0

    # --- main loop over train instances (1 per file) ---
    for (
        global_idx,
        n,
        coords_list,
        opt_perm,
        file_tag,
        local_idx,
    ) in _iter_train_instances(DATA_DIR):

        instances_total += 1

        row: Dict[str, Any] = {
            "dataset_idx": global_idx,
            "file": file_tag,
            "file_local_idx": local_idx,
            "N": n,
        }

        # длина оптимального тура
        opt_len = _tour_len_perm(opt_perm, coords_list)
        row["opt_length"] = float(opt_len)

        t0 = time.time()
        try:
            out = mod.solve_tsp_bounds(coords_list)
        except Exception as e:
            dur = time.time() - t0
            times.append(dur)
            row.update({
                "status": "error",
                "error_code": "solve_exception",
                "error_message": f"{type(e).__name__}: {e}",
                "time_sec": float(dur),
            })
            per_rows.append(row)
            continue

        dur = time.time() - t0
        times.append(dur)

        tour = out.get("tour", None)
        if not _valid_perm(tour, n):
            row.update({
                "status": "invalid_tour",
                "error_code": "tour_invalid",
                "time_sec": float(dur),
            })
            per_rows.append(row)
            continue

        instances_valid += 1

        L = _tour_len_perm(tour, coords_list)
        row.update({
            "status": "ok",
            "tour_length": float(L),
            "time_sec": float(dur),
        })

        if opt_len > 0:
            rel_err = (L - opt_len) / opt_len
            rel_err = max(0.0, rel_err)
            row["rel_error"] = float(rel_err)
            rel_errors.append(rel_err)
        else:
            row["rel_error"] = None

        per_rows.append(row)

    # --- saving ---
    if SAVE_JSONL:
        _save_jsonl(SAVE_JSONL, per_rows)
    if SAVE_DIR:
        _save_per_instance_dir(SAVE_DIR, per_rows)

    eval_time = time.time() - t0_eval

    if instances_total == 0:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error_code": "no_instances",
                "error_message": "No train_tsp_instance_*.npy found (or all empty)",
                "eval_time": eval_time,
            },
            artifacts={"per_file_metrics": per_rows},
        )

    if instances_valid == 0:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error_code": "no_valid_tours",
                "error_message": "No valid tours produced on train dataset",
                "instances_total": instances_total,
                "instances_valid": instances_valid,
                "eval_time": eval_time,
            },
            artifacts={"per_file_metrics": per_rows},
        )

    # --- aggregate metrics ---
    avg_rel_error = float(sum(rel_errors) / len(rel_errors)) if rel_errors else float("inf")
    avg_time = float(sum(times) / len(times)) if times else float("inf")

    beta = 3.0
    if math.isfinite(avg_rel_error):
        score_len = 1.0 / (1.0 + beta * avg_rel_error)
    else:
        score_len = 0.0

    if math.isfinite(avg_time) and avg_time >= 0.0:
        score_time = 1.0 / (1.0 + math.log1p(avg_time))
    else:
        score_time = 0.0

    alpha = 0.7
    combined_score = alpha * score_len + (1.0 - alpha) * score_time

    metrics = {
        "instances_total": instances_total,
        "instances_valid": instances_valid,
        "avg_rel_error": avg_rel_error,
        "time_per_instance": avg_time,
        "score_length": score_len,
        "score_time": score_time,
        "combined_score": combined_score,
        "eval_time": eval_time,
        "data_dir": DATA_DIR,
    }

    return EvaluationResult(metrics=metrics, artifacts={"per_file_metrics": per_rows})


def evaluate_stage1(program_path: str):
    return evaluate(program_path)


def evaluate_stage2(program_path: str):
    return evaluate(program_path)


def evaluate_stage3(program_path: str):
    return evaluate(program_path)
