import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import jax
import jax.numpy as jnp

from src.envs.block_moving.block_moving_env import BoxMovingEnv
from src.optimal_q import solve_state


@dataclass
class ComboResult:
    grid_size: int
    boxes: int
    episodes: int
    completed: int
    success: int
    success_rate: float | None
    path_len_min: int | None
    path_len_mean: float | None
    path_len_max: int | None
    exec_ms_min: float | None
    exec_ms_mean: float | None
    exec_ms_max: float | None
    combo_wall_s: float
    errors: list[str]


def _safe_stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return min(values), sum(values) / len(values), max(values)


def run_combo(grid_size: int, boxes: int, seeds: list[int], episode_length: int = 1000) -> ComboResult:
    combo_t0 = time.perf_counter()
    paths: list[int] = []
    exec_ms: list[float] = []
    success = 0
    errors: list[str] = []

    try:
        env = BoxMovingEnv(
            grid_size=grid_size,
            episode_length=episode_length,
            number_of_boxes_min=boxes,
            number_of_boxes_max=boxes,
            number_of_moving_boxes_max=boxes,
            level_generator="default",
        )
        step_fn = jax.jit(env.step)
        # Warmup JIT once per combo.
        warm_key = jax.random.PRNGKey(0)
        warm_state, _ = env.reset(warm_key)
        _ = step_fn(warm_state, jnp.array(0, dtype=jnp.int8))
    except Exception as e:
        return ComboResult(
            grid_size=grid_size,
            boxes=boxes,
            episodes=len(seeds),
            completed=0,
            success=0,
            success_rate=0.0,
            path_len_min=None,
            path_len_mean=None,
            path_len_max=None,
            exec_ms_min=None,
            exec_ms_mean=None,
            exec_ms_max=None,
            combo_wall_s=time.perf_counter() - combo_t0,
            errors=[f"env_init: {type(e).__name__}: {e}"],
        )

    for seed in seeds:
        try:
            t0 = time.perf_counter()
            key = jax.random.PRNGKey(seed)
            state, _ = env.reset(key)

            actions = solve_state(state.grid)
            current = state
            for action in actions:
                current, _, _, _ = step_fn(current, jnp.array(action, dtype=jnp.int8))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ok = bool(current.success)

            paths.append(len(actions))
            exec_ms.append(elapsed_ms)
            success += int(ok)
        except Exception as e:
            errors.append(f"seed={seed}: {type(e).__name__}: {e}")

    pmin, pmean, pmax = _safe_stats([float(v) for v in paths])
    tmin, tmean, tmax = _safe_stats(exec_ms)

    return ComboResult(
        grid_size=grid_size,
        boxes=boxes,
        episodes=len(seeds),
        completed=len(paths),
        success=success,
        success_rate=(success / len(seeds)) if seeds else None,
        path_len_min=int(pmin) if pmin is not None else None,
        path_len_mean=pmean,
        path_len_max=int(pmax) if pmax is not None else None,
        exec_ms_min=tmin,
        exec_ms_mean=tmean,
        exec_ms_max=tmax,
        combo_wall_s=time.perf_counter() - combo_t0,
        errors=errors,
    )


def run_stage(stage_name: str, combos: list[tuple[int, int]], seeds: list[int]) -> list[ComboResult]:
    print(f"[stage-start] {stage_name} combos={len(combos)} seeds={len(seeds)}")
    t0 = time.perf_counter()
    out: list[ComboResult] = []
    for idx, (g, b) in enumerate(combos, start=1):
        res = run_combo(g, b, seeds)
        out.append(res)
        print(
            f"[combo] {idx}/{len(combos)} g={g} b={b} "
            f"success={res.success}/{res.episodes} "
            f"path_mean={res.path_len_mean} exec_ms_mean={res.exec_ms_mean}"
        )
    print(f"[stage-end] {stage_name} wall_s={time.perf_counter() - t0:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OT-based optimal_q planner.")
    parser.add_argument("--full-seeds", type=int, default=5, help="Episodes per combo in full sweep.")
    parser.add_argument("--small-seeds", type=int, default=3, help="Episodes per combo in small warmup stage.")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/benchmark_optimal_q_results.json",
        help="Path for JSON benchmark output.",
    )
    args = parser.parse_args()

    grid_sizes = [3, 4, 5]
    box_counts = [3, 4, 5, 6]

    # Smaller settings first to ensure practical runtime before larger combos.
    small_combos = [(3, 3), (3, 4), (4, 3), (4, 4)]
    full_combos = [(g, b) for g in grid_sizes for b in box_counts]

    bench_t0 = time.perf_counter()
    small_results = run_stage("small-first", small_combos, list(range(args.small_seeds)))
    full_results = run_stage("full", full_combos, list(range(args.full_seeds)))
    bench_wall_s = time.perf_counter() - bench_t0

    summary = {
        "grid_sizes": grid_sizes,
        "box_counts": box_counts,
        "small_stage_seeds": args.small_seeds,
        "full_stage_seeds": args.full_seeds,
        "all_successful_full": all(r.success == r.episodes for r in full_results),
        "benchmark_wall_s": bench_wall_s,
    }

    payload = {
        "summary": summary,
        "small_stage": [asdict(r) for r in small_results],
        "full_stage": [asdict(r) for r in full_results],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[results-json] {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
