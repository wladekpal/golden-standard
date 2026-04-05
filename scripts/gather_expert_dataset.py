#!/usr/bin/env python3
"""Collect expert offline trajectories for BoxMoving and save them as a .npy dataset.

Example usage:
  python scripts/gather_expert_dataset.py \
    --output-path data/expert_default_6x6_5boxes.npy \
    --num-trajectories 1000 \
    --level-generator default \
    --grid-size 6 \
    --number-of-boxes-min 5 \
    --number-of-boxes-max 5 \
    --number-of-moving-boxes-max 5

  python scripts/gather_expert_dataset.py \
    --output-path data/expert_variable_special_6x6_5boxes.npy \
    --num-trajectories 1000 \
    --level-generator variable \
    --generator-special \
    --quarter-size 3 \
    --grid-size 6 \
    --number-of-boxes-min 5 \
    --number-of-boxes-max 5 \
    --number-of-moving-boxes-max 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from envs.block_moving.block_moving_env import BoxMovingEnv
    from optimal_q import solve_state_vmapped
except ModuleNotFoundError:
    from src.envs.block_moving.block_moving_env import BoxMovingEnv
    from src.optimal_q import solve_state_vmapped


@dataclass
class Summary:
    requested_trajectories: int
    collected_trajectories: int
    attempted_rollouts: int
    skipped_rollouts: int
    total_transitions: int
    success_rate_attempted: float
    collect_time_seconds: float


def log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gather expert offline dataset using solve_state planner.")

    parser.add_argument("--output-path", type=str, required=True, help="Output .npy path for collected dataset.")
    parser.add_argument("--num-trajectories", type=int, required=True, help="Number of trajectories to collect.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum rollout attempts. Default: max(10 * num_trajectories, num_trajectories).",
    )
    parser.add_argument(
        "--allow-failed-trajectories",
        action="store_true",
        help="If set, unsuccessful trajectories are kept. By default they are skipped.",
    )

    parser.add_argument("--level-generator", type=str, choices=["default", "variable"], default="default")
    parser.add_argument(
        "--generator-special",
        action="store_true",
        help="Enable special variable generator mode (used only with --level-generator variable).",
    )
    parser.add_argument(
        "--quarter-size",
        type=int,
        default=None,
        help="Quarter size for variable generator. If omitted, env default (grid_size // 2) is used.",
    )

    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument(
        "--fixed-length",
        type=int,
        default=None,
        help=(
            "Stored trajectory length (in transitions). If omitted, uses --episode-length. "
            "Trajectories are padded to this length so all have identical shapes."
        ),
    )
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=1,
        help="Number of environments to run in parallel while collecting trajectories.",
    )
    parser.add_argument(
        "--planner-max-actions",
        type=int,
        default=512,
        help="Max padded action length used by vmapped solve_state planner.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print collection progress every N batches.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable runtime progress logs.",
    )
    parser.add_argument("--number-of-boxes-min", type=int, default=5)
    parser.add_argument("--number-of-boxes-max", type=int, default=5)
    parser.add_argument("--number-of-moving-boxes-max", type=int, default=5)

    return parser



def validate_args(args: argparse.Namespace) -> None:
    if args.num_trajectories <= 0:
        raise ValueError("--num-trajectories must be > 0")
    if args.parallel_envs <= 0:
        raise ValueError("--parallel-envs must be > 0")
    if args.planner_max_actions <= 0:
        raise ValueError("--planner-max-actions must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")

    fixed_length = args.fixed_length if args.fixed_length is not None else args.episode_length
    if fixed_length <= 0:
        raise ValueError("--fixed-length (or --episode-length) must be > 0")

    if args.level_generator == "variable":
        quarter_size = args.quarter_size if args.quarter_size is not None else args.grid_size // 2
        if args.number_of_boxes_max > quarter_size * quarter_size:
            raise ValueError(
                "For variable generator, number_of_boxes_max must be <= quarter_size^2. "
                f"Got {args.number_of_boxes_max} > {quarter_size}^2."
            )
        if args.number_of_boxes_max != args.number_of_moving_boxes_max:
            raise ValueError(
                "For variable generator, number_of_boxes_max must equal number_of_moving_boxes_max. "
                f"Got {args.number_of_boxes_max} vs {args.number_of_moving_boxes_max}."
            )
        if args.number_of_boxes_min != args.number_of_boxes_max:
            raise ValueError(
                "For variable generator, number_of_boxes_min must equal number_of_boxes_max. "
                f"Got {args.number_of_boxes_min} vs {args.number_of_boxes_max}."
            )



def make_env(args: argparse.Namespace) -> BoxMovingEnv:
    env_kwargs: dict[str, Any] = dict(
        grid_size=args.grid_size,
        episode_length=args.episode_length,
        number_of_boxes_min=args.number_of_boxes_min,
        number_of_boxes_max=args.number_of_boxes_max,
        number_of_moving_boxes_max=args.number_of_moving_boxes_max,
        level_generator=args.level_generator,
    )

    if args.quarter_size is not None:
        env_kwargs["quarter_size"] = args.quarter_size

    if args.level_generator == "variable":
        env_kwargs["generator_special"] = bool(args.generator_special)

    return BoxMovingEnv(**env_kwargs)



def _masked_select(mask: jnp.ndarray, new: jnp.ndarray, old: jnp.ndarray) -> jnp.ndarray:
    m = mask
    while m.ndim < new.ndim:
        m = m[..., None]
    return jnp.where(m, new, old)



def rollout_expert_batch_vmapped(
    env: BoxMovingEnv,
    seeds: list[int],
    fixed_length: int,
    planner_max_actions: int,
    max_boxes: int,
    max_move: int,
    reset_batch: Any,
    step_batch: Any,
) -> list[dict[str, Any]]:
    batch_size = len(seeds)
    keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds], axis=0)
    states, _ = reset_batch(keys)

    actions_padded, planned_lengths = solve_state_vmapped(
        states.grid,
        max_boxes=max_boxes,
        max_actions=planner_max_actions,
        max_move=max_move,
    )
    actions_padded = jnp.asarray(actions_padded, dtype=jnp.int8)
    planned_lengths = jnp.asarray(planned_lengths, dtype=jnp.int32)

    grids = np.zeros((batch_size, fixed_length, env.grid_size, env.grid_size), dtype=np.int8)
    next_grids = np.zeros_like(grids)
    actions = np.full((batch_size, fixed_length), -1, dtype=np.int8)
    rewards = np.zeros((batch_size, fixed_length), dtype=np.float32)
    dones = np.zeros((batch_size, fixed_length), dtype=np.bool_)
    truncs = np.zeros((batch_size, fixed_length), dtype=np.bool_)

    current = states
    active = jnp.ones((batch_size,), dtype=jnp.bool_)
    effective_steps = np.zeros((batch_size,), dtype=np.int32)

    for t in range(fixed_length):
        current_grid = current.grid
        grids[:, t] = np.asarray(current_grid, dtype=np.int8)

        valid = active & (t < planned_lengths)

        if bool(jnp.any(valid)):
            action_t = jnp.where(valid, actions_padded[:, t], jnp.zeros((batch_size,), dtype=jnp.int8))
            action_t = jnp.asarray(action_t, dtype=jnp.int8)
            nxt, reward_t, done_t, info_t = step_batch(current, action_t)
            done_t = jnp.asarray(done_t, dtype=jnp.bool_)
            trunc_t = jnp.asarray(info_t["truncated"], dtype=jnp.bool_)

            next_grid = jnp.where(valid[:, None, None], nxt.grid, current_grid)
            next_grids[:, t] = np.asarray(next_grid, dtype=np.int8)

            padded_action = jnp.full((batch_size,), jnp.int8(-1), dtype=jnp.int8)
            actions[:, t] = np.asarray(jnp.where(valid, action_t, padded_action), dtype=np.int8)
            rewards[:, t] = np.asarray(jnp.where(valid, reward_t, jnp.float32(0.0)), dtype=np.float32)
            inactive = jnp.logical_not(active)
            dones[:, t] = np.asarray(jnp.where(valid, done_t, inactive), dtype=np.bool_)
            truncs[:, t] = np.asarray(jnp.where(valid, trunc_t, jnp.zeros_like(trunc_t)), dtype=np.bool_)

            current = jax.tree_util.tree_map(
                lambda n, c: _masked_select(valid, n, c),
                nxt,
                current,
            )
            active = jnp.where(valid, ~done_t, active)
            effective_steps += np.asarray(valid, dtype=np.int32)
        else:
            next_grids[:, t] = np.asarray(current_grid, dtype=np.int8)
            dones[:, t] = np.asarray(jnp.logical_not(active), dtype=np.bool_)

    successes = np.asarray(current.success, dtype=np.bool_)
    planned_lengths_np = np.asarray(planned_lengths, dtype=np.int32)

    out: list[dict[str, Any]] = []
    for i in range(batch_size):
        dr = 0.0
        for r in rewards[i][::-1]:
            dr = dr * 0.99 + float(r)

        out.append(
            {
                "seed": int(seeds[i]),
                "observations": grids[i],
                "next_observations": next_grids[i],
                "actions": actions[i],
                "rewards": rewards[i],
                "dones": dones[i],
                "truncated": truncs[i],
                # Keep per-trajectory arrays same length by setting executed_steps=fixed_length.
                "planned_actions": int(planned_lengths_np[i]),
                "effective_steps": int(effective_steps[i]),
                "executed_steps": int(fixed_length),
                "success": bool(successes[i]),
                "discounted_return": float(dr),
            }
        )

    return out



def flatten_trajectories_for_transition_view(trajectories: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    obs = []
    next_obs = []
    actions = []
    rewards = []
    terminals = []
    masks = []
    trajectory_id = []
    step_id = []

    for tid, traj in enumerate(trajectories):
        t = int(traj["executed_steps"])
        if t == 0:
            continue

        tr_obs = traj["observations"].reshape(t, -1)
        tr_next = traj["next_observations"].reshape(t, -1)
        tr_actions = traj["actions"]
        tr_rewards = traj["rewards"]
        tr_dones = traj["dones"]
        tr_trunc = traj["truncated"]
        tr_terminals = np.logical_or(tr_dones, tr_trunc)

        obs.append(tr_obs)
        next_obs.append(tr_next)
        actions.append(tr_actions)
        rewards.append(tr_rewards)
        terminals.append(tr_terminals.astype(np.float32))
        masks.append((1.0 - tr_terminals.astype(np.float32)).astype(np.float32))
        trajectory_id.append(np.full((t,), tid, dtype=np.int32))
        step_id.append(np.arange(t, dtype=np.int32))

    if not obs:
        return {
            "observations": np.zeros((0, 0), dtype=np.int8),
            "next_observations": np.zeros((0, 0), dtype=np.int8),
            "actions": np.zeros((0,), dtype=np.int8),
            "rewards": np.zeros((0,), dtype=np.float32),
            "terminals": np.zeros((0,), dtype=np.float32),
            "masks": np.zeros((0,), dtype=np.float32),
            "trajectory_id": np.zeros((0,), dtype=np.int32),
            "step_id": np.zeros((0,), dtype=np.int32),
        }

    return {
        "observations": np.concatenate(obs, axis=0),
        "next_observations": np.concatenate(next_obs, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "rewards": np.concatenate(rewards, axis=0),
        "terminals": np.concatenate(terminals, axis=0),
        "masks": np.concatenate(masks, axis=0),
        "trajectory_id": np.concatenate(trajectory_id, axis=0),
        "step_id": np.concatenate(step_id, axis=0),
    }



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    env = make_env(args)
    fixed_length = args.fixed_length if args.fixed_length is not None else args.episode_length
    planner_horizon = min(args.planner_max_actions, fixed_length)
    max_boxes = max(16, args.number_of_boxes_max * 2, args.number_of_boxes_max + 2)
    max_move = max(args.grid_size, 16)

    verbose = not args.quiet

    if verbose:
        log("Starting expert dataset collection")
        log(
            "Config: "
            + json.dumps(
                {
                    "output_path": args.output_path,
                    "num_trajectories": args.num_trajectories,
                    "seed": args.seed,
                    "parallel_envs": args.parallel_envs,
                    "fixed_length": fixed_length,
                    "planner_max_actions": args.planner_max_actions,
                    "effective_planner_horizon": planner_horizon,
                    "grid_size": args.grid_size,
                    "episode_length": args.episode_length,
                    "boxes": {
                        "min": args.number_of_boxes_min,
                        "max": args.number_of_boxes_max,
                        "moving_max": args.number_of_moving_boxes_max,
                    },
                    "level_generator": args.level_generator,
                    "generator_special": args.generator_special,
                    "quarter_size": args.quarter_size,
                    "allow_failed_trajectories": args.allow_failed_trajectories,
                },
                indent=2,
            )
        )
        if args.planner_max_actions > fixed_length:
            log(
                "planner_max_actions is larger than fixed_length; "
                f"capping planner horizon to {planner_horizon}."
            )

    reset_batch = jax.vmap(env.reset, in_axes=(0,))
    step_batch = jax.jit(jax.vmap(env.step, in_axes=(0, 0)))

    # Warm up JIT caches for stable benchmark/collection timing.
    warm_t0 = time.perf_counter()
    warm_batch = max(1, min(args.parallel_envs, args.num_trajectories))
    warm_keys = jnp.stack([jax.random.PRNGKey(1_000_000 + i) for i in range(warm_batch)], axis=0)
    warm_state, _ = reset_batch(warm_keys)
    warm_actions, warm_lengths = solve_state_vmapped(
        warm_state.grid,
        max_boxes=max_boxes,
        max_actions=planner_horizon,
        max_move=max_move,
    )
    warm_a0 = jnp.where(warm_lengths > 0, warm_actions[:, 0], jnp.zeros((warm_batch,), dtype=jnp.int8))
    _ = step_batch(warm_state, jnp.asarray(warm_a0, dtype=jnp.int8))
    if verbose:
        log(f"JIT warmup finished in {time.perf_counter() - warm_t0:.3f}s (warm_batch={warm_batch})")

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(10 * args.num_trajectories, args.num_trajectories)
    if verbose:
        log(f"Max attempts set to {max_attempts}")

    trajectories: list[dict[str, Any]] = []
    attempted = 0
    skipped = 0
    batch_idx = 0

    start = time.perf_counter()

    while len(trajectories) < args.num_trajectories and attempted < max_attempts:
        batch_idx += 1
        batch_t0 = time.perf_counter()
        batch_size = min(args.parallel_envs, max_attempts - attempted)
        seeds = [args.seed + attempted + i for i in range(batch_size)]
        attempted += batch_size

        batch = rollout_expert_batch_vmapped(
            env=env,
            seeds=seeds,
            fixed_length=fixed_length,
            planner_max_actions=planner_horizon,
            max_boxes=max_boxes,
            max_move=max_move,
            reset_batch=reset_batch,
            step_batch=step_batch,
        )

        accepted_this_batch = 0
        skipped_this_batch = 0

        for traj in batch:
            if (not args.allow_failed_trajectories) and (not traj["success"]):
                skipped += 1
                skipped_this_batch += 1
                continue
            trajectories.append(traj)
            accepted_this_batch += 1
            if len(trajectories) >= args.num_trajectories:
                break

        if verbose and (batch_idx % args.log_every == 0 or len(trajectories) >= args.num_trajectories):
            batch_successes = sum(int(t["success"]) for t in batch)
            eff_mean = float(np.mean([t["effective_steps"] for t in batch])) if batch else 0.0
            log(
                f"Batch {batch_idx}: seeds={seeds[0]}..{seeds[-1]} "
                f"generated={len(batch)} accepted={accepted_this_batch} skipped={skipped_this_batch} "
                f"success={batch_successes}/{len(batch)} effective_steps_mean={eff_mean:.2f} "
                f"collected={len(trajectories)}/{args.num_trajectories} attempted={attempted}/{max_attempts} "
                f"batch_s={time.perf_counter() - batch_t0:.3f} total_s={time.perf_counter() - start:.3f}"
            )

    elapsed = time.perf_counter() - start

    if len(trajectories) < args.num_trajectories:
        raise RuntimeError(
            f"Collected only {len(trajectories)}/{args.num_trajectories} trajectories "
            f"after {attempted} attempts. Increase --max-attempts or use --allow-failed-trajectories."
        )

    transition_view = flatten_trajectories_for_transition_view(trajectories)

    summary = Summary(
        requested_trajectories=args.num_trajectories,
        collected_trajectories=len(trajectories),
        attempted_rollouts=attempted,
        skipped_rollouts=skipped,
        total_transitions=int(transition_view["actions"].shape[0]),
        success_rate_attempted=(len(trajectories) / attempted) if attempted > 0 else 0.0,
        collect_time_seconds=elapsed,
    )

    if verbose:
        log("Collection finished")

    dataset = {
        "config": vars(args),
        "summary": asdict(summary),
        "transition_view": transition_view,
        "trajectories": np.array(trajectories, dtype=object),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(dataset, dtype=object), allow_pickle=True)

    print("Saved dataset:", output_path)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
