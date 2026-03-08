"""Replay one FlashInfer-Bench workload in a single process to expose exact exceptions.

Usage:
  python scripts/replay_first_workload.py
  python scripts/replay_first_workload.py --index 0
  python scripts/replay_first_workload.py --uuid <workload_uuid>
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.compile import BuilderRegistry
from scripts.pack_solution import pack_solution


def _tensor_info(name: str, value):
    if isinstance(value, torch.Tensor):
        print(
            f"  {name}: shape={tuple(value.shape)} dtype={value.dtype} "
            f"device={value.device} contiguous={value.is_contiguous()}"
        )
    else:
        print(f"  {name}: type={type(value).__name__} value={value}")


def _select_workload(workloads, index: int | None, uuid: str | None):
    if uuid is not None:
        for w in workloads:
            if w.workload.uuid == uuid:
                return w
        raise ValueError(f"Workload uuid not found: {uuid}")
    if index is None:
        index = 0
    if index < 0 or index >= len(workloads):
        raise IndexError(f"Workload index out of range: {index} (len={len(workloads)})")
    return workloads[index]


def main():
    parser = argparse.ArgumentParser(description="Replay one workload in-process to get full traceback")
    parser.add_argument("--index", type=int, default=0, help="Workload index (default: 0)")
    parser.add_argument("--uuid", type=str, default=None, help="Workload UUID (overrides --index)")
    args = parser.parse_args()

    dataset_path = os.environ.get("FIB_DATASET_PATH")
    if not dataset_path:
        raise EnvironmentError(
            "FIB_DATASET_PATH is not set. Example:\n"
            "  export FIB_DATASET_PATH=/mnt/d/Programming/cuda/flashinfer-bench-starter-kit/mlsys26-contest"
        )

    print("Packing solution...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    print(f"Loaded solution: {solution.name}")
    print(f"  definition={solution.definition}")
    print(f"  entry_point={solution.spec.entry_point}")
    print(f"  binding={solution.spec.binding}")
    print(f"  dps={solution.spec.destination_passing_style}")

    trace_set = TraceSet.from_path(dataset_path)
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise RuntimeError(f"No workloads for definition: {solution.definition}")

    wl_trace = _select_workload(workloads, args.index, args.uuid)
    workload = wl_trace.workload
    print(f"Selected workload: uuid={workload.uuid}")

    cfg = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1, use_isolated_runner=True)
    evaluator_cls = resolve_evaluator(definition)

    print("Building baseline inputs/refs...")
    baseline = evaluator_cls.build_baseline(
        definition=definition,
        workload=workload,
        cfg=cfg,
        device="cuda:0",
        trace_set_root=trace_set.root,
    )

    inp = baseline.inputs[0]
    ref_out = baseline.outputs[0]

    print("Input summary (trial 0):")
    for i, x in enumerate(inp):
        _tensor_info(f"inp[{i}]", x)

    print("Ref output summary (trial 0):")
    for i, x in enumerate(ref_out):
        _tensor_info(f"ref_out[{i}]", x)

    print("Building runnable...")
    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    print("Running solution directly (single-process replay)...")
    try:
        if runnable.metadata.destination_passing_style:
            out = [torch.empty_like(t) for t in ref_out]
            with torch.no_grad():
                runnable(*inp, *out)
            torch.cuda.synchronize("cuda:0")
        else:
            with torch.no_grad():
                result = runnable(*inp)
            torch.cuda.synchronize("cuda:0")
            if isinstance(result, tuple):
                out = list(result)
            else:
                out = [result]

        print("Replay succeeded. Output summary:")
        for i, x in enumerate(out):
            _tensor_info(f"out[{i}]", x)

        for i, (o, r) in enumerate(zip(out, ref_out)):
            o_f = o.float()
            r_f = r.float()
            abs_err = (o_f - r_f).abs()
            rel_err = abs_err / r_f.abs().clamp_min(1e-8)
            print(
                f"  compare[{i}]: max_abs={abs_err.max().item():.6e}, "
                f"mean_abs={abs_err.mean().item():.6e}, "
                f"max_rel={rel_err.max().item():.6e}, "
                f"allclose={torch.allclose(o_f, r_f, atol=1e-2, rtol=1e-2)}"
            )

    except Exception as exc:
        print("\n[REPLAY ERROR] Exact exception below:")
        print(f"Type: {type(exc).__name__}")
        print(f"Message: {exc}")
        print("\nTraceback:")
        traceback.print_exc()

        print("\nCUDA diagnostics after failure:")
        try:
            torch.cuda.synchronize("cuda:0")
            print("  cuda synchronize succeeded")
        except Exception as sync_exc:
            print(f"  cuda synchronize error: {type(sync_exc).__name__}: {sync_exc}")

        raise


if __name__ == "__main__":
    main()
