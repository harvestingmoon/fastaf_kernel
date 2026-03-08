"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.compile import BuilderRegistry
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def run_benchmark_inprocess(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark in-process (avoids worker CUDA IPC path) and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5, use_isolated_runner=False)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    evaluator_cls = resolve_evaluator(definition)
    runnable = BuilderRegistry.get_instance().build(definition, solution)
    results = {definition.name: {}}

    for workload_trace in workloads:
        workload = workload_trace.workload
        try:
            baseline = evaluator_cls.build_baseline(
                definition=definition,
                workload=workload,
                cfg=config,
                device="cuda:0",
                trace_set_root=trace_set.root,
            )
            evaluation = evaluator_cls.evaluate(
                definition=definition,
                sol_runnable=runnable,
                inputs=baseline.inputs,
                ref_outputs=baseline.outputs,
                ref_mean_latency_ms=baseline.mean_latency_ms,
                cfg=config,
                log_path="",
                device="cuda:0",
            )
            entry = {
                "status": evaluation.status.value,
                "solution": solution.name,
            }
            if evaluation.performance:
                entry["latency_ms"] = evaluation.performance.latency_ms
                entry["reference_latency_ms"] = evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = evaluation.performance.speedup_factor
            if evaluation.correctness:
                entry["max_abs_error"] = evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = evaluation.correctness.max_relative_error
            if evaluation.log:
                entry["log"] = evaluation.log
        except Exception as exc:
            entry = {
                "status": "RUNTIME_ERROR",
                "solution": solution.name,
                "log": f"In-process exception: {type(exc).__name__}: {exc}",
            }

        results[definition.name][workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def main():
    """Pack solution and run benchmark."""
    os.environ.setdefault("FIB_INPROCESS_BENCH", "1")

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    use_inprocess = os.environ.get("FIB_INPROCESS_BENCH", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_inprocess:
        print("Benchmark mode: in-process (FIB_INPROCESS_BENCH=1)")
        results = run_benchmark_inprocess(solution)
    else:
        results = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()
