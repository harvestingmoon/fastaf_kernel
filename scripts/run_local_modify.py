"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from types import SimpleNamespace

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def _ensure_flashinfer_bench_available() -> None:
    """Ensure this script runs under a Python env that has flashinfer_bench."""
    try:
        import flashinfer_bench  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    candidates = [
        os.environ.get("FI_BENCH_PYTHON"),
        "/home/direct11/anaconda3/envs/fi-bench/bin/python",
        str(Path.home() / "anaconda3" / "envs" / "fi-bench" / "bin" / "python"),
    ]

    script_path = str(Path(__file__).resolve())
    for candidate in candidates:
        if not candidate:
            continue
        if not Path(candidate).exists():
            continue
        if Path(candidate).resolve() == Path(sys.executable).resolve():
            continue

        try:
            subprocess.run(
                [candidate, "-c", "import flashinfer_bench"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue

        print(f"Relaunching with fi-bench interpreter: {candidate}")
        candidate_bin = str(Path(candidate).resolve().parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
        if candidate_bin not in path_parts:
            os.environ["PATH"] = candidate_bin + os.pathsep + os.environ.get("PATH", "")
        os.execv(candidate, [candidate, script_path, *sys.argv[1:]])

    raise ModuleNotFoundError(
        "flashinfer_bench is not installed in the current interpreter, and no usable "
        "fi-bench interpreter was found. Set FI_BENCH_PYTHON to your fi-bench python, "
        "for example:\n"
        "  export FI_BENCH_PYTHON=/home/<user>/anaconda3/envs/fi-bench/bin/python"
    )


_ensure_flashinfer_bench_available()

# Ensure active interpreter's bin/Scripts dir is in PATH for tools like ninja
python_bin = str(Path(sys.executable).resolve().parent)
path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
if python_bin not in path_parts:
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

# If ninja still isn't found but exists next to Python, expose it via PATH
if shutil.which("ninja") is None:
    ninja_candidate = Path(python_bin) / ("ninja.exe" if os.name == "nt" else "ninja")
    if ninja_candidate.exists():
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.compile import BuilderRegistry
from scripts.pack_solution import pack_solution


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        isolated_runner = _env_flag("FIB_USE_ISOLATED_RUNNER", False)
        fast_mode = _env_flag("FIB_FAST_MODE", False)

        warmup_runs = 1 if fast_mode else 3
        iterations = 10 if fast_mode else 100
        num_trials = 1 if fast_mode else 5

        config = BenchmarkConfig(
            warmup_runs=warmup_runs,
            iterations=iterations,
            num_trials=num_trials,
            use_isolated_runner=isolated_runner,
            profile_baseline=not fast_mode,
        )
        mode = "isolated" if isolated_runner else "single-process"
        print(f"Runner mode: {mode} (set FIB_USE_ISOLATED_RUNNER=1 to enable isolated)")
        if fast_mode:
            print("Fast mode enabled (FIB_FAST_MODE=1): warmup=1, iterations=10, trials=1, baseline profiling off")

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    max_workloads_env = os.environ.get("FIB_MAX_WORKLOADS")
    if max_workloads_env:
        try:
            max_workloads = int(max_workloads_env)
            if max_workloads > 0:
                workloads = workloads[:max_workloads]
                print(f"Limiting workloads to first {max_workloads} via FIB_MAX_WORKLOADS")
        except ValueError:
            print(f"Ignoring invalid FIB_MAX_WORKLOADS value: {max_workloads_env}")

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
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
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
    """Run benchmark fully in-process (no benchmark worker IPC) and return results."""
    if config is None:
        fast_mode = _env_flag("FIB_FAST_MODE", False)
        warmup_runs = 1 if fast_mode else 3
        iterations = 10 if fast_mode else 100
        num_trials = 1 if fast_mode else 5

        config = BenchmarkConfig(
            warmup_runs=warmup_runs,
            iterations=iterations,
            num_trials=num_trials,
            use_isolated_runner=False,
            profile_baseline=not fast_mode,
        )
        if fast_mode:
            print("Fast mode enabled (FIB_FAST_MODE=1): warmup=1, iterations=10, trials=1, baseline profiling off")

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    max_workloads_env = os.environ.get("FIB_MAX_WORKLOADS")
    if max_workloads_env:
        try:
            max_workloads = int(max_workloads_env)
            if max_workloads > 0:
                workloads = workloads[:max_workloads]
                print(f"Limiting workloads to first {max_workloads} via FIB_MAX_WORKLOADS")
        except ValueError:
            print(f"Ignoring invalid FIB_MAX_WORKLOADS value: {max_workloads_env}")

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    evaluator_cls = resolve_evaluator(definition)

    use_cute_dsl = _env_flag("FIB_USE_CUTE_DSL", False)
    if use_cute_dsl:
        if definition.name != "gdn_prefill_qk4_v8_d128_k_last":
            raise ValueError(
                "FIB_USE_CUTE_DSL=1 is currently supported only for "
                "gdn_prefill_qk4_v8_d128_k_last"
            )

        from solution.cute_dsl import gdn_decode as cute_gdn

        if not getattr(cute_gdn, "HAS_CUTE_KERNEL_PRIMS", False):
            raise RuntimeError(
                "FIB_USE_CUTE_DSL=1 requested, but this CUTLASS installation lacks required "
                "CuTe kernel primitives. Use optimized CUDA path (FIB_USE_CUTE_DSL=0) for speed, "
                "or install a full CuTe kernel-capable build."
            )

        gated_delta_net_prefill = cute_gdn.gated_delta_net_prefill

        class _CuteRunnable:
            def __init__(self, fn):
                self._fn = fn
                self.metadata = SimpleNamespace(destination_passing_style=False)

            def __call__(self, *args):
                return self._fn(*args)

        runnable = _CuteRunnable(gated_delta_net_prefill)
        print("Using CuTe DSL runnable (FIB_USE_CUTE_DSL=1)")
    else:
        runnable = BuilderRegistry.get_instance().build(definition, solution)

    results = {definition.name: {}}

    for workload_trace in workloads:
        workload = workload_trace.workload
        print(f"Evaluating workload {workload.uuid} (in-process)...")

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
        except Exception as exc:
            results[definition.name][workload.uuid] = {
                "status": "RUNTIME_ERROR",
                "solution": solution.name,
                "log": f"In-process exception: {type(exc).__name__}: {exc}",
            }
            continue

        entry = {
            "status": evaluation.status.value,
            "solution": solution.name,
        }
        if evaluation.log:
            entry["log"] = evaluation.log
        if evaluation.performance:
            entry["latency_ms"] = evaluation.performance.latency_ms
            entry["reference_latency_ms"] = evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = evaluation.performance.speedup_factor
        if evaluation.correctness:
            entry["max_abs_error"] = evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = evaluation.correctness.max_relative_error
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

            if status and status.upper() != "PASSED":
                log_path = result.get("log")
                if log_path:
                    print(f"    log: {log_path}")


def _contains_invalid_resource_handle(results: dict) -> bool:
    """Return True if benchmark results include CUDA invalid resource handle errors."""
    for _, traces in results.items():
        for _, result in traces.items():
            log = str(result.get("log", "")).lower()
            status = str(result.get("status", "")).upper()
            if "invalid resource handle" in log:
                return True
            if status == "RUNTIME_ERROR" and "cuda error" in log and "resource handle" in log:
                return True
    return False


def main():
    """Pack solution and run benchmark."""
    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    use_cute_dsl = _env_flag("FIB_USE_CUTE_DSL", False)
    prefer_inprocess = _env_flag("FIB_PREFER_INPROCESS", True)

    if use_cute_dsl and not prefer_inprocess:
        print("FIB_USE_CUTE_DSL=1 requires in-process mode; overriding FIB_PREFER_INPROCESS=1")
        prefer_inprocess = True

    if prefer_inprocess:
        print("Benchmark mode: in-process (set FIB_PREFER_INPROCESS=0 to use runner mode)")
        results = run_benchmark_inprocess(solution)
    else:
        results = run_benchmark(solution)

    auto_fallback = _env_flag("FIB_AUTO_INPROCESS_FALLBACK", True)
    if (not prefer_inprocess) and auto_fallback and _contains_invalid_resource_handle(results):
        print("Detected CUDA IPC invalid resource handle in benchmark runner; retrying in-process benchmark...")
        results = run_benchmark_inprocess(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()
