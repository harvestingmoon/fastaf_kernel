"""Direct test of the CUDA GDN kernel with reference comparison statistics."""
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_TEST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT_DIR / 'solution' / 'cuda'))
sys.path.insert(0, str(LOCAL_TEST_DIR))

from binding import kernel
from origina_gdn_prefill import run as reference_run


def sync_cuda(stage: str):
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[ERROR] CUDA sync failed at stage '{stage}': {type(e).__name__}: {e}")
            raise


def run_stage(stage: str, fn):
    print(f"[STAGE] {stage}")
    sync_cuda(f"before {stage}")
    t0 = time.time()
    result = fn()
    sync_cuda(f"after {stage}")
    elapsed = time.time() - t0
    print(f"[STAGE] {stage} done in {elapsed*1000:.2f} ms")
    return result, elapsed


def summarize_error(name: str, actual: torch.Tensor, expected: torch.Tensor, atol=1e-2, rtol=1e-2):
    actual_f = actual.float()
    expected_f = expected.float()

    diff = actual_f - expected_f
    abs_err = diff.abs()
    rel_err = abs_err / expected_f.abs().clamp_min(1e-8)

    abs_flat = abs_err.flatten()
    rel_flat = rel_err.flatten()

    stats = {
        'allclose': torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol),
        'max_abs': abs_flat.max().item(),
        'mean_abs': abs_flat.mean().item(),
        'rmse': torch.sqrt((diff * diff).mean()).item(),
        'p95_abs': torch.quantile(abs_flat, 0.95).item(),
        'p99_abs': torch.quantile(abs_flat, 0.99).item(),
        'max_rel': rel_flat.max().item(),
        'mean_rel': rel_flat.mean().item(),
        'nan_count': torch.isnan(actual_f).sum().item(),
        'inf_count': torch.isinf(actual_f).sum().item(),
    }

    print(f"\n{name} comparison stats:")
    print(f"  allclose(atol={atol}, rtol={rtol}): {stats['allclose']}")
    print(f"  max abs error:  {stats['max_abs']:.6e}")
    print(f"  mean abs error: {stats['mean_abs']:.6e}")
    print(f"  RMSE:           {stats['rmse']:.6e}")
    print(f"  p95 abs error:  {stats['p95_abs']:.6e}")
    print(f"  p99 abs error:  {stats['p99_abs']:.6e}")
    print(f"  max rel error:  {stats['max_rel']:.6e}")
    print(f"  mean rel error: {stats['mean_rel']:.6e}")
    print(f"  NaN count:      {stats['nan_count']}")
    print(f"  Inf count:      {stats['inf_count']}")

    return stats

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("ERROR: CUDA not available")
    sys.exit(1)

# Create test inputs matching the GDN prefill definition
# num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128
total_seq_len = 17
num_seqs = 2
num_q_heads = 4
num_k_heads = 4
num_v_heads = 8
head_size = 128

print(f"\nTest configuration:")
print(f"  Total seq len: {total_seq_len}")
print(f"  Num sequences: {num_seqs}")
print(f"  Q/K/V heads: {num_q_heads}/{num_k_heads}/{num_v_heads}")
print(f"  Head size: {head_size}")

# Create test tensors
print("\nCreating test tensors...")
q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=torch.bfloat16, device=device)
k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=torch.bfloat16, device=device)
v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=torch.bfloat16, device=device)
state = torch.randn(num_seqs, num_v_heads, head_size, head_size, dtype=torch.float32, device=device)
A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
a = torch.randn(total_seq_len, num_v_heads, dtype=torch.bfloat16, device=device)
dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
b = torch.randn(total_seq_len, num_v_heads, dtype=torch.bfloat16, device=device)

# Create cu_seqlens for 2 sequences (e.g., lengths 8 and 9)
cu_seqlens = torch.tensor([0, 8, 17], dtype=torch.int32, device=device)

scale = 1.0 / (head_size ** 0.5)

print("Running kernel...")
try:
    (output, new_state), elapsed_kernel = run_stage(
        "cuda_kernel",
        lambda: kernel(
            q.clone(),
            k.clone(),
            v.clone(),
            state.clone(),
            A_log.clone(),
            a.clone(),
            dt_bias.clone(),
            b.clone(),
            cu_seqlens.clone(),
            scale,
        ),
    )

    (ref_output, ref_new_state), elapsed_ref = run_stage(
        "reference_run",
        lambda: reference_run(
            q.clone(),
            k.clone(),
            v.clone(),
            state.clone(),
            A_log.clone(),
            a.clone(),
            dt_bias.clone(),
            b.clone(),
            cu_seqlens.clone(),
            scale,
        ),
    )
    
    print(f"\n✓ SUCCESS!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  New state shape: {new_state.shape}")
    print(f"  State dtype: {new_state.dtype}")
    print(f"  CUDA kernel time: {elapsed_kernel*1000:.2f} ms")
    print(f"  Reference time:   {elapsed_ref*1000:.2f} ms")
    
    # Basic sanity checks
    expected_output_shape = (total_seq_len, num_v_heads, head_size)
    expected_state_shape = (num_seqs, num_v_heads, head_size, head_size)
    
    assert output.shape == expected_output_shape, f"Wrong output shape: {output.shape} vs {expected_output_shape}"
    assert new_state.shape == expected_state_shape, f"Wrong state shape: {new_state.shape} vs {expected_state_shape}"

    out_nan = torch.isnan(output).sum().item()
    out_inf = torch.isinf(output).sum().item()
    state_nan = torch.isnan(new_state).sum().item()
    state_inf = torch.isinf(new_state).sum().item()
    if out_nan or out_inf or state_nan or state_inf:
        raise RuntimeError(
            f"Non-finite detected. output NaN/Inf={out_nan}/{out_inf}, "
            f"new_state NaN/Inf={state_nan}/{state_inf}"
        )

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isnan(new_state).any(), "State contains NaN"

    out_stats = summarize_error("Output", output, ref_output)
    state_stats = summarize_error("New state", new_state, ref_new_state)

    if out_stats['allclose'] and state_stats['allclose']:
        print("\n✓ Kernel matches reference within tolerance")
    else:
        print("\n⚠ Kernel does not fully match reference (see stats above)")

    print("\n✓ All sanity checks passed!")
    
except Exception as e:
    print(f"\n✗ ERROR ({type(e).__name__}): {e}")
    print("[DIAG] Attempting post-failure CUDA sync to surface async errors...")
    try:
        sync_cuda("post-failure")
    except Exception as sync_e:
        print(f"[DIAG] Post-failure sync error: {type(sync_e).__name__}: {sync_e}")

    print("[DIAG] Tensor metadata:")
    print(f"  q: shape={tuple(q.shape)}, dtype={q.dtype}, device={q.device}")
    print(f"  k: shape={tuple(k.shape)}, dtype={k.dtype}, device={k.device}")
    print(f"  v: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    print(f"  state: shape={tuple(state.shape)}, dtype={state.dtype}, device={state.device}")
    print(f"  A_log: shape={tuple(A_log.shape)}, dtype={A_log.dtype}, device={A_log.device}")
    print(f"  a: shape={tuple(a.shape)}, dtype={a.dtype}, device={a.device}")
    print(f"  dt_bias: shape={tuple(dt_bias.shape)}, dtype={dt_bias.dtype}, device={dt_bias.device}")
    print(f"  b: shape={tuple(b.shape)}, dtype={b.dtype}, device={b.device}")
    print(f"  cu_seqlens: shape={tuple(cu_seqlens.shape)}, dtype={cu_seqlens.dtype}, device={cu_seqlens.device}, values={cu_seqlens.tolist()}")

    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("CUDA kernel test completed successfully!")
print("="*60)
