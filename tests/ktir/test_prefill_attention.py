# SPDX-License-Identifier: Apache-2.0
"""Validate the generated prefill-attention Spyre KTIR against a NumPy reference.

The KTIR is *generated* from the Triton kernel source by ``scripts/gen_ktir.py``
(driver: ``kernels/prefill_attention/lower.py``). The Spyre lowering turns every
pointer arg into an ``index`` and names args positionally::

    func.func @_prefill_attention_kernel_spyre(
        %arg0: index,  // Q     [SEQ, H, D]  f16  (packed, single request)
        %arg1: index,  // K     [SEQ, H, D]  f16
        %arg2: index,  // V     [SEQ, H, D]  f16
        %arg3: index,  // MASK  [SEQ, SEQ]   f32  (additive: 0.0 / -inf)
        %arg4: index,  // Out   [SEQ, H, D]  f16
        %arg5: f32,    // sm_scale (natural; kernel uses tl.exp)
        %arg6..%arg13: i32,   // q/k/v/o bs+h strides
        %arg14: i32,   // num_q_heads
        %arg15: i32,   // num_kv_heads
        %arg16: i32,   // batch
        %arg17: i32,   // num_m_blocks
    )

Shape frozen into the KTIR (see lower.py): one request (SEQ=128), 4 query / 4 kv
heads, head_dim 64, fp16, causal. BLOCK_M=BLOCK_N=64, so num_m_blocks=2. The
causal mask is precomputed on the host and passed as the additive ``MASK``
tensor — no ``tl.arange`` in the kernel, so the KTIR is ``tt.make_range``-free
and executes on the ktir-cpu simulator.

The reference is NumPy causal SDPA (f32 accumulation), so this runs with only
ktir_cpu — no GPU / Spyre-Triton build.

Run:
    uv run --active python -m pytest tests/ktir/test_prefill_attention.py -m ktir_cpu -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter

MLIR_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "kernels" / "prefill_attention" / "spyre.ktir"
)

FUNC = "_prefill_attention_kernel_spyre"

# Concrete shapes baked into the generated KTIR (see lower.py). One request.
SEQ = 128
H = 4
KVH = 4
D = 64
BLOCK = 64
NUM_M_BLOCKS = SEQ // BLOCK  # 2

SM_SCALE = np.float32(1.0 / math.sqrt(D))
NEG_INF = np.float32(-1.0e9)


def causal_mask():
    """Additive [SEQ, SEQ] causal mask: 0.0 where key <= query, -inf otherwise."""
    row = np.arange(SEQ)[:, None]
    col = np.arange(SEQ)[None, :]
    return np.where(col <= row, np.float32(0.0), NEG_INF).astype(np.float32)


def numpy_reference(q, k, v, mask):
    """Causal SDPA in f32, matching the kernel's upcast-accumulate-downcast.

    q, k, v: [SEQ, H, D] f16 (MHA). mask: [SEQ, SEQ] additive f32.
    """
    qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
    out = np.zeros((SEQ, H, D), dtype=np.float32)
    for h in range(H):
        s = (qf[:, h, :] * float(SM_SCALE)) @ kf[:, h, :].T + mask
        s -= s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
        out[:, h, :] = w @ vf[:, h, :]
    return out.astype(np.float16)


def _run(q, k, v, mask):
    interp = KTIRInterpreter()
    interp.load(MLIR_PATH.read_text())

    out = np.zeros((SEQ, H, D), dtype=np.float16)
    q_bs, q_h = H * D, D          # contiguous [SEQ, H, D] strides (elements)
    outputs = interp.execute_function(
        FUNC,
        arg0=q, arg1=k, arg2=v, arg3=mask, arg4=out,
        arg5=SM_SCALE,
        arg6=np.int32(q_bs), arg7=np.int32(q_h),   # q strides
        arg8=np.int32(q_bs), arg9=np.int32(q_h),   # k
        arg10=np.int32(q_bs), arg11=np.int32(q_h),  # v
        arg12=np.int32(q_bs), arg13=np.int32(q_h),  # o
        arg14=np.int32(H),
        arg15=np.int32(KVH),
        arg16=np.int32(1),          # batch
        arg17=np.int32(NUM_M_BLOCKS),
    )
    return outputs["arg4"]


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir():
    """Generated prefill KTIR matches the causal-SDPA reference (f16 tol)."""
    rng = np.random.default_rng(42)
    q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    mask = causal_mask()

    result = _run(q, k, v, mask)
    expected = numpy_reference(q, k, v, mask)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-2, atol=1e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"PASS: max abs error = {max_err:.6f}")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_first_row():
    """Causal query 0 attends only to key/value 0: out[0, h] == v[0, h]."""
    rng = np.random.default_rng(7)
    q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)

    result = _run(q, k, v, causal_mask()).astype(np.float32)
    np.testing.assert_allclose(result[0], v[0].astype(np.float32), rtol=1e-2, atol=1e-2)
    print("PASS: causal first-row identity")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_uniform_v():
    """All V rows identical → every output row equals that V row."""
    rng = np.random.default_rng(11)
    q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    v_row = (rng.standard_normal((KVH, D)) * 0.1).astype(np.float16)
    v = np.broadcast_to(v_row, (SEQ, KVH, D)).copy()

    result = _run(q, k, v, causal_mask()).astype(np.float32)
    expected = np.broadcast_to(v_row.astype(np.float32), (SEQ, H, D))
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
    print("PASS: uniform V")
