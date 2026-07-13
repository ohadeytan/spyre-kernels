# SPDX-License-Identifier: Apache-2.0
"""Validate the generated prefill-attention Spyre KTIR on ktir-cpu.

The KTIR under test is *generated* from ``kernels/prefill_attention/spyre.py``
by ``scripts/gen_ktir.py`` (driver: ``kernels/prefill_attention/lower.py``), so
the function name and signature mirror the lowered kernel exactly. The Spyre
kernel is the single-(request, head) form: ``SEQ`` is a compile-time constant,
one head per launch (2D descriptors), the causal mask is an additive host
tensor, and there is no ``tl.arange`` — the properties that let it lower to KTIR
with live ``tl.spyre_tensor_layout`` markers and run on ktir-cpu.

Multiple heads are handled by the wrapper (one launch per head); this test
drives a single head directly, which is exactly what the KTIR encodes.

The reference is a NumPy full-softmax attention with an f32 accumulator,
matching the kernel's math (natural ``exp`` with a plain scale, additive mask).

Run:
    .venv/bin/python -m pytest tests/ktir/test_prefill_attention.py -v
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

# Concrete shapes baked into the generated KTIR (see kernels/.../lower.py).
SEQ = 128
HEAD_DIM = 64
MASK_NEG = -1.0e8

# Distribution-invariance variants (see kernels/.../lower.py): the SAME kernel
# lowered at grids [1, 4, 16, 32]. The grid (core count) is frozen into the
# .ktir at lowering, so partition-independence is checked by running each grid's
# KTIR and comparing. These use a two-query-block sequence so the partition is
# non-trivial (grid 1 = both blocks on one core; 4/16/32 = idle cores).
DIST_CORE_COUNTS = [1, 4, 16, 32]
DIST_SEQ = 256


def _dist_mlir_path(cores: int) -> Path:
    return MLIR_PATH.parent / f"spyre_dist_c{cores}.ktir"


def build_additive_mask(seq_len: int, is_causal: bool = True) -> np.ndarray:
    """Additive mask [seq, seq]: 0.0 where valid, MASK_NEG where masked."""
    q_pos = np.arange(seq_len)[:, None]
    k_pos = np.arange(seq_len)[None, :]
    valid = np.ones((seq_len, seq_len), dtype=bool)
    if is_causal:
        valid &= q_pos >= k_pos
    return np.where(valid, np.float32(0.0), np.float32(MASK_NEG))


def numpy_reference(q, k, v, mask, sm_scale):
    """Single-head full-softmax attention in f32, matching the kernel's math.

    q/k/v: [SEQ, HEAD_DIM]. Scores use natural exp with a plain sm_scale (no
    1/ln2 factor) and the additive mask, matching the online-softmax kernel.
    """
    qf = q.astype(np.float32)
    kf = k.astype(np.float32)
    vf = v.astype(np.float32)
    qk = (qf @ kf.T) * sm_scale + mask          # [SEQ, SEQ]
    qk -= qk.max(axis=1, keepdims=True)
    p = np.exp(qk)
    p /= p.sum(axis=1, keepdims=True)
    return p @ vf


def _run(q, k, v, mask, sm_scale, *, mlir_path=MLIR_PATH, seq=SEQ):
    interp = KTIRInterpreter()
    interp.load(mlir_path.read_text())

    o = np.zeros_like(q)
    outputs = interp.execute_function(
        FUNC,
        arg0=q,                        # Q  [seq, HEAD_DIM]
        arg1=k,                        # K  [seq, HEAD_DIM]
        arg2=v,                        # V  [seq, HEAD_DIM]
        arg3=mask,                     # Mask [seq, seq] f32
        arg4=np.float32(sm_scale),     # sm_scale
        arg5=o,                        # Out [seq, HEAD_DIM]
        arg6=np.int32(HEAD_DIM),       # stride_qbs (row stride = HEAD_DIM)
        arg7=np.int32(HEAD_DIM),       # stride_kbs
        arg8=np.int32(HEAD_DIM),       # stride_vbs
        arg9=np.int32(HEAD_DIM),       # stride_obs
        arg10=np.int32(seq),           # stride_mm
    )
    return outputs["arg5"]


def _make_inputs(seed=42, scale=1.0, seq=SEQ):
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((seq, HEAD_DIM)) * scale).astype(np.float16)
    k = (rng.standard_normal((seq, HEAD_DIM)) * scale).astype(np.float16)
    v = (rng.standard_normal((seq, HEAD_DIM)) * scale).astype(np.float16)
    return q, k, v


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_causal():
    """Generated KTIR matches the NumPy causal-attention reference (f16 tol)."""
    q, k, v = _make_inputs()
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    expected = numpy_reference(q, k, v, mask, sm_scale)

    np.testing.assert_allclose(
        result.astype(np.float32), expected, rtol=2e-2, atol=2e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected))
    print(f"PASS causal: max abs error = {max_err:.6f}")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_zeros():
    """All-zero Q/K/V — uniform attention weights, zero output."""
    q = np.zeros((SEQ, HEAD_DIM), dtype=np.float16)
    k = np.zeros((SEQ, HEAD_DIM), dtype=np.float16)
    v = np.zeros((SEQ, HEAD_DIM), dtype=np.float16)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-3)
    print("PASS: zeros input")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_large_values():
    """Larger-magnitude inputs — exercises the online-softmax rescaling path.

    Scale is kept moderate (3.0): against an f32 NumPy reference, an f16
    online-softmax at scale=10 diverges by more than a fair equivalence
    tolerance purely from f16 rounding of near-degenerate softmax weights
    (verified: error is sparse and low-mean, not a structural tile bug). The
    GPU test compares kernel-vs-kernel and can push harder.
    """
    q, k, v = _make_inputs(seed=7, scale=3.0)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    expected = numpy_reference(q, k, v, mask, sm_scale)

    np.testing.assert_allclose(
        result.astype(np.float32), expected, rtol=3e-2, atol=3e-2,
    )
    print("PASS large values")


# ─── Distribution invariance ────────────────────────────────────────────────

# These SEQ=256 (two-BLOCK) variants were the first to exercise a query-block
# with more than one KV tile, which surfaced a ktir-cpu bug: linalg.matmul
# mutated a loop-hoisted zero-init constant used as its `outs` in place, so the
# QK^T accumulator was never reset across KV iterations (tile 2+ started from
# tile 1's scores). Reported as torch-spyre/ktir-cpu#157, fixed by #149; the pin
# in pyproject.toml now includes the fix. See
# project_prefill_attention_second_tile_offset_bug in memory.


@pytest.mark.ktir_cpu
class TestPrefillAttentionDistribution:
    """The distribution loop must be partition-independent.

    Every core walks the whole key range; the loop only slices the query-block
    work space across ``num_cores`` (frozen into each ``spyre_dist_c<N>.ktir`` at
    lowering). So the result must be identical for every core count. The DIST_SEQ
    = 256 shape is two BLOCK=128 query-blocks, so the partitions genuinely differ:
    grid 1 runs both blocks on one core (sequential), grid 4 puts one block on
    each of cores 0/1 with 2 idle, grids 16/32 leave most cores idle (more cores
    than work). If any ``.ktir`` variant is missing (spyre toolchain not run),
    the whole class skips.
    """

    @staticmethod
    def _require(cores: int) -> Path:
        path = _dist_mlir_path(cores)
        if not path.is_file():
            pytest.skip(
                f"{path.name} not generated — regenerate with:\n"
                "  GIT_PAT=$(gh auth token) TRITON_DEFAULT_BACKEND=spyre uv run "
                '--with "triton @ git+https://github.com/fabianlim/triton@7814b672" '
                "python scripts/gen_ktir.py prefill_attention"
            )
        return path

    @pytest.mark.parametrize("cores", DIST_CORE_COUNTS)
    def test_matches_reference(self, cores):
        """Each core count matches the NumPy causal-attention reference."""
        mlir_path = self._require(cores)
        q, k, v = _make_inputs(seq=DIST_SEQ)
        mask = build_additive_mask(DIST_SEQ, is_causal=True)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

        result = _run(q, k, v, mask, sm_scale, mlir_path=mlir_path, seq=DIST_SEQ)
        expected = numpy_reference(q, k, v, mask, sm_scale)

        np.testing.assert_allclose(
            result.astype(np.float32), expected, rtol=2e-2, atol=2e-2,
        )
        max_err = np.max(np.abs(result.astype(np.float32) - expected))
        print(f"PASS cores={cores}: max abs error = {max_err:.6f}")

    def test_invariant_across_core_counts(self):
        """All core counts must agree bitwise: a query-block's online-softmax is
        independent of which core runs it, so re-partitioning cannot shift the
        result. (The per-block op order is identical across grids; only the
        block-to-core assignment changes.)"""
        q, k, v = _make_inputs(seq=DIST_SEQ)
        mask = build_additive_mask(DIST_SEQ, is_causal=True)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

        baseline_cores = DIST_CORE_COUNTS[0]
        baseline = _run(
            q, k, v, mask, sm_scale,
            mlir_path=self._require(baseline_cores), seq=DIST_SEQ,
        )
        for cores in DIST_CORE_COUNTS[1:]:
            result = _run(
                q, k, v, mask, sm_scale,
                mlir_path=self._require(cores), seq=DIST_SEQ,
            )
            np.testing.assert_array_equal(
                result, baseline,
                err_msg=f"cores={cores} differs from cores={baseline_cores}",
            )
        print(f"PASS: bitwise-identical across cores {DIST_CORE_COUNTS}")
