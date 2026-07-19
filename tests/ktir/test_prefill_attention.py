# SPDX-License-Identifier: Apache-2.0
"""Validate the generated prefill-attention Spyre KTIR on ktir-cpu.

The KTIR under test is *generated* from ``kernels/prefill_attention/spyre.py``
by ``scripts/gen_ktir.py`` (driver: ``kernels/prefill_attention/lower.py``), so
the function name and signature mirror the lowered kernel exactly. The Spyre
kernel is the single-request, **all-heads** form: ``SEQ`` is a compile-time
constant, the head is the *batch* axis of 3D ``[HEADS, SEQ, HEAD_DIM]``
descriptors, the causal mask is an additive host tensor, and there is no
``tl.arange`` — the properties that let it lower to KTIR with live
``tl.spyre_tensor_layout`` markers and run on ktir-cpu.

Every head is handled in **one launch**: ``tl.dot`` over the trailing two dims
of the 3D operands lowers to ``linalg.batch_matmul`` (torch-spyre/triton PR #19,
``dispatchBatchMatmul``). This test drives all heads at once, exactly what the
KTIR encodes.

The reference is a NumPy full-softmax attention with an f32 accumulator, applied
per head and stacked, matching the kernel's math (natural ``exp`` with a plain
scale, additive mask; the mask is head-independent).

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
GQA_MLIR_PATH = MLIR_PATH.parent / "spyre_gqa.ktir"
PAD_MLIR_PATH = MLIR_PATH.parent / "spyre_pad.ktir"
BATCH_MLIR_PATH = MLIR_PATH.parent / "spyre_batch.ktir"

FUNC = "_prefill_attention_kernel_spyre"

# Concrete shapes baked into the generated KTIR (see kernels/.../lower.py).
HEADS = 4
SEQ = 128
HEAD_DIM = 64
MASK_NEG = -1.0e8

# GQA variant (spyre_gqa): 4 query heads share 2 KV heads (group size 2).
GQA_KV_HEADS = 2
# Head-dim padding variant (spyre_pad): logical head dim 48, padded to one stick.
PAD_LK = 48
PAD_DMODEL = 64
# Fixed-batch variant (spyre_batch): BATCH requests folded into the head axis as
# BATCH*HEADS "heads" (spyre_batch bakes HEADS = BATCH_N * HEADS = 8).
BATCH_N = 2

# Distribution-invariance variants (see kernels/.../lower.py): the SAME kernel
# lowered at grids [1, 4, 16, 32]. The grid (core count) is frozen into the
# .ktir at lowering, so partition-independence is checked by running each grid's
# KTIR and comparing. These use a two-query-block sequence (and HEADS heads) so
# the (head, query-block) work space is non-trivial to partition.
DIST_CORE_COUNTS = [1, 4, 16, 32]
DIST_SEQ = 256


def _dist_mlir_path(cores: int) -> Path:
    return MLIR_PATH.parent / f"spyre_dist_c{cores}.ktir"


def build_additive_mask(
    seq_len: int,
    is_causal: bool = True,
    sliding_window_q: int = 0,
    sliding_window_k: int = 0,
) -> np.ndarray:
    """Additive mask [seq, seq]: 0.0 where valid, MASK_NEG where masked.

    Encodes every masking mode the original ``_fwd_kernel`` applied per-tile,
    but precomputed on the host (the marked kernel has no ``tl.arange`` to build
    positions in-kernel). All conditions are AND-combined, matching the original:

    - ``is_causal``: a query attends only to keys at or before its position
      (``q_pos >= k_pos``).
    - ``sliding_window_q > 0``: bounds how far *back* a query may attend
      (``q_pos - k_pos <= sliding_window_q``) — the backward window.
    - ``sliding_window_k > 0``: bounds how far *forward* a query may attend
      (``k_pos - q_pos <= sliding_window_k``) — the forward window.

    A window of 0 disables that bound (matches the original's ``> 0`` guard).
    """
    q_pos = np.arange(seq_len)[:, None]
    k_pos = np.arange(seq_len)[None, :]
    valid = np.ones((seq_len, seq_len), dtype=bool)
    if is_causal:
        valid &= q_pos >= k_pos
    if sliding_window_q > 0:
        valid &= (q_pos - k_pos) <= sliding_window_q
    if sliding_window_k > 0:
        valid &= (k_pos - q_pos) <= sliding_window_k
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


def numpy_reference_batched(q, k, v, mask, sm_scale):
    """All-heads reference: apply numpy_reference per head and stack.

    q/k/v/out: [HEADS, SEQ, HEAD_DIM]. The mask is head-independent [SEQ, SEQ].
    Matches the kernel, where the head is the batch axis of the batched matmul.
    """
    return np.stack(
        [numpy_reference(q[h], k[h], v[h], mask, sm_scale) for h in range(q.shape[0])]
    )


def numpy_reference_gqa(q, k, v, mask, sm_scale):
    """GQA reference: query head h attends KV head h // (HEADS // KV_HEADS).

    q: [HEADS, SEQ, D]; k/v: [KV_HEADS, SEQ, D] with KV_HEADS <= HEADS and
    HEADS % KV_HEADS == 0. Mirrors the kernel's ``kv_head = head // kv_group_num``
    sharing, so it is the oracle for a num_q_heads != num_kv_heads launch.
    """
    n_q = q.shape[0]
    n_kv = k.shape[0]
    group = n_q // n_kv
    return np.stack(
        [numpy_reference(q[h], k[h // group], v[h // group], mask, sm_scale)
         for h in range(n_q)]
    )


def _run(q, k, v, mask, sm_scale, *, mlir_path=MLIR_PATH, seq=SEQ):
    """Execute the generated KTIR. q/k/v/o are 3D [heads, seq, dmodel]; the mask
    is 2D [seq, seq] (head-independent). Constexprs (HEADS, KV_HEADS, SEQ, BLOCK,
    Lk, DMODEL, …) are baked into the KTIR — only pointers, sm_scale, and the 9
    i32 strides are passed at runtime (arg order verified against the signature).

    Strides are derived from each array's own PHYSICAL shape, so this handles all
    variants uniformly: GQA (K/V carry fewer heads than Q) and head-dim padding
    (the physical last dim = DMODEL, so the per-token row stride grows) just fall
    out of the input shapes the caller passes."""
    interp = KTIRInterpreter()
    interp.load(mlir_path.read_text())

    def strides(arr):
        # (per-head stride, per-token/row stride) in elements, from the physical
        # shape [heads, seq, dmodel].
        _, s, d = arr.shape
        return np.int32(s * d), np.int32(d)

    qh, qbs = strides(q)
    kh, kbs = strides(k)
    vh, vbs = strides(v)
    o = np.zeros_like(q)
    oh, obs = strides(o)
    outputs = interp.execute_function(
        FUNC,
        arg0=q,                        # Q    [HEADS, seq, DMODEL]
        arg1=k,                        # K    [KV_HEADS, seq, DMODEL]
        arg2=v,                        # V    [KV_HEADS, seq, DMODEL]
        arg3=mask,                     # Mask [seq, seq] f32
        arg4=np.float32(sm_scale),     # sm_scale
        arg5=o,                        # Out  [HEADS, seq, DMODEL]
        arg6=qh,                       # stride_qh
        arg7=qbs,                      # stride_qbs
        arg8=kh,                       # stride_kh
        arg9=kbs,                      # stride_kbs
        arg10=vh,                      # stride_vh
        arg11=vbs,                     # stride_vbs
        arg12=oh,                      # stride_oh
        arg13=obs,                     # stride_obs
        arg14=np.int32(seq),           # stride_mm (mask row stride = SEQ_KV = seq)
    )
    return outputs["arg5"]


def _make_inputs(seed=42, scale=1.0, seq=SEQ):
    rng = np.random.default_rng(seed)
    shape = (HEADS, seq, HEAD_DIM)
    q = (rng.standard_normal(shape) * scale).astype(np.float16)
    k = (rng.standard_normal(shape) * scale).astype(np.float16)
    v = (rng.standard_normal(shape) * scale).astype(np.float16)
    return q, k, v


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_batch_matmul_present():
    """The multi-head kernel must lower the head axis to linalg.batch_matmul
    (PR #19 dispatchBatchMatmul), not a plain per-head linalg.matmul, and no
    tt.dot / tt.spyre_tensor_layout may survive."""
    text = MLIR_PATH.read_text()
    assert "linalg.batch_matmul" in text, "QK^T / P·V did not batch over heads"
    assert "tt.dot" not in text, "tt.dot survived lowering"
    assert "spyre_tensor_layout" not in text, "layout marker not consumed by the pass"


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_causal():
    """Generated KTIR matches the NumPy causal-attention reference (f16 tol).

    All HEADS heads run in one launch; the reference is applied per head and
    stacked. Also asserts each head's slice matches the single-head reference,
    so the batched matmul is not mixing heads.
    """
    q, k, v = _make_inputs()
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    expected = numpy_reference_batched(q, k, v, mask, sm_scale)

    assert result.shape == (HEADS, SEQ, HEAD_DIM)
    np.testing.assert_allclose(
        result.astype(np.float32), expected, rtol=2e-2, atol=2e-2,
    )
    # Per-head equivalence: head h of the batched output equals a standalone
    # single-head attention on head h (no cross-head contamination).
    for h in range(HEADS):
        np.testing.assert_allclose(
            result[h].astype(np.float32),
            numpy_reference(q[h], k[h], v[h], mask, sm_scale),
            rtol=2e-2, atol=2e-2,
        )
    max_err = np.max(np.abs(result.astype(np.float32) - expected))
    print(f"PASS causal ({HEADS} heads): max abs error = {max_err:.6f}")


@pytest.mark.ktir_cpu
@pytest.mark.parametrize(
    "window_q, window_k",
    [
        (32, 0),    # causal + backward window: attend only the last 32 keys
        (0, 16),    # forward window only (non-causal): keys up to 16 ahead
        (32, 16),   # bidirectional band around the diagonal
    ],
)
def test_prefill_attention_ktir_sliding_window(window_q, window_k):
    """Sliding-window attention is a pure host-mask change — the same generated
    KTIR handles it with no kernel modification. This reclaims the original
    ``_fwd_kernel``'s SLIDING_WINDOW_Q / SLIDING_WINDOW_K capability, which the
    marked kernel folds entirely into the additive host mask (a window is just a
    different set of MASK_NEG entries). Causal is enabled iff a backward window
    is set, mirroring the original's typical causal+window use.
    """
    q, k, v = _make_inputs()
    mask = build_additive_mask(
        SEQ,
        is_causal=(window_q > 0),
        sliding_window_q=window_q,
        sliding_window_k=window_k,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    expected = numpy_reference_batched(q, k, v, mask, sm_scale)

    np.testing.assert_allclose(
        result.astype(np.float32), expected, rtol=2e-2, atol=2e-2,
    )
    max_err = np.max(np.abs(result.astype(np.float32) - expected))
    print(f"PASS sliding-window q={window_q} k={window_k}: max abs error = {max_err:.6f}")
    shape = (HEADS, SEQ, HEAD_DIM)
    q = np.zeros(shape, dtype=np.float16)
    k = np.zeros(shape, dtype=np.float16)
    v = np.zeros(shape, dtype=np.float16)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-3)
    print("PASS: zeros input")


def _require_variant(path: Path):
    """Skip (not fail) if an optional variant's .ktir was not generated."""
    if not path.is_file():
        pytest.skip(
            f"{path.name} not generated — regenerate with:\n"
            "  GIT_PAT=$(gh auth token) TRITON_DEFAULT_BACKEND=spyre uv run "
            '--with "triton @ git+https://github.com/fabianlim/triton@a6938df5" '
            "python scripts/gen_ktir.py prefill_attention"
        )


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_gqa():
    """Grouped-query attention: 4 query heads share 2 KV heads, one launch.

    K/V carry KV_HEADS on the batch axis; the kernel reads KV head
    ``head // (HEADS // KV_HEADS)``. This is a num_q_heads != num_kv_heads batched
    matmul — head-sharing at the descriptor load index, not the matmul batch dim.
    The reference expands each KV head across its query group. Also checks that a
    query head matches a *standalone* attention on its shared KV head (the group
    mapping is exactly right, not just self-consistent).
    """
    _require_variant(GQA_MLIR_PATH)
    rng = np.random.default_rng(42)
    q = rng.standard_normal((HEADS, SEQ, HEAD_DIM)).astype(np.float16)
    k = rng.standard_normal((GQA_KV_HEADS, SEQ, HEAD_DIM)).astype(np.float16)
    v = rng.standard_normal((GQA_KV_HEADS, SEQ, HEAD_DIM)).astype(np.float16)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale, mlir_path=GQA_MLIR_PATH)
    expected = numpy_reference_gqa(q, k, v, mask, sm_scale)

    assert result.shape == (HEADS, SEQ, HEAD_DIM)
    np.testing.assert_allclose(
        result.astype(np.float32), expected, rtol=2e-2, atol=2e-2,
    )
    group = HEADS // GQA_KV_HEADS
    for h in range(HEADS):
        np.testing.assert_allclose(
            result[h].astype(np.float32),
            numpy_reference(q[h], k[h // group], v[h // group], mask, sm_scale),
            rtol=2e-2, atol=2e-2,
        )
    max_err = np.max(np.abs(result.astype(np.float32) - expected))
    print(f"PASS GQA ({HEADS}q/{GQA_KV_HEADS}kv heads): max abs error = {max_err:.6f}")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_head_dim_padding():
    """Padded head dim: logical Lk=48 physically padded to one stick (DMODEL=64).

    The marked kernel stick-tiles the padded DMODEL; the descriptor's logical
    head-dim shape is Lk=48. Because the head dim rides the matmul reduction axis,
    the pad is NOT free zero-fill — the host must allocate Q/K/V/O with the head
    dim padded to DMODEL and the pad zeroed. This test honors that contract:
    it builds Lk=48 data, zero-pads to DMODEL, runs, and compares the first Lk
    output columns against an Lk-wide reference (the pad columns stay ~0).
    """
    _require_variant(PAD_MLIR_PATH)
    rng = np.random.default_rng(42)
    shape48 = (HEADS, SEQ, PAD_LK)
    q48 = rng.standard_normal(shape48).astype(np.float16)
    k48 = rng.standard_normal(shape48).astype(np.float16)
    v48 = rng.standard_normal(shape48).astype(np.float16)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(PAD_LK)

    def pad(a):
        out = np.zeros((HEADS, SEQ, PAD_DMODEL), dtype=np.float16)
        out[:, :, :PAD_LK] = a
        return out

    result = _run(pad(q48), pad(k48), pad(v48), mask, sm_scale, mlir_path=PAD_MLIR_PATH)
    # Reference on the true Lk=48 data.
    expected48 = numpy_reference_batched(q48, k48, v48, mask, sm_scale)

    assert result.shape == (HEADS, SEQ, PAD_DMODEL)
    # First PAD_LK columns must match the Lk-wide reference.
    np.testing.assert_allclose(
        result[:, :, :PAD_LK].astype(np.float32), expected48, rtol=2e-2, atol=2e-2,
    )
    # Pad columns carry no real output (V's pad lanes are zero → acc pad ~ 0).
    np.testing.assert_allclose(
        result[:, :, PAD_LK:].astype(np.float32), 0.0, atol=1e-2,
    )
    max_err = np.max(np.abs(result[:, :, :PAD_LK].astype(np.float32) - expected48))
    print(f"PASS head-dim padding (Lk={PAD_LK}->DMODEL={PAD_DMODEL}): max abs error = {max_err:.6f}")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_fixed_batch():
    """Fixed multi-request batch, folded into the head axis.

    A batch of BATCH_N uniform-length requests is BATCH_N*HEADS independent
    attention problems — identical to more heads — so the host lays Q/K/V/O out
    as ``[BATCH_N*HEADS, SEQ, HEAD_DIM]`` and the *unchanged* kernel handles it.
    (A native 4D ``[BATCH, HEADS, …]`` descriptor is a compiler gap; folding is
    the pure-Proposal-2 route.) This is correct only when every request shares
    the mask — true for uniform-length prefill, where the mask is head/request-
    independent. Checks every (request, head) slice against the reference.
    """
    _require_variant(BATCH_MLIR_PATH)
    rng = np.random.default_rng(42)
    # Real batched data: [BATCH_N, HEADS, SEQ, HEAD_DIM].
    shape = (BATCH_N, HEADS, SEQ, HEAD_DIM)
    q = rng.standard_normal(shape).astype(np.float16)
    k = rng.standard_normal(shape).astype(np.float16)
    v = rng.standard_normal(shape).astype(np.float16)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Fold (batch, head) -> a single [BATCH_N*HEADS, SEQ, HEAD_DIM] head axis.
    fold = lambda a: a.reshape(BATCH_N * HEADS, SEQ, HEAD_DIM)
    result = _run(fold(q), fold(k), fold(v), mask, sm_scale, mlir_path=BATCH_MLIR_PATH)
    result = result.reshape(BATCH_N, HEADS, SEQ, HEAD_DIM)

    assert result.shape == (BATCH_N, HEADS, SEQ, HEAD_DIM)
    for b in range(BATCH_N):
        expected_b = numpy_reference_batched(q[b], k[b], v[b], mask, sm_scale)
        np.testing.assert_allclose(
            result[b].astype(np.float32), expected_b, rtol=2e-2, atol=2e-2,
        )
    max_err = max(
        np.max(np.abs(result[b].astype(np.float32)
                      - numpy_reference_batched(q[b], k[b], v[b], mask, sm_scale)))
        for b in range(BATCH_N)
    )
    print(f"PASS fixed batch (B={BATCH_N} x {HEADS} heads): max abs error = {max_err:.6f}")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_large_values():
    """Larger-magnitude inputs — exercises the online-softmax rescaling path.

    Scale is kept moderate (3.0): against an f32 NumPy reference, an f16
    online-softmax at scale=10 diverges by more than a fair equivalence
    tolerance purely from f16 rounding of near-degenerate softmax weights
    (verified: error is sparse and low-mean, not a structural tile bug — mean
    abs err ~1.6e-3, only a handful of the 32768 elements exceed 3e-2). The
    all-heads shape has HEADS× the elements of a single head, so the extreme
    tail reaches ~4e-2; the tolerance covers it. The GPU test compares
    kernel-vs-kernel and can push harder.
    """
    q, k, v = _make_inputs(seed=7, scale=3.0)
    mask = build_additive_mask(SEQ, is_causal=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    result = _run(q, k, v, mask, sm_scale)
    expected = numpy_reference_batched(q, k, v, mask, sm_scale)

    np.testing.assert_allclose(
        result.astype(np.float32), expected, rtol=4.5e-2, atol=4.5e-2,
    )
    print("PASS large values")


# ─── Distribution invariance ────────────────────────────────────────────────

# The SEQ=256 (two-BLOCK) variants exercise a query-block spanning more than one
# KV tile, so they cover the online-softmax rescaling across tiles. Requires the
# ktir-cpu outs-materialization fix (pin rev 78a05609) for correct accumulation
# across KV iterations.


@pytest.mark.ktir_cpu
class TestPrefillAttentionDistribution:
    """The distribution loop must be partition-independent.

    Every core walks the whole key range; the loop only slices the
    (head, query-block) work space across ``num_cores`` (frozen into each
    ``spyre_dist_c<N>.ktir`` at lowering). So the result must be identical for
    every core count. The DIST_SEQ = 256 shape is two BLOCK=128 query-blocks and
    HEADS heads, so the work space (HEADS * m_blocks = 8 items) genuinely differs
    per partition: grid 1 runs all 8 items on one core (sequential), grid 4 gives
    each core 2, grids 16/32 leave most cores idle (more cores than work). If any
    ``.ktir`` variant is missing (spyre toolchain not run), the whole class skips.
    """

    @staticmethod
    def _require(cores: int) -> Path:
        path = _dist_mlir_path(cores)
        if not path.is_file():
            pytest.skip(
                f"{path.name} not generated — regenerate with:\n"
                "  GIT_PAT=$(gh auth token) TRITON_DEFAULT_BACKEND=spyre uv run "
                '--with "triton @ git+https://github.com/fabianlim/triton@a6938df5" '
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
        expected = numpy_reference_batched(q, k, v, mask, sm_scale)

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
