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
GQA_MLIR_PATH = MLIR_PATH.parent / "spyre_gqa.ktir"
PAD_MLIR_PATH = MLIR_PATH.parent / "spyre_pad.ktir"

FUNC = "_prefill_attention_kernel_spyre"

# Concrete shapes baked into the generated KTIR (see lower.py). One request.
SEQ = 128
H = 4
KVH = 4
D = 64
BLOCK = 64
NUM_M_BLOCKS = SEQ // BLOCK  # 2

# GQA variant (spyre_gqa): 4 query heads share 2 KV heads (group size 2).
GQA_KVH = 2
# Head-dim padding variant (spyre_pad): logical head dim 48, padded to one stick.
PAD_LK = 48
PAD_DMODEL = 64
# Fixed-batch: B requests folded into the head axis as B*H "heads" (base KTIR,
# launched with runtime num_q_heads = B*H).
BATCH_N = 2

SM_SCALE = np.float32(1.0 / math.sqrt(D))
NEG_INF = np.float32(-1.0e9)


def causal_mask():
    """Additive [SEQ, SEQ] causal mask: 0.0 where key <= query, -inf otherwise."""
    row = np.arange(SEQ)[:, None]
    col = np.arange(SEQ)[None, :]
    return np.where(col <= row, np.float32(0.0), NEG_INF).astype(np.float32)


def numpy_reference(q, k, v, mask, *, sm_scale=None, kv_group=1):
    """Causal SDPA in f32, matching the kernel's upcast-accumulate-downcast.

    q: [SEQ, Hq, D] f16. k, v: [SEQ, Hkv, D] f16 with ``Hq == Hkv * kv_group``
    (query head ``h`` reads KV head ``h // kv_group``). mask: [SEQ, S_kv] additive
    f32. Handles the MHA case (kv_group == 1, Hkv == Hq) and GQA identically.
    """
    scale = float(SM_SCALE if sm_scale is None else sm_scale)
    qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
    seq, hq, d = qf.shape
    out = np.zeros((seq, hq, d), dtype=np.float32)
    for h in range(hq):
        kv = h // kv_group
        s = (qf[:, h, :] * scale) @ kf[:, kv, :].T + mask
        s -= s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
        out[:, h, :] = w @ vf[:, kv, :]
    return out.astype(np.float16)


def _run(q, k, v, mask, *, mlir_path=MLIR_PATH, out=None, seqlen=None):
    """Execute the generated KTIR on ktir-cpu.

    Strides are derived from each array's own physical shape ``[S, heads, D]``
    (row-major, element units), so this handles the base MHA case, GQA (K/V with
    fewer heads), head-dim padding (a wider physical D), and a B*HEADS-folded
    batch uniformly — the only per-array facts the kernel needs are its
    ``stride(seq)`` and ``stride(head)``. ``seqlen`` is the runtime per-request
    length bounding the KV loop; it defaults to the full padded ``SEQ``.
    """
    interp = KTIRInterpreter()
    interp.load(mlir_path.read_text())

    n_q_heads, n_kv_heads = q.shape[1], k.shape[1]
    if out is None:
        out = np.zeros_like(q)
    if seqlen is None:
        seqlen = q.shape[0]

    def _strides(a):  # (stride_bs, stride_h) in elements, row-major [S, heads, D]
        return np.int32(a.shape[1] * a.shape[2]), np.int32(a.shape[2])

    q_bs, q_h = _strides(q)
    k_bs, k_h = _strides(k)
    v_bs, v_h = _strides(v)
    o_bs, o_h = _strides(out)

    outputs = interp.execute_function(
        FUNC,
        arg0=q, arg1=k, arg2=v, arg3=mask, arg4=out,
        arg5=SM_SCALE,
        arg6=q_bs, arg7=q_h,
        arg8=k_bs, arg9=k_h,
        arg10=v_bs, arg11=v_h,
        arg12=o_bs, arg13=o_h,
        arg14=np.int32(n_q_heads),
        arg15=np.int32(n_kv_heads),
        arg16=np.int32(1),          # batch (folded into the head axis)
        arg17=np.int32(NUM_M_BLOCKS),
        arg18=np.int32(seqlen),     # runtime per-request length (<= SEQ)
    )
    return outputs["arg4"]


def _dist_mlir_path(cores: int) -> Path:
    return MLIR_PATH.parent / f"spyre_dist_c{cores}.ktir"


DIST_CORE_COUNTS = (1, 4, 16, 32)


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


# ─── Reclaimed capabilities: GQA / head-dim padding / sliding window / batch ──


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(
            f"{path.name} not generated — regenerate with:\n"
            "  GIT_PAT=$(gh auth token) TRITON_DEFAULT_BACKEND=spyre uv run "
            '--with "$SPYRE_TRITON" python scripts/gen_ktir.py prefill_attention'
        )
    return path


def windowed_mask(seq, *, sliding_window_q=0, sliding_window_k=0):
    """Additive [seq, seq] causal mask with optional bidirectional sliding
    windows, matching original.py's bands: keep key j for query i when
    ``j <= i`` (causal) and ``i - j <= W_Q`` and ``j - i <= W_K`` (a window of 0
    means unbounded on that side). -inf elsewhere."""
    row = np.arange(seq)[:, None]
    col = np.arange(seq)[None, :]
    keep = col <= row
    if sliding_window_q > 0:
        keep &= (row - col) <= sliding_window_q
    if sliding_window_k > 0:
        keep &= (col - row) <= sliding_window_k
    return np.where(keep, np.float32(0.0), NEG_INF).astype(np.float32)


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_gqa():
    """GQA (spyre_gqa): 4 query heads share 2 KV heads. Each query head h reads
    KV head h // 2; result matches the per-head GQA reference."""
    path = _require(GQA_MLIR_PATH)
    rng = np.random.default_rng(42)
    q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, GQA_KVH, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ, GQA_KVH, D)) * 0.1).astype(np.float16)
    mask = causal_mask()
    kv_group = H // GQA_KVH

    result = _run(q, k, v, mask, mlir_path=path)
    expected = numpy_reference(q, k, v, mask, kv_group=kv_group)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2
    )
    # Sanity: query heads 0,1 (group 0) use KV head 0; heads 2,3 use KV head 1.
    print("PASS: GQA 4q/2kv per-head equivalence")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_head_dim_padding():
    """Head-dim padding (spyre_pad): logical Lk=48 padded to a full stick (64).

    Honors the host contract — the physical head dim is PAD_DMODEL and lanes
    [PAD_LK, PAD_DMODEL) are zero. Padding sits on the QK/PV reduction axis, so
    zeros contribute nothing and the first PAD_LK output columns match a
    reference computed on the logical (unpadded) head dim."""
    path = _require(PAD_MLIR_PATH)
    rng = np.random.default_rng(3)

    def padded(shape_heads):
        a = np.zeros((SEQ, shape_heads, PAD_DMODEL), dtype=np.float16)
        a[:, :, :PAD_LK] = (rng.standard_normal((SEQ, shape_heads, PAD_LK)) * 0.1).astype(np.float16)
        return a

    q, k, v = padded(H), padded(KVH), padded(KVH)
    mask = causal_mask()
    # sm_scale for the pad variant is baked to 1/sqrt(D=64) in the KTIR (SM_SCALE),
    # matching the kernel — the reference must use the same scale.
    result = _run(q, k, v, mask, mlir_path=path)
    expected = numpy_reference(q, k, v, mask)  # zeros in pad lanes are inert

    # Compare only the logical head-dim columns; pad columns are unspecified.
    np.testing.assert_allclose(
        result[:, :, :PAD_LK].astype(np.float32),
        expected[:, :, :PAD_LK].astype(np.float32),
        rtol=1e-2, atol=1e-2,
    )
    print("PASS: head-dim padding Lk=48 -> stick 64 (first 48 cols)")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_sliding_window():
    """Sliding window (base KTIR, windowed host mask): the window lives entirely
    in the additive MASK, so the unchanged kernel reproduces original.py's
    bidirectional bands. Backward window W_Q keeps keys within W_Q positions
    behind each query."""
    rng = np.random.default_rng(19)
    q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    # W_Q=32 keeps each query's own 32-back window; every row keeps >=1 key
    # (its own position j=i), so no row is fully masked (no 0/0 NaN).
    mask = windowed_mask(SEQ, sliding_window_q=32)

    result = _run(q, k, v, mask)
    expected = numpy_reference(q, k, v, mask)

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2
    )
    print("PASS: sliding window W_Q=32")


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_fixed_batch():
    """Fixed multi-request batch (base KTIR, B*H head fold): B=2 uniform-length
    requests laid out as [B*H, SEQ, D] and driven as B*H 'heads'. Each
    (request, head) slice must match the single-request reference (shared causal
    mask, uniform prefill)."""
    rng = np.random.default_rng(23)
    bh = BATCH_N * H
    q = (rng.standard_normal((SEQ, bh, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, bh, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ, bh, D)) * 0.1).astype(np.float16)
    mask = causal_mask()

    result = _run(q, k, v, mask)  # base KTIR; runtime num_q_heads = bh via strides
    expected = numpy_reference(q, k, v, mask)  # MHA over all bh "heads"

    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2
    )
    print(f"PASS: fixed batch B={BATCH_N} folded into {bh} heads")


def _bounded_bidirectional_mask(seqlen):
    """Additive [SEQ, SEQ] mask that attends to ALL keys < seqlen (bidirectional)
    and masks keys >= seqlen. Bidirectional so that a shorter seqlen genuinely
    changes every query row (a causal mask would leave rows < seqlen unchanged),
    which lets the test prove the runtime KV-loop bound is honored."""
    col = np.arange(SEQ)[None, :]
    keep = col < seqlen
    return np.where(keep, np.float32(0.0), NEG_INF).astype(np.float32)


@pytest.mark.ktir_cpu
def test_prefill_attention_ktir_variable_length():
    """Variable length (base KTIR, runtime seqlen arg): SEQ is the padded max;
    the runtime seqlen bounds the KV loop so a launch attends only over its true
    length. Validated two ways: (1) a short-seqlen run matches a reference
    computed over [:seqlen]; (2) it differs from the full-length run under a
    bidirectional mask (proving the bound is real, not a no-op)."""
    rng = np.random.default_rng(31)
    q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
    k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
    v = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)

    short = 64  # a whole BLOCK; attends to keys [0, 64)
    mask = _bounded_bidirectional_mask(short)

    result = _run(q, k, v, mask, seqlen=short).astype(np.float32)

    # Reference: bidirectional softmax over the first `short` keys only.
    qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
    expected = np.zeros((SEQ, H, D), np.float32)
    for h in range(H):
        s = (qf[:, h, :] * float(SM_SCALE)) @ kf[:short, h, :].T
        s -= s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
        expected[:, h, :] = w @ vf[:short, h, :]

    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)

    # The runtime bound must actually matter: a full-length run over the same
    # bidirectional-but-unbounded mask differs from the seqlen-bounded one.
    full_mask = _bounded_bidirectional_mask(SEQ)
    full = _run(q, k, v, full_mask, seqlen=SEQ).astype(np.float32)
    assert not np.allclose(full[0], result[0], rtol=1e-2, atol=1e-2), (
        "seqlen bound had no effect — full-length and short runs agree"
    )
    print(f"PASS: variable length seqlen={short} (of SEQ={SEQ})")


# ─── Distribution invariance ────────────────────────────────────────────────


@pytest.mark.ktir_cpu
class TestPrefillAttentionDistribution:
    """The distribution loop must be partition-independent.

    Each core walks the whole key range; the loop only slices the
    (batch, head, m_block) work space across ``num_cores`` (frozen into each
    ``spyre_dist_c<N>.ktir`` at lowering). So the result must be identical for
    every core count. The default shape has 8 work items
    (batch 1 × 4 heads × 2 m_blocks), so the partitions genuinely differ: grid 1
    runs all 8 on one core, grid 4 puts 2 per core, grids 16/32 leave most cores
    idle (more cores than work). If a ``.ktir`` variant is missing (spyre
    toolchain not run), that parametrization skips.
    """

    @staticmethod
    def _require(cores: int) -> Path:
        path = _dist_mlir_path(cores)
        if not path.is_file():
            pytest.skip(
                f"{path.name} not generated — regenerate with:\n"
                "  GIT_PAT=$(gh auth token) TRITON_DEFAULT_BACKEND=spyre uv run "
                '--with "$SPYRE_TRITON" python scripts/gen_ktir.py prefill_attention'
            )
        return path

    @pytest.mark.parametrize("cores", DIST_CORE_COUNTS)
    def test_matches_reference(self, cores):
        """Each core count matches the NumPy causal-attention reference."""
        mlir_path = self._require(cores)
        rng = np.random.default_rng(42)
        q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
        k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
        v = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
        mask = causal_mask()

        result = _run(q, k, v, mask, mlir_path=mlir_path)
        expected = numpy_reference(q, k, v, mask)

        np.testing.assert_allclose(
            result.astype(np.float32), expected.astype(np.float32),
            rtol=1e-2, atol=1e-2,
        )
        max_err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
        print(f"PASS cores={cores}: max abs error = {max_err:.6f}")

    def test_invariant_across_core_counts(self):
        """All core counts must agree bitwise: a work item's online-softmax is
        independent of which core runs it, so re-partitioning cannot shift the
        result (per-item op order is identical across grids; only the
        item-to-core assignment changes)."""
        rng = np.random.default_rng(42)
        q = (rng.standard_normal((SEQ, H, D)) * 0.1).astype(np.float16)
        k = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
        v = (rng.standard_normal((SEQ, KVH, D)) * 0.1).astype(np.float16)
        mask = causal_mask()

        baseline_cores = DIST_CORE_COUNTS[0]
        baseline = _run(q, k, v, mask, mlir_path=self._require(baseline_cores))
        for cores in DIST_CORE_COUNTS[1:]:
            result = _run(q, k, v, mask, mlir_path=self._require(cores))
            np.testing.assert_array_equal(
                result, baseline,
                err_msg=f"cores={cores} differs from cores={baseline_cores}",
            )
        print(f"PASS: bitwise-identical across cores {DIST_CORE_COUNTS}")
