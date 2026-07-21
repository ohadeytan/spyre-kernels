# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the tensor-descriptor prefill attention kernel.

Compares ``_prefill_attention_kernel_td`` against the original ``_fwd_kernel``
across representative shapes. Both kernels are launched through
``kernels/prefill_attention/wrapper.py`` (via its ``kernel_fn=`` dispatch) — no
forked launch path.

The two kernels feed numerically-identical values into the two ``tl.dot`` calls
(the descriptor's OOB zero-fill equals the original's mask-based ``other=0.0``
fill), so they differ only in float op order: the _td kernel loads K un-
transposed and applies ``tl.trans`` before ``tl.dot(q, k)``, which can shift the
QK accumulation order versus the original's pre-transposed load. That reordering
lands in the f16/bf16 output at ~1 ULP of the output dtype, so tolerances are
sized to the output dtype, not f32.

Run: pytest tests/triton/test_prefill_attention_td.py -v
Requires: GPU with triton tensor-descriptor support.
"""

import pytest
import torch

from kernels.prefill_attention.tensor_descriptor import _prefill_attention_kernel_td
from kernels.prefill_attention.wrapper import context_attention_fwd


# Output is f16/bf16 (not f32): the QK reorder from the K transpose shows up at
# ~1 ULP of the output dtype. f32 accumulators keep the gap near the dtype floor.
TOL = {
    torch.float16: dict(atol=1e-3, rtol=1e-3),   # 10-bit mantissa
    torch.bfloat16: dict(atol=8e-3, rtol=8e-3),  # 7-bit mantissa
}

DTYPES = [torch.float16, torch.bfloat16]


# ─── Launch helpers (both go through the wrapper) ──────────────────

def attn_ref(q, k, v, b_start_loc, b_seq_len, max_len, **kwargs):
    """Reference: original _fwd_kernel via the wrapper."""
    o = torch.zeros_like(q)
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_len, **kwargs)
    return o


def attn_td(q, k, v, b_start_loc, b_seq_len, max_len, **kwargs):
    """Tensor-descriptor kernel via the wrapper's kernel_fn dispatch."""
    o = torch.zeros_like(q)
    context_attention_fwd(
        q, k, v, o, b_start_loc, b_seq_len, max_len,
        kernel_fn=_prefill_attention_kernel_td, **kwargs
    )
    return o


def _make_inputs(seq_lens, num_q_heads, num_kv_heads, head_dim, device, dtype,
                 scale=1.0):
    torch.manual_seed(42)
    total_tokens = sum(seq_lens)
    batch = len(seq_lens)
    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype) * scale
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype) * scale
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype) * scale
    b_seq_len = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    b_start_loc = torch.zeros(batch, device=device, dtype=torch.int32)
    for i in range(1, batch):
        b_start_loc[i] = b_start_loc[i - 1] + seq_lens[i - 1]
    max_input_len = max(seq_lens)
    return q, k, v, b_start_loc, b_seq_len, max_input_len


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ─── Test parameters ────────────────────────────────────────────────

# BLOCK = 128 for f16/bf16 (see wrapper). Seq lengths span single-tile,
# tile-aligned, and non-divisible cases.
SEQ_LENS_CONFIGS = [
    [16],              # single tiny sequence (< BLOCK)
    [128],             # exactly one tile
    [200],             # non-divisible: one full tile + partial tail
    [256],             # two full tiles
    [300],             # two tiles + partial tail
    [16, 32],          # multiple short batches
    [128, 200],        # multiple batches, one non-divisible
    [37, 128, 91],     # asymmetric, non-divisible batch sizes
]


# ─── Correctness vs original ────────────────────────────────────────

class TestPrefillAttentionTDCorrectness:
    """Tensor-descriptor kernel must match the original kernel."""

    @pytest.mark.parametrize("seq_lens", SEQ_LENS_CONFIGS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_equivalence_causal(self, device, seq_lens, dtype):
        q, k, v, bsl, bseq, max_len = _make_inputs(seq_lens, 4, 4, 64, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("seq_lens", SEQ_LENS_CONFIGS)
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_equivalence_non_causal(self, device, seq_lens, dtype):
        q, k, v, bsl, bseq, max_len = _make_inputs(seq_lens, 4, 4, 64, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=False)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=False)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_gqa(self, device, dtype):
        """GQA: num_q_heads > num_kv_heads (kv_group_num > 1)."""
        q, k, v, bsl, bseq, max_len = _make_inputs([200], 8, 2, 64, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_head_dims(self, device, head_dim, dtype):
        q, k, v, bsl, bseq, max_len = _make_inputs([200], 4, 4, head_dim, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])


# ─── Edge cases ─────────────────────────────────────────────────────

class TestPrefillAttentionTDEdgeCases:
    """Boundary and stress cases for the tensor-descriptor kernel."""

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_single_tile_bitwise(self, device, dtype):
        """seq_len <= BLOCK: the KV loop runs a single tile, so there is no
        cross-tile reduction and no per-iteration K reload — the QK op order is
        identical. Output must be bitwise-identical."""
        q, k, v, bsl, bseq, max_len = _make_inputs([64], 4, 4, 64, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_minimum_size(self, device, dtype):
        """Smallest usable sequence (16 rows, one partial tile)."""
        q, k, v, bsl, bseq, max_len = _make_inputs([16], 4, 4, 64, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_non_divisible_seqlen(self, device, dtype):
        """Sequence length not a multiple of BLOCK — exercises the OOB tail on
        the row (BLOCK_M) and KV (BLOCK_N) descriptors."""
        q, k, v, bsl, bseq, max_len = _make_inputs([333], 4, 4, 64, device, dtype)

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_asymmetric_batches(self, device, dtype):
        """Highly uneven per-batch sequence lengths."""
        q, k, v, bsl, bseq, max_len = _make_inputs(
            [8, 256, 40, 129], 4, 4, 64, device, dtype
        )

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_zero_input(self, device, dtype):
        """All-zero Q/K/V — uniform attention weights, zero output."""
        q, k, v, bsl, bseq, max_len = _make_inputs([200], 4, 4, 64, device, dtype)
        q.zero_(); k.zero_(); v.zero_()

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda dt: str(dt).split(".")[-1])
    def test_large_values(self, device, dtype):
        """Large-magnitude inputs — stresses the online-softmax rescaling path."""
        q, k, v, bsl, bseq, max_len = _make_inputs(
            [200], 4, 4, 64, device, dtype, scale=10.0
        )

        out_ref = attn_ref(q, k, v, bsl, bseq, max_len, is_causal=True)
        out_td = attn_td(q, k, v, bsl, bseq, max_len, is_causal=True)

        torch.testing.assert_close(out_td, out_ref, **TOL[dtype])
