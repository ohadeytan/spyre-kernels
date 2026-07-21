# SPDX-License-Identifier: Apache-2.0
"""GPU numerical tests for the Spyre prefill attention kernel.

``_prefill_attention_kernel_spyre`` is the KTIR-lowerable Spyre form: packed
(S, H, D) layout, a <=32-core distribution loop, physical stick-tiled head_dim
descriptors, and a host-precomputed additive causal mask (no tl.arange, so it
lowers to tt.make_range-free KTIR — see tests/ktir/test_prefill_attention.py).

Both kernels are launched through ``kernels/prefill_attention/wrapper.py`` — the
Spyre kernel via ``context_attention_fwd_spyre`` (which also drives ``num_cores``
for the distribution tests), the reference via ``context_attention_fwd`` driving
the vLLM-derived original ``_fwd_kernel``. The Spyre kernel is single-request, so
the reference is fed the same buffer as one packed request (b_start_loc=[0],
b_seq_len=[S]) — the batch=1 packed layout is the same contiguous [S, H, D].

The Spyre kernel loads K un-transposed and ``tl.trans``-poses before ``tl.dot``,
so the QK accumulation order can differ from the original at ~1 ULP of the f16
output; tolerances are sized to the output dtype. The distribution loop does not
change per-item math, so every ``num_cores`` is bitwise-identical to the 32-core
launch.

Run: pytest tests/triton/test_prefill_attention_spyre.py -v
Requires: GPU with triton tensor-descriptor support.
"""

import pytest
import torch

from kernels.prefill_attention.wrapper import (
    _additive_causal_mask,
    context_attention_fwd,
    context_attention_fwd_spyre,
)

TOL = {
    torch.float16: dict(atol=2e-3, rtol=2e-3),
}

NUM_CORES = [1, 4, 16, 32]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def attn_ref(q, k, v):
    """Reference: original _fwd_kernel via the wrapper, single packed request.

    q, k, v are [S, H, D]; a batch=1 packed request over the same contiguous
    buffer (b_start_loc=[0], b_seq_len=[S]). Causal.
    """
    S = q.shape[0]
    o = torch.zeros_like(q)
    b_start_loc = torch.zeros(1, device=q.device, dtype=torch.int32)
    b_seq_len = torch.tensor([S], device=q.device, dtype=torch.int32)
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, S, is_causal=True)
    return o


def attn_spyre(q, k, v, mask=None, num_cores=32, block=32, **kwargs):
    o = torch.zeros_like(q)
    context_attention_fwd_spyre(
        q, k, v, o, mask=mask, num_cores=num_cores, block=block, **kwargs
    )
    return o


def _make(S, H, D, device, dtype=torch.float16, scale_in=0.1):
    torch.manual_seed(42)
    q = torch.randn(S, H, D, device=device, dtype=dtype) * scale_in
    k = torch.randn(S, H, D, device=device, dtype=dtype) * scale_in
    v = torch.randn(S, H, D, device=device, dtype=dtype) * scale_in
    return q, k, v


# ─── Correctness vs the original kernel ──────────────────────────────

class TestPrefillAttentionSpyreCorrectness:
    """Spyre kernel must match the vLLM-derived original _fwd_kernel."""

    @pytest.mark.parametrize("S", [64, 128, 256])
    @pytest.mark.parametrize("H,D", [(4, 64), (8, 128)])
    def test_equivalence_causal(self, device, S, H, D):
        dtype = torch.float16
        q, k, v = _make(S, H, D, device, dtype)

        out_ref = attn_ref(q, k, v)
        out_spyre = attn_spyre(q, k, v)  # default (causal) mask

        torch.testing.assert_close(out_spyre, out_ref, **TOL[dtype])

    def test_explicit_causal_mask_matches_default(self, device):
        """Passing the wrapper's causal mask == relying on the default one."""
        dtype = torch.float16
        q, k, v = _make(128, 4, 64, device, dtype)
        mask = _additive_causal_mask(128, device)

        out_default = attn_spyre(q, k, v)
        out_explicit = attn_spyre(q, k, v, mask=mask)

        torch.testing.assert_close(out_explicit, out_default, atol=0, rtol=0)

    def test_first_row_identity(self, device):
        """Causal query 0 attends only to key/value 0."""
        dtype = torch.float16
        q, k, v = _make(128, 4, 64, device, dtype)

        out_spyre = attn_spyre(q, k, v)

        torch.testing.assert_close(out_spyre[0], v[0], **TOL[dtype])


# ─── Distribution invariance ─────────────────────────────────────────

class TestPrefillAttentionSpyreDistribution:
    """The distribution loop partitions the flattened (head, m_block) work over
    ``num_cores``; per-item math is identical regardless of partitioning, so
    every core count matches the original and is bitwise-identical to the
    32-core launch."""

    @pytest.mark.parametrize("num_cores", NUM_CORES)
    def test_matches_original(self, device, num_cores):
        dtype = torch.float16
        q, k, v = _make(128, 4, 64, device, dtype)

        out_ref = attn_ref(q, k, v)
        out_spyre = attn_spyre(q, k, v, num_cores=num_cores)

        torch.testing.assert_close(out_spyre, out_ref, **TOL[dtype])

    @pytest.mark.parametrize("num_cores", NUM_CORES)
    def test_bitwise_across_partitions(self, device, num_cores):
        dtype = torch.float16
        q, k, v = _make(128, 8, 64, device, dtype)

        out32 = attn_spyre(q, k, v, num_cores=32)
        out_n = attn_spyre(q, k, v, num_cores=num_cores)

        torch.testing.assert_close(out_n, out32, atol=0, rtol=0)

    @pytest.mark.parametrize("num_cores", [64, 256])
    def test_more_cores_than_work(self, device, num_cores):
        """More cores than (head x m_block) work items — extra cores idle."""
        dtype = torch.float16
        q, k, v = _make(64, 1, 64, device, dtype)  # 1 head, 1 m-block => 1 work item

        out_ref = attn_ref(q, k, v)
        out_spyre = attn_spyre(q, k, v, num_cores=num_cores, block=64)

        torch.testing.assert_close(out_spyre, out_ref, **TOL[dtype])


# ─── Reclaimed capabilities vs the original kernel ────────────────────────────


def _make_gqa(S, H, KVH, D, device, dtype=torch.float16, scale_in=0.1):
    torch.manual_seed(42)
    q = torch.randn(S, H, D, device=device, dtype=dtype) * scale_in
    k = torch.randn(S, KVH, D, device=device, dtype=dtype) * scale_in
    v = torch.randn(S, KVH, D, device=device, dtype=dtype) * scale_in
    return q, k, v


class TestPrefillAttentionSpyreCapabilities:
    """GQA, sliding window, head-dim padding, and fixed multi-request batch —
    each must match the vLLM-derived original ``_fwd_kernel`` (through the same
    wrapper) on the reclaimed configuration."""

    @pytest.mark.parametrize("H,KVH", [(4, 2), (8, 2), (8, 4)])
    def test_gqa_matches_original(self, device, H, KVH):
        """Grouped-query attention: fewer KV heads, query head h reads KV head
        h // (H // KVH). The kernel already does this at the descriptor load
        index; only the KV head count shrinks."""
        dtype = torch.float16
        S, D = 128, 64
        q, k, v = _make_gqa(S, H, KVH, D, device, dtype)

        # Reference: original kernel derives kv_group_num from the shapes.
        o_ref = torch.zeros_like(q)
        b_start_loc = torch.zeros(1, device=device, dtype=torch.int32)
        b_seq_len = torch.tensor([S], device=device, dtype=torch.int32)
        context_attention_fwd(q, k, v, o_ref, b_start_loc, b_seq_len, S, is_causal=True)

        o_spyre = attn_spyre(q, k, v)
        torch.testing.assert_close(o_spyre, o_ref, **TOL[dtype])

    @pytest.mark.parametrize("W_Q,W_K", [(32, 0), (0, 32), (48, 16)])
    def test_sliding_window_matches_original(self, device, W_Q, W_K):
        """Sliding window Q/K: the window lives in the additive host mask; the
        original applies it as in-kernel position bands. Both must agree."""
        dtype = torch.float16
        S, H, D = 128, 4, 64
        q, k, v = _make(S, H, D, device, dtype)

        o_ref = torch.zeros_like(q)
        b_start_loc = torch.zeros(1, device=device, dtype=torch.int32)
        b_seq_len = torch.tensor([S], device=device, dtype=torch.int32)
        context_attention_fwd(
            q, k, v, o_ref, b_start_loc, b_seq_len, S, is_causal=True,
            sliding_window_q=W_Q, sliding_window_k=W_K,
        )

        o_spyre = attn_spyre(q, k, v, sliding_window_q=W_Q, sliding_window_k=W_K)
        torch.testing.assert_close(o_spyre, o_ref, **TOL[dtype])

    def test_head_dim_padding_matches_padded_original(self, device):
        """Head-dim padding: a logical head dim not filling a full stick. Both
        kernels run on the physically padded (zero-filled) buffer; compare the
        logical columns. The original kernel masks the head-dim tail (mask_d);
        the spyre kernel relies on the host zeros."""
        dtype = torch.float16
        S, H = 128, 4
        Lk, Dpad = 48, 64
        torch.manual_seed(42)
        q = torch.zeros(S, H, Dpad, device=device, dtype=dtype)
        k = torch.zeros(S, H, Dpad, device=device, dtype=dtype)
        v = torch.zeros(S, H, Dpad, device=device, dtype=dtype)
        q[:, :, :Lk] = torch.randn(S, H, Lk, device=device, dtype=dtype) * 0.1
        k[:, :, :Lk] = torch.randn(S, H, Lk, device=device, dtype=dtype) * 0.1
        v[:, :, :Lk] = torch.randn(S, H, Lk, device=device, dtype=dtype) * 0.1

        # Reference: original kernel with logical Lk (it masks the head-dim tail).
        o_ref = torch.zeros_like(q)
        b_start_loc = torch.zeros(1, device=device, dtype=torch.int32)
        b_seq_len = torch.tensor([S], device=device, dtype=torch.int32)
        context_attention_fwd(
            q, k, v, o_ref, b_start_loc, b_seq_len, S, is_causal=True,
            softmax_scale=1.0 / (Lk ** 0.5),
        )

        o_spyre = torch.zeros_like(q)
        from kernels.prefill_attention.wrapper import context_attention_fwd_spyre
        context_attention_fwd_spyre(q, k, v, o_spyre, logical_head_dim=Lk)

        torch.testing.assert_close(
            o_spyre[:, :, :Lk], o_ref[:, :, :Lk], **TOL[dtype]
        )

    @pytest.mark.parametrize("B,H", [(2, 4), (3, 2)])
    def test_fixed_batch_matches_per_request(self, device, B, H):
        """Fixed multi-request batch folded into the head axis: [B*H, S, D]. Each
        (request, head) slice must match running that request alone."""
        dtype = torch.float16
        S, D = 128, 64
        torch.manual_seed(42)
        q = torch.randn(S, B * H, D, device=device, dtype=dtype) * 0.1
        k = torch.randn(S, B * H, D, device=device, dtype=dtype) * 0.1
        v = torch.randn(S, B * H, D, device=device, dtype=dtype) * 0.1

        o_batch = torch.zeros_like(q)
        from kernels.prefill_attention.wrapper import context_attention_fwd_spyre
        context_attention_fwd_spyre(q, k, v, o_batch)

        # Reference: each request's H-head block run through the original alone.
        for b in range(B):
            sl = slice(b * H, (b + 1) * H)
            qb, kb, vb = q[:, sl].contiguous(), k[:, sl].contiguous(), v[:, sl].contiguous()
            ob = attn_ref(qb, kb, vb)
            torch.testing.assert_close(o_batch[:, sl], ob, **TOL[dtype])

    @pytest.mark.parametrize("short", [64, 96])
    def test_variable_length_matches_original(self, device, short):
        """Variable length: SEQ_MAX buffer, runtime seqlen bounds the attention.
        The original kernel bounds attention with b_seq_len; the spyre kernel
        uses a matching causal mask bounded to `short` plus the runtime seqlen
        arg. Only the first `short` output rows are defined (queries past the
        request length are not part of the request)."""
        dtype = torch.float16
        S_max, H, D = 128, 4, 64
        q, k, v = _make(S_max, H, D, device, dtype)

        # Reference: original with b_seq_len = [short] (causal, valid-length).
        o_ref = torch.zeros_like(q)
        b_start_loc = torch.zeros(1, device=device, dtype=torch.int32)
        b_seq_len = torch.tensor([short], device=device, dtype=torch.int32)
        context_attention_fwd(q, k, v, o_ref, b_start_loc, b_seq_len, S_max, is_causal=True)

        # Spyre: causal mask bounded to `short` keys + runtime seqlen.
        from kernels.prefill_attention.wrapper import build_additive_mask
        mask = build_additive_mask(S_max, device, is_causal=True)
        # Mask out keys >= short (the padded tail beyond this request's length).
        col = torch.arange(S_max, device=device)[None, :]
        mask = torch.where(col < short, mask, torch.full_like(mask, -1.0e9)).contiguous()

        o_spyre = attn_spyre(q, k, v, mask=mask, seqlen=short)

        # Compare only the valid query rows [0, short).
        torch.testing.assert_close(o_spyre[:short], o_ref[:short], **TOL[dtype])
