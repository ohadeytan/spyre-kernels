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


def attn_spyre(q, k, v, mask=None, num_cores=32, block=32):
    o = torch.zeros_like(q)
    context_attention_fwd_spyre(
        q, k, v, o, mask=mask, num_cores=num_cores, block=block
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
