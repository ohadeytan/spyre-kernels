import math

import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.prefill_attention.original import _fwd_kernel
from kernels.prefill_attention.spyre import _prefill_attention_kernel_spyre

RCP_LN2 = 1.0 / math.log(2.0)


def _additive_causal_mask(seq_len, device):
    """Host-precomputed additive mask [S, S]: 0.0 where key <= query, -inf else.

    The Spyre kernel adds this to the scores instead of computing positions with
    tl.arange (which would leave tt.make_range in the KTIR). -1e9 is a finite
    stand-in for -inf so exp() underflows to 0 cleanly in fp32.
    """
    row = torch.arange(seq_len, device=device)[:, None]
    col = torch.arange(seq_len, device=device)[None, :]
    return torch.where(
        col <= row,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), -1.0e9, device=device, dtype=torch.float32),
    ).expand(seq_len, seq_len).contiguous()


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    kernel_fn=_fwd_kernel,
):
    """Launch the original (or tensor-descriptor) packed prefill kernel.

    For the Spyre kernel use ``context_attention_fwd_spyre`` — its signature
    diverges (additive host mask, no strided per-batch pointer arithmetic), so it
    does not share this launch path.
    """
    Lk = q.shape[-1]
    sm_scale = 1.0 / (Lk ** 0.5) if softmax_scale is None else softmax_scale
    sm_scale *= RCP_LN2

    batch = b_seq_len.shape[0]
    head = q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    BLOCK = 64 if q.dtype == torch.float32 else 128
    BLOCK = min(BLOCK, triton.next_power_of_2(max_input_len))

    num_m_blocks = triton.cdiv(max_input_len, BLOCK)
    num_warps = 4 if Lk <= 64 else 8

    # GPU form: one program per (batch, head, m_block).
    grid = (batch, head, num_m_blocks)

    # The tensor-descriptor kernel needs the head counts to build its descriptor
    # shapes (the original derived head indexing purely from strides). Supply
    # them only when the target kernel declares them, and register the TMA
    # allocator that make_tensor_descriptor requires.
    extra = {}
    if "num_q_heads" in kernel_fn.arg_names:
        ensure_triton_allocator()
        extra = {
            "num_q_heads": q.shape[1],
            "num_kv_heads": k.shape[1],
        }

    kernel_fn[grid](
        q, k, v, sm_scale,
        b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        **extra,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW_Q=0,
        SLIDING_WINDOW_K=0,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )


def context_attention_fwd_spyre(
    q: torch.Tensor,        # (S, H, D)   packed queries (single request)
    k: torch.Tensor,        # (S, KVH, D)
    v: torch.Tensor,        # (S, KVH, D)
    o: torch.Tensor,        # (S, H, D)
    softmax_scale: float | None = None,
    mask: torch.Tensor | None = None,   # (S, S) additive causal mask, f32
    num_cores: int = 32,
    block: int = 64,
):
    """Launch the Spyre prefill kernel (packed layout + <=32-core distribution).

    KTIR-lowerable: memory is descriptors-only and the causal mask is a host
    additive [S, S] tensor (default causal), so there is no in-kernel tl.arange.
    The kernel calls tl.exp (natural), so sm_scale carries no RCP_LN2 factor.

    Distribution-invariant: the flattened (head, m_block) work is split over
    ``num_cores``; the result is independent of the partition. Single request
    here (batch = 1), matching the KTIR config; the seq_len rides as the SEQ
    constexpr.
    """
    S, H, D = q.shape
    KVH = k.shape[1]
    kv_group_num = H // KVH
    sm_scale = 1.0 / (D ** 0.5) if softmax_scale is None else softmax_scale

    if mask is None:
        mask = _additive_causal_mask(S, q.device)

    num_m_blocks = triton.cdiv(S, block)

    ensure_triton_allocator()
    _prefill_attention_kernel_spyre[(num_cores,)](
        q, k, v, mask, o, sm_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        H, KVH, 1, num_m_blocks,          # num_q_heads, num_kv_heads, batch, num_m_blocks
        kv_group_num=kv_group_num,
        BLOCK_M=block,
        BLOCK_DMODEL=triton.next_power_of_2(D),
        BLOCK_N=block,
        Lk=D,
        S=128 // q.element_size(),
        SEQ=S,
    )
