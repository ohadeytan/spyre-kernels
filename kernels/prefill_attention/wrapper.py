import math

import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.prefill_attention.original import _fwd_kernel
from kernels.prefill_attention.spyre import _prefill_attention_kernel_spyre

RCP_LN2 = 1.0 / math.log(2.0)


def build_additive_mask(
    seq_len,
    device,
    is_causal: bool = True,
    sliding_window_q: int = 0,
    sliding_window_k: int = 0,
):
    """Host-precomputed additive attention mask [S, S]: 0.0 where a key is
    attended, -inf (finite stand-in) where masked.

    The Spyre kernel adds this to the scores instead of computing positions with
    tl.arange (which would leave tt.make_range in the KTIR). Encodes the same
    bands as original.py:
      - causal:            keep key j for query i when j <= i
      - backward window:   i - j <= sliding_window_q   (0 = unbounded)
      - forward window:    j - i <= sliding_window_k   (0 = unbounded)
    -1e9 is a finite stand-in for -inf so exp() underflows to 0 cleanly in fp32.
    """
    row = torch.arange(seq_len, device=device)[:, None]
    col = torch.arange(seq_len, device=device)[None, :]
    keep = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    if is_causal:
        keep &= col <= row
    if sliding_window_q > 0:
        keep &= (row - col) <= sliding_window_q
    if sliding_window_k > 0:
        keep &= (col - row) <= sliding_window_k
    return torch.where(
        keep,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), -1.0e9, device=device, dtype=torch.float32),
    ).contiguous()


def _additive_causal_mask(seq_len, device):
    """Back-compat alias: the default causal mask (no sliding window)."""
    return build_additive_mask(seq_len, device, is_causal=True)


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
    sliding_window_q: int = 0,
    sliding_window_k: int = 0,
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
        SLIDING_WINDOW_Q=sliding_window_q,
        SLIDING_WINDOW_K=sliding_window_k,
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
    mask: torch.Tensor | None = None,   # (S, S) additive mask, f32
    num_cores: int = 32,
    block: int = 64,
    sliding_window_q: int = 0,
    sliding_window_k: int = 0,
    logical_head_dim: int | None = None,
    seqlen: int | None = None,
):
    """Launch the Spyre prefill kernel (packed layout + <=32-core distribution).

    KTIR-lowerable: memory is descriptors-only and the mask is a host additive
    [S, S] tensor (default causal), so there is no in-kernel tl.arange. The
    kernel calls tl.exp (natural), so sm_scale carries no RCP_LN2 factor.

    Reclaimed capabilities (each needs nothing beyond this launch path):
      - **GQA**: pass K/V with fewer heads (``KVH < H``); the kernel reads KV head
        ``cur_head // (H // KVH)``.
      - **Sliding window**: ``sliding_window_q`` / ``sliding_window_k`` fold into
        the default mask (ignored if an explicit ``mask`` is given).
      - **Head-dim padding**: pass q/k/v/o with the physical head dim padded to a
        full stick and zero-filled past ``logical_head_dim``; set
        ``logical_head_dim`` to the true head dim. sm_scale defaults to
        1/sqrt(logical_head_dim).
      - **Fixed batch**: fold B uniform-length requests into the head axis
        (``H = B * heads``) on the host; this path is agnostic to the head count.
      - **Variable length**: ``S`` (the buffer's seq dim) is the padded max;
        pass ``seqlen`` (<= S) to bound the KV loop to a request's true length.
        Mask keys past ``seqlen`` (an already-bounded mask or the default causal
        one both suffice for causal prefill, where rows < seqlen never see keys
        beyond their own position anyway). ``seqlen`` is an i32 arg, not a
        B_Seqlen load — ktir-cpu-executable today.

    Distribution-invariant: the flattened (head, m_block) work is split over
    ``num_cores``; the result is independent of the partition. Single request
    here (batch = 1); ``S`` is the padded max seq dim, ``seqlen`` the runtime
    per-request length.
    """
    S, H, D = q.shape           # D is the PHYSICAL (possibly padded) head dim; S is the padded max
    KVH = k.shape[1]
    kv_group_num = H // KVH
    Lk = D if logical_head_dim is None else logical_head_dim
    sm_scale = 1.0 / (Lk ** 0.5) if softmax_scale is None else softmax_scale
    seq = S if seqlen is None else seqlen   # runtime per-request length (<= S)

    if mask is None:
        mask = build_additive_mask(
            S, q.device, is_causal=True,
            sliding_window_q=sliding_window_q, sliding_window_k=sliding_window_k,
        )

    num_m_blocks = triton.cdiv(S, block)

    ensure_triton_allocator()
    _prefill_attention_kernel_spyre[(num_cores,)](
        q, k, v, mask, o, sm_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        H, KVH, 1, num_m_blocks, seq,    # num_q_heads, num_kv_heads, batch, num_m_blocks, seqlen
        kv_group_num=kv_group_num,
        BLOCK_M=block,
        BLOCK_DMODEL=D,       # physical head dim (a whole number of sticks)
        BLOCK_N=block,
        Lk=Lk,                # logical head dim (<= D)
        S=128 // q.element_size(),
        SEQ=S,
    )
