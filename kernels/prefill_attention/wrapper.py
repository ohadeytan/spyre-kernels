import math

import torch
import triton

from kernels._tma import ensure_triton_allocator
from kernels.prefill_attention.original import _fwd_kernel

RCP_LN2 = 1.0 / math.log(2.0)


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
    Lk = q.shape[-1]
    sm_scale = 1.0 / (Lk ** 0.5) if softmax_scale is None else softmax_scale
    sm_scale *= RCP_LN2

    batch = b_seq_len.shape[0]
    head = q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    BLOCK = 64 if q.dtype == torch.float32 else 128
    BLOCK = min(BLOCK, triton.next_power_of_2(max_input_len))

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    # The tensor-descriptor kernel needs the head counts to build its 3D
    # descriptor shapes (the original derived head indexing purely from
    # strides). Supply them only when the target kernel declares them, and
    # register the TMA allocator that make_tensor_descriptor requires.
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
