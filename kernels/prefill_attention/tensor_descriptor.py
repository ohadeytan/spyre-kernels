# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Tensor-descriptor conversion of prefill attention _fwd_kernel.
# Original: kernels/prefill_attention/original.py
# Changes summarized in kernels/prefill_attention/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _prefill_attention_kernel_td(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    num_q_heads,
    num_kv_heads,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW_Q: tl.constexpr,
    SLIDING_WINDOW_K: tl.constexpr,
    Lk: tl.constexpr,
):
    """Flash-attention prefill over a packed (total_tokens, heads, head_dim)
    layout, using tensor descriptors for all Q/K/V/O accesses."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Rebase each descriptor at this batch's first token so descriptor
    # coordinates run 0..cur_batch_seq_len. The batch start is a scalar offset
    # folded into the base pointer, not a per-row runtime index, so plain
    # desc.load / desc.store suffice (no gather).
    q_base = Q + cur_batch_in_all_start_index * stride_qbs
    k_base = K + cur_batch_in_all_start_index * stride_kbs
    v_base = V + cur_batch_in_all_start_index * stride_vbs
    o_base = Out + cur_batch_in_all_start_index * stride_obs

    # 3D descriptors over (seq_len, heads, head_dim). The last (head_dim) axis
    # carries BLOCK_DMODEL * dtype_bytes >= 16 bytes; the head axis selects one
    # head via a length-1 tile.
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[cur_batch_seq_len, num_q_heads, Lk],
        strides=[stride_qbs, stride_qh, 1],
        block_shape=[BLOCK_M, 1, BLOCK_DMODEL],
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[cur_batch_seq_len, num_kv_heads, Lk],
        strides=[stride_kbs, stride_kh, 1],
        block_shape=[BLOCK_N, 1, BLOCK_DMODEL],
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[cur_batch_seq_len, num_kv_heads, Lk],
        strides=[stride_vbs, stride_vh, 1],
        block_shape=[BLOCK_N, 1, BLOCK_DMODEL],
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[cur_batch_seq_len, num_q_heads, Lk],
        strides=[stride_obs, stride_oh, 1],
        block_shape=[BLOCK_M, 1, BLOCK_DMODEL],
    )

    q_row0 = start_m * BLOCK_M
    # OOB rows/head-dim lanes are zero-filled by the descriptor; the seq-len and
    # head-dim tail masks the original applied here are redundant.
    q = q_desc.load([q_row0, cur_head, 0]).reshape([BLOCK_M, BLOCK_DMODEL])

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # Calculate the end position for attention computation
    end_n = cur_batch_seq_len
    # Apply causal attention pruning
    end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M) if IS_CAUSAL else end_n

    start_n_limit = 0
    end_n_limit = block_mask * end_n

    for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
        # -- prepare attention mask ----
        # These are *compute* masks (causal / sliding-window / valid-position),
        # not tail masks. The descriptor handles OOB fill; these encode the
        # attention semantics and are kept verbatim from the original.
        pos_q = offs_m[:, None]  # Query positions [BLOCK_M, 1]
        pos_k = start_n + offs_n[None, :]  # Key positions [1, BLOCK_N]

        # Valid sequence mask
        mask = pos_k < cur_batch_seq_len
        # Causal mask
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        # Bidirectional sliding window masks
        sliding_mask_q = (
            pos_q - pos_k <= SLIDING_WINDOW_Q if SLIDING_WINDOW_Q > 0 else None
        )
        sliding_mask_k = (
            pos_k - pos_q <= SLIDING_WINDOW_K if SLIDING_WINDOW_K > 0 else None
        )
        if sliding_mask_q is not None:
            mask &= sliding_mask_q
        if sliding_mask_k is not None:
            mask &= sliding_mask_k

        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # Descriptors require the last dim contiguous, so K is loaded as
        # (BLOCK_N, BLOCK_DMODEL) and transposed before the dot (the original
        # loaded it pre-transposed via strides).
        k_tile = k_desc.load([start_n, cur_kv_head, 0]).reshape(
            [BLOCK_N, BLOCK_DMODEL]
        )
        k = tl.trans(k_tile)

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = v_desc.load([start_n, cur_kv_head, 0]).reshape([BLOCK_N, BLOCK_DMODEL])
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    acc = acc.to(Out.dtype.element_ty)
    # The store clamps at shape=[cur_batch_seq_len, ...], so the partial tail
    # row tile never writes past the sequence — no output mask needed.
    o_desc.store([q_row0, cur_head, 0], acc.reshape([BLOCK_M, 1, BLOCK_DMODEL]))
