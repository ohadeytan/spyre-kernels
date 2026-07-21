# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of prefill attention _fwd_kernel.
# Original: kernels/prefill_attention/original.py
# Changes summarized in kernels/prefill_attention/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _prefill_attention_kernel_spyre(
    Q,
    K,
    V,
    MASK,
    Out,
    sm_scale,
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
    batch,
    num_m_blocks,
    seqlen,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
    S: tl.constexpr,
    SEQ: tl.constexpr,
):
    """Flash-attention prefill over a packed (total_tokens, heads, head_dim)
    layout. Spyre-aware and KTIR-lowerable:

    - A ≤32-core distribution loop replaces the (batch, head, m_block) GPU grid;
      the work space is flattened and split over `tl.num_programs(0)` cores. The
      result is independent of the partition (distribution-invariant).
    - head_dim rides in the physical stick-tiled layout (D -> D//S, S) directly
      in the descriptor shapes. The descriptor carries the **physical (padded)**
      head dim `BLOCK_DMODEL` (a whole number of sticks); the logical head dim
      `Lk` may be smaller. When `Lk` is not a stick multiple, the host pads the
      physical head dim to `BLOCK_DMODEL` and zero-fills lanes `[Lk, BLOCK_DMODEL)`
      (on the reduction axis, so they must be real zeros). No-op when
      `BLOCK_DMODEL == Lk`.
    - **Descriptors only, no `tl.arange`.** The Spyre KTIR backend lowers every
      pointer arg to a memory view (no `tt.addptr`) and does not lower
      `tt.make_range` (from `tl.arange`) to a ktdp/linalg form. So: (a) memory is
      addressed purely through `make_tensor_descriptor`; (b) the causal /
      validity mask is **precomputed on the host** and passed as an additive
      `MASK[SEQ, SEQ]` tensor (0.0 allowed, -inf masked), loaded per tile through
      a descriptor and added to the scores — no in-kernel position vectors.

    Variable length: `SEQ` is the compile-time **padded max** (descriptor and
    mask extents); the runtime `seqlen` arg (<= SEQ) bounds the KV loop so each
    launch attends only over its true length. Passing `seqlen == SEQ` is the
    full-length case. seqlen is a plain i32 arg — NOT a `B_Seqlen`/`B_Start_Loc`
    scalar load (which would emit a rank-0 memory view ktir-cpu can't execute
    yet, or need an `tl.arange` one-hot → `tt.make_range`). The distribution loop
    and packed descriptors still carry the general (batch, head, m_block)
    structure, so the GPU launch drives any `num_cores`.
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # Flatten the (batch, head, m_block) work space and distribute it over the
    # <=32 cores. Each work item is one (cur_head, start_m) pair (batch == 1 in
    # the lowered config); the result is independent of the partition.
    total_work = batch * num_q_heads * num_m_blocks
    work_per_core = tl.cdiv(total_work, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, total_work)

    for work in range(work_start, work_end):
        start_m = work % num_m_blocks
        head_batch = work // num_m_blocks
        cur_head = head_batch % num_q_heads
        cur_kv_head = cur_head // kv_group_num

        # Physical stick layout: the head dim rides stick-tiled and innermost,
        # factored (D -> D//S, S). The descriptor `shape` carries the **physical
        # (padded) head dim** BLOCK_DMODEL, a whole number of sticks; the logical
        # head dim `Lk` may be smaller (Lk <= BLOCK_DMODEL). When Lk is not a
        # stick multiple the host pads the physical head dim to BLOCK_DMODEL and
        # zero-fills lanes [Lk, BLOCK_DMODEL): those lanes sit on the QK/PV
        # reduction axis, so they must be real zeros (they are NOT descriptor OOB
        # fill). Degenerate (no padding) when BLOCK_DMODEL == Lk. The (D//S, S)
        # stick pair is adjacent + innermost, so one reshape collapses each loaded
        # tile to the logical [BLOCK_*, BLOCK_DMODEL] for tl.dot. Row-major strides
        # over the physical shape [seq, heads, BLOCK_DMODEL//S, S] are
        # [stride_*bs, stride_*h, S, 1].
        q_desc = tl.make_tensor_descriptor(
            Q,
            shape=[SEQ, num_q_heads, BLOCK_DMODEL // S, S],
            strides=[stride_qbs, stride_qh, S, 1],
            block_shape=[BLOCK_M, 1, BLOCK_DMODEL // S, S],
        )
        k_desc = tl.make_tensor_descriptor(
            K,
            shape=[SEQ, num_kv_heads, BLOCK_DMODEL // S, S],
            strides=[stride_kbs, stride_kh, S, 1],
            block_shape=[BLOCK_N, 1, BLOCK_DMODEL // S, S],
        )
        v_desc = tl.make_tensor_descriptor(
            V,
            shape=[SEQ, num_kv_heads, BLOCK_DMODEL // S, S],
            strides=[stride_vbs, stride_vh, S, 1],
            block_shape=[BLOCK_N, 1, BLOCK_DMODEL // S, S],
        )
        o_desc = tl.make_tensor_descriptor(
            Out,
            shape=[SEQ, num_q_heads, BLOCK_DMODEL // S, S],
            strides=[stride_obs, stride_oh, S, 1],
            block_shape=[BLOCK_M, 1, BLOCK_DMODEL // S, S],
        )
        # Additive mask over (query row, key col). Loaded per (q-block, kv-block)
        # tile; encodes causal + validity semantics the original built from
        # tl.arange positions.
        m_desc = tl.make_tensor_descriptor(
            MASK,
            shape=[SEQ, SEQ],
            strides=[SEQ, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        q_row0 = start_m * BLOCK_M
        # OOB rows / head-dim lanes are zero-filled by the descriptor. Collapse
        # the stick pair to the logical [BLOCK_M, BLOCK_DMODEL] tile for tl.dot.
        q = q_desc.load([q_row0, cur_head, 0, 0]).reshape([BLOCK_M, BLOCK_DMODEL])

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # KV loop bound is the RUNTIME per-request seqlen (<= SEQ, the padded
        # max). Only whole BLOCK_N tiles up to seqlen are visited; the host mask
        # zeroes any keys within the last tile that overrun seqlen. When
        # seqlen == SEQ this is the full-length case (byte-identical to a static
        # SEQ bound). seqlen rides as an i32 arg, not a B_Seqlen scalar load, so
        # no rank-0 memory view is emitted (ktir-cpu-executable today).
        n_kv_blocks = tl.cdiv(seqlen, BLOCK_N)
        for kv_start in range(0, n_kv_blocks * BLOCK_N, BLOCK_N):
            # Descriptors require the last dim contiguous, so K is loaded as
            # (BLOCK_N, BLOCK_DMODEL) and transposed before the dot.
            k_tile = k_desc.load([kv_start, cur_kv_head, 0, 0]).reshape(
                [BLOCK_N, BLOCK_DMODEL]
            )
            k = tl.trans(k_tile)

            qk = tl.dot(q, k) * sm_scale
            # Add the host mask tile (0.0 / -inf) instead of computing positions.
            mask = m_desc.load([q_row0, kv_start])
            qk = qk + mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = v_desc.load([kv_start, cur_kv_head, 0, 0]).reshape(
                [BLOCK_N, BLOCK_DMODEL]
            )
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_ij

        acc = acc / l_i[:, None]
        acc = acc.to(Out.dtype.element_ty)
        # Reshape the 2D accumulator back to the physical stick layout to store.
        o_desc.store(
            [q_row0, cur_head, 0, 0],
            acc.reshape([BLOCK_M, 1, BLOCK_DMODEL // S, S]),
        )
