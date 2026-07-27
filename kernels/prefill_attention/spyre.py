# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Spyre-aware conversion of _fwd_kernel (prefill attention).
# Original: kernels/prefill_attention/original.py
# Changes summarized in kernels/prefill_attention/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _prefill_attention_kernel_spyre(
    Q,
    K,
    V,
    Mask,
    sm_scale,
    Out,
    SEQ,      # runtime i32: query/key length of this request
    SEQ_KV,   # runtime i32: padded key length = cdiv(SEQ, BLOCK_N) * BLOCK_N
    stride_qh,
    stride_qbs,
    stride_kh,
    stride_kbs,
    stride_vh,
    stride_vbs,
    stride_oh,
    stride_obs,
    stride_mm,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STICK: tl.constexpr,
    Lk: tl.constexpr,
    DMODEL: tl.constexpr,
):
    """Flash-attention prefill for a single request, all heads — batch == 1.

    Spyre-aware form with live ``tl.spyre_tensor_layout`` markers that lower
    through ``RewriteDescriptorLayout``:

    - **3D descriptors** ``[HEADS, SEQ, DMODEL]`` (K/V: ``[KV_HEADS, …]``) with the
      head as the *batch* axis (logical dim 0). ``tl.dot`` over the trailing two
      dims lowers to ``linalg.batch_matmul`` via the pass's ``dispatchBatchMatmul``
      (PR #19). No host per-head slicing / launch: one launch covers every head.
      The head is an *identity* dim in each marker (a bare ``0``), so it passes
      through the physicalization untouched.
    - **Variable length.** ``SEQ`` (and its key-padded ``SEQ_KV``) are **runtime
      i32 args**, so one lowered kernel serves any per-request length. ``SEQ``
      feeds the **dynamic descriptor extent** ``shape=[HEADS, SEQ, Lk]`` (the
      sequence axis is a ``memref<…x?x…>`` kDynamic dim; strides and
      ``block_shape`` stay compile-time constant, as the pass requires) and the
      **runtime KV-loop bound** ``range(0, SEQ, BLOCK_N)`` / ``m_blocks =
      cdiv(SEQ, BLOCK_M)`` (an ``scf.for`` with a runtime trip count). The length
      arrives as an arg (the caller reads ``B_Seqlen[req]`` on the host) rather
      than a ``tl.load(B_Seqlen + req)`` scalar load, whose rank-0 view the
      ktir-cpu oracle does not execute. A packed/ragged base offset (``tt.addptr``
      into a descriptor base) is a compiler gap.
    - **GQA.** ``KV_HEADS`` may be fewer than ``HEADS`` (grouped-query attention):
      each query head ``h`` shares the KV head ``h // (HEADS // KV_HEADS)``. The
      sharing is resolved at the descriptor *load index* (``k_desc.load([kv_head,
      …])``), not at the matmul's batch dim — the pass classifies each operand
      independently, so a 4-vs-2 head mismatch produces one ``batch_matmul`` over
      length-1 head tiles with no equal-batch requirement. ``KV_HEADS == HEADS``
      (the multi-head-attention case) makes the group size 1.
    - **Head-dim padding.** The logical head dim ``Lk`` need not be a stick
      multiple (e.g. 48/80/96). Descriptors carry the logical ``Lk`` in ``shape``
      (so masking/OOB is exact), but every ``block_shape``/marker/tile uses the
      padded ``DMODEL`` (a power of two, one full ``STICK`` wide, ``>= Lk``). Unlike
      OOB *seq* lanes, the head-dim pad rides the matmul reduction axis, so the
      **host must allocate Q/K/V/O with the head dim padded to ``DMODEL`` and the
      pad zeroed** (zeros contribute nothing to the QK^T / P·V contractions).
      ``DMODEL == Lk`` (already a stick multiple) is the no-padding case.
    - **QK^T.** ``Q`` is stick-tiled on the contraction axis ``DMODEL``; ``K`` is
      left unmarked and transposed on its trailing two dims with ``tl.trans``. A
      marked operand may not reach ``tl.dot`` through a ``tt.trans``, but
      ``tl.trans`` on an unmarked (logical) load passes through Phase 2 as a
      scratchpad operand. A bare ``tl.dot(q, k)`` with both ``[*, *, DMODEL]`` fails
      the frontend's equal-reduction-dim check.
    - **P·V.** The softmax result ``p`` is a descriptor-less logical intermediate
      that flows straight into ``tl.dot(p, v)``; ``V`` is marked (stick-on-DMODEL).
    - No ``tl.arange``: masking is the additive host tensor ``Mask[SEQ, SEQ_KV]``
      (broadcast across heads — the causal/validity mask is head-independent).
    - Distribution loop over the (head, query-block) work space (any core count).
    - **Fixed multi-request batch** (uniform sequence length) folds into the head
      axis: a batch of ``B`` requests is ``B * HEADS`` independent attention
      problems, so the host lays Q/K/V/O out as ``[B*HEADS, SEQ, Lk]`` and this
      kernel runs unchanged (a request is just more "heads"). Correct only when the
      mask is shared across requests (true for uniform-length prefill).
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    m_blocks = tl.cdiv(SEQ, BLOCK_M)
    # Flatten (head, query-block) into one work space so the distribution loop
    # stays a flat [num_cores] grid, partition-invariant for any core count.
    total_work = HEADS * m_blocks
    work_per_core = tl.cdiv(total_work, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, total_work)

    # GQA group size: each query head h shares KV head h // kv_group_num.
    # kv_group_num == 1 when KV_HEADS == HEADS (multi-head attention).
    kv_group_num: tl.constexpr = HEADS // KV_HEADS

    # 3D logical descriptors over (head, seq, head_dim). The head-dim `shape` is
    # the LOGICAL Lk (so masking / OOB is exact), but the `block_shape` (and the
    # marker/tiles below) use the padded DMODEL — a power of two, one full STICK
    # wide, >= Lk. When DMODEL == Lk this is the no-padding case. Lk/DMODEL is the
    # matmul contraction axis; HEADS (K/V: KV_HEADS) is the batched-matmul axis.
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[HEADS, SEQ, Lk], strides=[stride_qh, stride_qbs, 1],
        block_shape=[1, BLOCK_M, DMODEL],
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[KV_HEADS, SEQ, Lk], strides=[stride_kh, stride_kbs, 1],
        block_shape=[1, BLOCK_N, DMODEL],
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[KV_HEADS, SEQ, Lk], strides=[stride_vh, stride_vbs, 1],
        block_shape=[1, BLOCK_N, DMODEL],
    )
    o_desc = tl.make_tensor_descriptor(
        Out, shape=[HEADS, SEQ, Lk], strides=[stride_oh, stride_obs, 1],
        block_shape=[1, BLOCK_M, DMODEL],
    )
    # Additive attention mask (0.0 valid / -1e8 masked), precomputed on the host
    # so no tl.arange is needed. Head-independent, so a single [SEQ, SEQ_KV] view
    # is broadcast across heads. The -1e8 sentinel is finite, so a fully-masked
    # tile yields exp(-1e8 - m) ~= 0, not exp(nan). The key axis SEQ_KV pads SEQ
    # to a BLOCK_N multiple with -1e8 columns: the descriptor zero-fills OOB lanes
    # and 0.0 means "attend", so a ragged final key tile must be masked explicitly.
    m_desc = tl.make_tensor_descriptor(
        Mask, shape=[SEQ, SEQ_KV], strides=[stride_mm, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    # Physical stick layout (Proposal 2): Q/V/O stick-tiled on the padded head_dim
    # DMODEL (logical dim 2); STICK = 128 // dtype_bytes, DMODEL a single stick. The
    # head axis (logical dim 0) is an identity dim (bare 0) and SEQ (logical dim 1)
    # is identity (bare 1), so the physical layout is [DMODEL//STICK, HEADS, SEQ,
    # DMODEL%STICK]. K is NOT marked (its tl.trans for QK^T must stay a logical
    # operand); the mask is NOT marked (it is an elementwise addend to the rank-3
    # qk, not a matmul operand). Markers must be inline literals.
    tl.spyre_tensor_layout(q_desc, [(2, "floordiv", STICK), 0, 1, (2, "mod", STICK)])
    tl.spyre_tensor_layout(v_desc, [(2, "floordiv", STICK), 0, 1, (2, "mod", STICK)])
    tl.spyre_tensor_layout(o_desc, [(2, "floordiv", STICK), 0, 1, (2, "mod", STICK)])

    for work in range(work_start, work_end):
        head = work // m_blocks
        m_blk = work % m_blocks
        q_row0 = m_blk * BLOCK_M
        # GQA: query head `head` reads the KV head it shares (identity when 1:1).
        kv_head = head // kv_group_num

        # [1, BLOCK_M, DMODEL]. OOB rows and head-dim pad lanes zero-filled by the
        # shape (the host also zeros the physical head-dim pad — see the docstring).
        q = q_desc.load([head, q_row0, 0])

        m_i = tl.zeros([1, BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([1, BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([1, BLOCK_M, DMODEL], dtype=tl.float32)

        # Every core walks the whole key range; the additive mask zeroes
        # contributions past the boundary, so the loop is partition-independent.
        for start_n in range(0, SEQ, BLOCK_N):
            k = k_desc.load([kv_head, start_n, 0])  # [1, BLOCK_N, DMODEL] (unmarked)

            # QK^T (batched over the head axis): K is unmarked, so tl.trans(k)
            # (transpose of the trailing two dims) is a logical operand the layout
            # pass passes through; Q is marked stick-on-DMODEL. Result [1, BLOCK_M,
            # BLOCK_N].
            qk = tl.dot(q, tl.trans(k, (0, 2, 1))) * sm_scale
            # Broadcast the head-independent mask over the length-1 head axis.
            qk += m_desc.load([q_row0, start_n])[None, :, :]

            m_ij = tl.maximum(m_i, tl.max(qk, 2))
            qk -= m_ij[:, :, None]
            # Natural exp (not exp2): the wrapper passes sm_scale WITHOUT the
            # 1/ln2 factor, so exp(sm_scale * qk) == exp2((sm_scale/ln2) * qk).
            # ktir-cpu implements math.exp but not math.exp2.
            p = tl.exp(qk)
            l_ij = tl.sum(p, 2)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]

            v = v_desc.load([kv_head, start_n, 0])  # [1, BLOCK_N, DMODEL]
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)                # p is a logical intermediate
            m_i = m_ij

        acc = acc / l_i[:, :, None]
        acc = acc.to(Out.dtype.element_ty)
        o_desc.store([head, q_row0, 0], acc)
