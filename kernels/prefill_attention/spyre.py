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
    stride_qbs,
    stride_kbs,
    stride_vbs,
    stride_obs,
    stride_mm,
    SEQ: tl.constexpr,
    SEQ_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STICK: tl.constexpr,
    Lk: tl.constexpr,
):
    """Flash-attention prefill for a single (request, head) — batch == 1, one head.

    Spyre-aware form with live ``tl.spyre_tensor_layout`` markers that lower
    through ``RewriteDescriptorLayout``:

    - **2D descriptors** ``[SEQ, Lk]`` for one head (the wrapper slices each head
      on the host and launches per head). No head dim → no ``tt.reshape`` between
      a marked load and ``tl.dot`` (rejected by the pass); rebasing per head would
      need ``tt.addptr`` into a descriptor, which the backend does not lower.
    - **QK^T.** ``Q`` is stick-tiled on the contraction axis ``Lk``; ``K`` is left
      unmarked and transposed with ``tl.trans``. A marked operand may not reach
      ``tl.dot`` through a ``tt.trans``, but ``tl.trans`` on an unmarked (logical)
      load passes through Phase 2 as a scratchpad operand. A bare ``tl.dot(q, k)``
      with both ``[*, Lk]`` fails the frontend's equal-reduction-dim check.
    - **P·V.** The softmax result ``p`` is a descriptor-less logical intermediate
      that flows straight into ``tl.dot(p, v)``; ``V`` is marked (stick-on-Lk).
    - No ``tl.arange``: masking is the additive host tensor ``Mask[SEQ, SEQ_KV]``.
    - ``SEQ`` is a compile-time constant, so no per-batch scalar loads.
    - Distribution loop over the query-block work space (any core count).
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    m_blocks = tl.cdiv(SEQ, BLOCK_M)
    work_per_core = tl.cdiv(m_blocks, num_cores)
    work_start = pid * work_per_core
    work_end = tl.minimum(work_start + work_per_core, m_blocks)

    # 2D logical descriptors over (seq, head_dim) for this head. Lk is the
    # contiguous last dim (>= 16 bytes) and the matmul contraction axis.
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[SEQ, Lk], strides=[stride_qbs, 1], block_shape=[BLOCK_M, Lk],
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[SEQ, Lk], strides=[stride_kbs, 1], block_shape=[BLOCK_N, Lk],
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[SEQ, Lk], strides=[stride_vbs, 1], block_shape=[BLOCK_N, Lk],
    )
    o_desc = tl.make_tensor_descriptor(
        Out, shape=[SEQ, Lk], strides=[stride_obs, 1], block_shape=[BLOCK_M, Lk],
    )
    # Additive attention mask (0.0 valid / -1e8 masked), precomputed on the host
    # so no tl.arange is needed. The -1e8 sentinel is finite, so a fully-masked
    # tile yields exp(-1e8 - m) ~= 0, not exp(nan). The key axis SEQ_KV pads SEQ
    # to a BLOCK_N multiple with -1e8 columns: the descriptor zero-fills OOB lanes
    # and 0.0 means "attend", so a ragged final key tile must be masked explicitly.
    m_desc = tl.make_tensor_descriptor(
        Mask, shape=[SEQ, SEQ_KV], strides=[stride_mm, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    # Physical stick layout (Proposal 2): Q/V/O stick-tiled on head_dim Lk
    # (logical dim 1); STICK = 128 // dtype_bytes. K is NOT marked (its tl.trans
    # for QK^T must stay a logical operand); the mask is NOT marked (it is an
    # elementwise addend to the rank-2 qk, not a matmul operand). Markers must be
    # inline literals.
    tl.spyre_tensor_layout(q_desc, [(1, "floordiv", STICK), 0, (1, "mod", STICK)])
    tl.spyre_tensor_layout(v_desc, [(1, "floordiv", STICK), 0, (1, "mod", STICK)])
    tl.spyre_tensor_layout(o_desc, [(1, "floordiv", STICK), 0, (1, "mod", STICK)])

    for work in range(work_start, work_end):
        q_row0 = work * BLOCK_M

        # [BLOCK_M, Lk]. OOB rows/lanes zero-filled by the shape.
        q = q_desc.load([q_row0, 0])

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, Lk], dtype=tl.float32)

        # Every core walks the whole key range; the additive mask zeroes
        # contributions past the boundary, so the loop is partition-independent.
        for start_n in range(0, SEQ, BLOCK_N):
            k = k_desc.load([start_n, 0])          # [BLOCK_N, Lk] (unmarked)

            # QK^T: K is unmarked, so tl.trans(k) is a logical operand the layout
            # pass passes through; Q is marked stick-on-Lk.
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk += m_desc.load([q_row0, start_n])

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            # Natural exp (not exp2): the wrapper passes sm_scale WITHOUT the
            # 1/ln2 factor, so exp(sm_scale * qk) == exp2((sm_scale/ln2) * qk).
            # ktir-cpu implements math.exp but not math.exp2.
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v = v_desc.load([start_n, 0])          # [BLOCK_N, Lk]
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc)                # p is a logical intermediate
            m_i = m_ij

        acc = acc / l_i[:, None]
        acc = acc.to(Out.dtype.element_ty)
        o_desc.store([q_row0, 0], acc)
