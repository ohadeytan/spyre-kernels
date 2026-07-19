# prefill_attention conversion notes

## Tensor-descriptor conversion

- Source: original.py → tensor_descriptor.py (kernel `_prefill_attention_kernel_td`)
- Raw pointer arithmetic (`off_q`/`off_k`/`off_v`/`off_o`, `k_ptrs`/`v_ptrs`
  advanced per iteration) replaced with 3D `tl.make_tensor_descriptor` views over
  the packed `(seq_len, heads, head_dim)` layout. Each descriptor is rebased at
  `cur_batch_in_all_start_index * stride_*bs` so its coordinates run
  `0..cur_batch_seq_len`; the batch start is a scalar folded into the base
  pointer, not a per-row runtime index, so plain `desc.load`/`desc.store` suffice
  (no gather/scatter needed).
- **Tail masks dropped:** the original's seq-len (`offs_m < seq_len`,
  `pos_k < seq_len` on the load) and head-dim (`mask_d = offs_d < Lk`) masks on
  Q/K/V loads and the O store are redundant — the descriptor `shape` carries both
  boundaries and zero-fills OOB. Zero is the additive identity for the `tl.dot`
  accumulation and for masked-out `qk` (overwritten by the causal `tl.where`), so
  the drop is safe.
- **Compute masks kept:** causal / sliding-window / valid-position masking
  (`tl.where(mask, qk * sm_scale, -1.0e8)`) is attention *semantics*, not a tail
  fill, so it is preserved verbatim.
- K is loaded as `(BLOCK_N, BLOCK_DMODEL)` and transposed via `tl.trans` before
  `tl.dot(q, k)`. Descriptors require the last dim contiguous, so K cannot be
  loaded pre-transposed the way the original did via `strides=(1, stride_kbs)`.
- Last-dim (§4) is satisfied: the contiguous axis is `head_dim` (`BLOCK_DMODEL`),
  ≥ 16 bytes; the length-1 head axis is the middle dim.
- **Signature change:** added `num_q_heads` / `num_kv_heads` runtime args (needed
  for the descriptor `shape`, which the original never materialized). The wrapper
  passes them only when the target kernel declares them (`arg_names` check) and
  registers the TMA allocator, so it still drives the original kernel unchanged.

## Spyre-aware conversion

- Source: tensor_descriptor.py → spyre.py (kernel `_prefill_attention_kernel_spyre`).
  Wrapper entry point: `context_attention_fwd_spyre`. Lowering driver:
  `lower.py` (`VARIANTS["spyre"]`) → `spyre.ktir`.
- **Goal (both achieved): live `tl.spyre_tensor_layout` markers that lower through
  `RewriteDescriptorLayout`, AND runs on ktir-cpu.** The marked kernel is
  spyre-build-only (stock PyPI Triton has no `tl.spyre_tensor_layout`), so its
  numerical oracle is ktir-cpu (tests/ktir/, vs a NumPy reference) — matching the
  CI tier model (GPU tier = `td` kernels; ktir-cpu tier = lowered spyre kernels).
- **Live markers (Proposal 2).** Q/V/O carry `tl.spyre_tensor_layout` stick-tiling
  the head dim (logical dim 2 of the 3D `[HEADS, SEQ, DMODEL]` descriptors):
  `[(2,"floordiv",STICK), 0, 1, (2,"mod",STICK)]`, `STICK = 128 // dtype_bytes`. The
  head (logical dim 0) and SEQ (logical dim 1) are *identity* dims (bare `0` / `1`),
  so the physical layout is `[DMODEL//STICK, HEADS, SEQ, DMODEL%STICK]`. The pass
  physicalizes the marked loads and orients them into canonical matmul form; three
  constraints shape the kernel:
  - **QK^T: K is NOT marked, and is transposed with `tl.trans(k, (0, 2, 1))`** (the
    trailing two dims, leaving the head/batch axis in place). A marked operand may
    not reach `tl.dot` through a `tt.trans` (the pass rejects it), but `tl.trans` on
    an *unmarked* (logical) load is a canonical scratchpad operand the pass passes
    through. A bare `tl.dot(q, k)` with both `[*, *, DMODEL]` fails the frontend's
    equal-reduction-dim check, and a transposed descriptor *view* is frontend-
    rejected (`make_tensor_descriptor` requires last stride == 1).
  - **P·V: `p` is a descriptor-less logical intermediate** (the softmax result) fed
    straight into `tl.dot(p, v)`; V is marked (the pass's scratchpad passthrough).
  - **The mask is NOT marked.** It is an elementwise addend to the logical `qk`, not
    a matmul operand — marking it physicalizes the load to a stick tile that fails to
    add to `qk` (`arith.addf` type mismatch). It is head-independent, so a single 2D
    `[SEQ, SEQ_KV]` view is broadcast over the length-1 head axis
    (`m_desc.load(...)[None, :, :]`).
- **All heads in one launch (batched matmul, 3D descriptors).** The descriptors are
  3D `[HEADS, SEQ, DMODEL]` (K/V: `[KV_HEADS, …]`) with the head as the *batch* axis
  (logical dim 0, an identity dim in each marker). `tl.dot` over the trailing two
  dims lowers to `linalg.batch_matmul` via the pass's `dispatchBatchMatmul`. The
  distribution loop flattens the `(head, query-block)` work space, so one launch
  covers every head with no host per-head slicing. Answers issue #3 Q1/Q3.
- **KTIR-runnability constraints:** descriptors only (no `tt.addptr` on args →
  `batch==1`, `SEQ` constexpr, no `B_Seqlen`/`B_Start_Loc` scalar loads); no
  `tl.arange` (masking is the additive host tensor `Mask[SEQ, SEQ_KV]`, 0.0 / −1e8,
  added to the scaled scores); `tl.exp` not `exp2` (ktir-cpu has `math.exp`, not
  `math.exp2`) with a plain `sm_scale` (no `1/ln2`) — `exp(s·qk) == exp2((s/ln2)·qk)`.
  - **Ragged-tail mask pad:** the key axis is `SEQ_KV = cdiv(SEQ,BLOCK)*BLOCK` with
    −1e8 pad columns. A descriptor zero-fills OOB lanes and 0.0 means "attend" in an
    additive mask, so without the pad a partial final key tile would let
    out-of-range keys (zero-filled K/V) leak in.
- **Invariants:** distribution loop over the `(head, query-block)` work space
  (`HEADS * cdiv(SEQ, BLOCK_M)` items) via `tl.program_id`/`tl.num_programs` +
  `cdiv`/`minimum` — a flat `[num_cores]` grid, partition-invariant for any core
  count. Fixed constexpr tiles.
- **Signature:** `Q/K/V/Mask/Out` + per-head strides (head + row stride for each of
  Q/K/V/O, plus `stride_mm` for the mask row); constexprs `HEADS` / `KV_HEADS` /
  `SEQ` / `SEQ_KV` / `BLOCK_M` / `BLOCK_N` / `STICK` / `Lk` / `DMODEL`. Causal and
  sliding-window are encoded in the host mask (not signature args); the mask carries
  no head stride (head-independent).
- **Tests:** `tests/ktir/test_prefill_attention.py` on ktir-cpu vs a NumPy
  attention reference applied per head and stacked — causal, zeros, moderate-scale
  rescaling, per-head equivalence (batched matmul does not mix heads),
  **sliding-window** (backward / forward / band, host-mask only), **GQA**
  (`spyre_gqa`, 4q/2kv), **head-dim padding** (`spyre_pad`, `Lk=48 → DMODEL=64`),
  **fixed batch** (`spyre_batch`, B=2×4 heads folded), a `linalg.batch_matmul`-present
  assertion (no surviving `tt.dot` / `spyre_tensor_layout`), and **distribution
  invariance** (grids `[1, 4, 16, 32]`, `spyre_dist_c*`, SEQ=256 two-query-block ×
  HEADS — must match the reference and agree bitwise). No GPU test: the marked kernel
  cannot run on stock PyPI Triton, and the `td` kernel's GPU test covers the
  descriptor logic against the vLLM reference.
- **Dep pins:** torch-spyre/triton PR #19 (`a6938df5`, `dispatchBatchMatmul` — open
  PR branch) and ktir-cpu `78a05609` (main; includes the #149 `outs`-materialization fix).

### Capability ledger vs. `original.py` / `td`

`td` (`tensor_descriptor.py`) is the faithful descriptor port of `original.py` and
defines "no dropped capabilities." Each row is **kept** or a **compiler gap**
(blocked in the toolchain, not by authoring choice).

| Capability | Status | Notes |
|---|---|---|
| Online-softmax flash core | **kept** | identical loop; `exp` not `exp2` (ktir-cpu). |
| Multi-head, one launch | **kept** | head = batch axis → `linalg.batch_matmul`. |
| Causal mask | **kept** | host additive mask. |
| Sliding-window (Q & K) | **kept** | host mask, no kernel change — `build_additive_mask` encodes `pos_q−pos_k ≤ W_Q` / `pos_k−pos_q ≤ W_K`. |
| GQA (`num_q_heads ≠ num_kv_heads`) | **kept** | K/V descriptors carry `KV_HEADS` on the batch axis; the kernel reads KV head `head // kv_group_num`. Head-sharing is a descriptor **load index** (`→ arith.divsi`), not a matmul batch dim, so a 4-vs-2 head mismatch is one `batch_matmul` over length-1 tiles. `KV_HEADS == HEADS` → plain MHA. |
| Head dim `Lk` not a stick multiple (48/80/96) | **kept** (host contract) | logical `Lk` in the descriptor `shape`; `block_shape`/marker/tiles use the padded `DMODEL` (one full `STICK`, power of two, `≥ Lk`). The head pad rides the matmul reduction axis (not free zero-fill), so the **host must allocate Q/K/V/O with the head dim padded to `DMODEL` and zeroed**. Valid only while `DMODEL == STICK` (one stick). |
| Fixed multi-request batch, uniform seqlen | **kept** (head-axis fold) | a batch of `B` uniform-length requests = `B*HEADS` independent problems, so the host lays Q/K/V/O out as `[B*HEADS, SEQ, Lk]` and the unchanged kernel runs it. Correct only when the mask is shared across requests (uniform-length prefill). |
| Head dim spanning > 1 stick (fp32 `Lk=64`, `Lk > 64`) | **compiler gap** | the pass "D3 guard" (`dispatchSource`) rejects `physBlock > 1` on a *parallel* floor dim — V's/O's head dim in P·V (*"does not yet support multi-stick parallel floor dims"*). Reduction-axis multi-stick (QK^T) is fine. |
| Native 4D `[B, HEADS, SEQ, Lk]` batch descriptor | **compiler gap** | `LowerDescriptorMemory` rejects the collapsed-leading-dim `tt.dot` (`'ktdp.load' access tile shape must match result tensor shape`); the pass has no two-batch-dim matmul handler (only `BatchMatmulOp`, one batch dim). Use the head-axis fold instead. |
| Variable-length / packed ragged batch (`B_Start_Loc` / `B_Seqlen`) | **compiler gap** | per-request base rebasing needs `tt.addptr` into a `make_tensor_descriptor` base — the `bmm_addptr` fixtures are `"disabled"` (`test_lower_desc_memory.py::TestAddptrIntoDescriptor`). Dynamic extents themselves lower (`bmm_dynamic`); only the ragged/packed offset and the runtime seqlen scalar load are blocked. |

**Bottom line:** relative to `td`, Proposal 2 keeps the flash core, multi-head,
causal, sliding-window, GQA, single-stick padded head dims, and fixed uniform-length
batch — with no Proposal-1 technique (logical descriptors + markers throughout, no
hand-authored physical layout / reshape). The remaining gaps — multi-stick parallel
head dims, native 4D batch descriptors, and ragged variable-length batching — are
all compiler-bound, not authoring choices.
