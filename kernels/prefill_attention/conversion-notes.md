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
  head_dim `Lk` (logical dim 1): `[(1,"floordiv",STICK), 0, (1,"mod",STICK)]`,
  `STICK = 128 // dtype_bytes`. These lower through the pass (Phase 1 physicalizes
  the loads; Phase 2 slices/orients into canonical matmul form). Getting them to
  lower turned on three non-obvious facts, each probed against the pass:
  - **QK^T: K is NOT marked, and is transposed with `tl.trans`.** A marked operand
    may not reach `tl.dot` through a `tt.trans` — the pass rejects the glue
    ("source op operand is neither a physical load nor a logical tensor of the
    expected rank"). But `tl.trans` on an *unmarked* (logical) load is fine: Phase 2
    treats it as a canonical scratchpad operand and passes it through. A bare
    `tl.dot(q, k)` with both `[*, Lk]` is not an option either — it fails the Triton
    frontend's equal-reduction-dim check (QK^T is a genuine logical transpose, not
    just a physical stick reorientation), and a transposed descriptor *view* is
    frontend-rejected (`make_tensor_descriptor` requires last stride == 1). So
    `tl.dot(q, tl.trans(k))` with K unmarked is the one lowerable form.
  - **P·V: `p` is a descriptor-less logical intermediate** (the softmax result),
    fed straight into `tl.dot(p, v)`; V is marked. This is the pass's scratchpad
    passthrough (doc Example 2's `bc`).
  - **The mask is NOT marked.** It is an elementwise addend to the rank-2 logical
    `qk`, not a matmul operand — marking it physicalizes the load to a rank-3 stick
    tile that then fails to add to the rank-2 `qk` (`arith.addf` type mismatch).
- **One head per launch (2D descriptors).** The marked form needs 2D per-head
  descriptors: a head *dimension* forces a `tt.reshape` between the marked load and
  `tl.dot` (rejected), and rebasing the base pointer per head (`Q + head*stride`)
  needs `tt.addptr` into a descriptor — a disabled backend gap
  (`TestAddptrIntoDescriptor`). So the wrapper slices each head on the host
  (`q[:, h]`, `k[:, kv_h]`, …) and launches the kernel per head; inside, `Q` is a
  plain 2D `[SEQ, Lk]` view. N launches, one per query head.
- **KTIR-runnability constraints** (shared with the prior descriptor-only form):
  descriptors only (no `tt.addptr` on args → freeze `batch==1`, `SEQ` constexpr, no
  `B_Seqlen`/`B_Start_Loc` scalar loads); no `tl.arange` (masking is the additive
  host tensor `Mask[SEQ, SEQ_KV]`, 0.0 / −1e8, added to the scaled scores); `tl.exp`
  not `exp2` (ktir-cpu has `math.exp`, not `math.exp2`) with a plain `sm_scale`
  (no `1/ln2`) — `exp(s·qk) == exp2((s/ln2)·qk)`.
  - **Ragged-tail mask pad:** the key axis is `SEQ_KV = cdiv(SEQ,BLOCK)*BLOCK` with
    −1e8 pad columns. A descriptor zero-fills OOB lanes and 0.0 means "attend" in an
    additive mask, so without the pad a partial final key tile would let
    out-of-range keys (zero-filled K/V) leak in.
- **Invariants:** distribution loop over this head's query-blocks
  (`cdiv(SEQ, BLOCK_M)` items) via `tl.program_id`/`tl.num_programs` + `cdiv`/`minimum`
  — a flat `[num_cores]` grid, partition-invariant for any core count. Fixed
  constexpr tiles. `@triton.autotune` / `tl.multiple_of` stripped.
- **Scratchpad batching:** none — each query-block already holds a dense
  `[BLOCK_M, Lk]` accumulator plus K/V/mask tiles.
- **Signature:** single-head — `Q/K/V/Mask/Out` + one row stride each + `stride_mm`;
  constexprs `SEQ` / `SEQ_KV` / `BLOCK_M` / `BLOCK_N` / `STICK` / `Lk`. Dropped the
  head args, `B_Start_Loc` / `B_Seqlen` / `IS_CAUSAL` / `SLIDING_WINDOW_*` (folded
  into the host mask / per-head launch).
- **Tests:** `tests/ktir/test_prefill_attention.py` on ktir-cpu vs a NumPy
  single-head attention reference — correctness (causal, zeros, moderate-scale
  rescaling) plus **distribution invariance**: the kernel lowered at grids
  `[1, 4, 16, 32]` (`lower.py` `spyre_dist_c*` variants, SEQ=256 two-query-block
  shape) must match the reference and agree bitwise across core counts. No GPU
  test: the marked kernel cannot run on stock PyPI Triton, and the `td` kernel's
  GPU test already covers the descriptor logic against the vLLM reference.
  - The two-query-block variants surfaced a ktir-cpu bug (in-place `linalg.matmul`
    mutating a loop-hoisted `outs` constant, so the QK^T accumulator never reset
    across KV tiles) — reported as torch-spyre/ktir-cpu#157, fixed by #149.
