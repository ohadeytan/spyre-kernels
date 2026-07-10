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

- Source: tensor_descriptor.py → spyre.py (kernel `_prefill_attention_kernel_spyre`)
- **KTIR-lowerable, descriptors-only, no `tl.arange`.** The Spyre KTIR backend
  lowers every `!tt.ptr` arg to a memory view (so no `tt.addptr` on a pointer
  arg) and does **not** lower `tt.make_range` (emitted by `tl.arange`) to a
  ktdp/linalg form. ktir-cpu implements no `tt.*` ops, so any surviving
  `tt.make_range` blocks execution. Both constraints are met here: memory is
  addressed purely through `make_tensor_descriptor`, and the causal/validity
  mask is **precomputed on the host** and passed as an additive `MASK[SEQ, SEQ]`
  tensor (0.0 allowed, −inf masked), loaded per (q-block, kv-block) tile and
  added to the scores. No in-kernel position vectors. Confirmed against
  `torch-spyre/spyre-kernels`: only `tl.arange`-free kernels (`rms_norm`,
  `paged_attn`) round-trip to executable KTIR; the `paged_attn` spyre variant
  uses the same host-data-not-positions approach (a SLOTS gather there, an
  additive mask here). Verified: generated `spyre.ktir` has zero `tt.*` ops and
  passes on ktir-cpu (`tests/ktir/test_prefill_attention.py`).
- **Distribution loop (grid-fits-32-cores):** the original 3D GPU grid
  `(batch, head, m_block)` is flattened into one linear work space
  `batch * num_q_heads * num_m_blocks` and distributed over a fixed
  `NUM_CORES=32` grid with `tl.program_id(0)` / `tl.num_programs(0)` and a
  `work_per_core = cdiv(total, num_cores)` loop (`tl.minimum` on the end bound).
  Each iteration decodes one `(cur_head, start_m)` work item. The result is
  distribution-invariant (bitwise across `num_cores`; verified on GPU). This
  matches the KB's `Q_tiles × CORES_PER_Q_TILE ≤ 32` constraint (one Q-tile per
  core). The distribution loop, the dynamic KV loop, and the physical stick
  layout all lower cleanly (probed) — `tl.arange` was the sole `tt.make_range`
  source.
- **Single-request KTIR config (`batch == 1`).** The per-request seq_len is the
  static `SEQ` constexpr, so there are **no** `B_Seqlen`/`B_Start_Loc` scalar
  loads (indexing those with a runtime `cur_batch` needs an `tl.arange` one-hot,
  which reintroduces `tt.make_range`; arange-free lane extraction — `[:, 0:1]`
  slicing — is unsupported in this Triton). The packed descriptors + distribution
  loop keep the general `(head, m_block)` structure, so the GPU launch still
  drives any `num_cores`; the frozen KTIR is the `batch = 1` case.
- **Physical stick layout (Proposal 1):** head_dim `Lk` is the stick-tiled
  innermost dim, factored `Lk -> (Lk//S, S)` with `S = 128 // dtype_bytes`
  (64 fp16/bf16, 32 fp32) passed as a constexpr — not hard-coded 64. Descriptors
  are 4D `[SEQ, heads, Lk//S, S]`, row-major strides `[stride_*bs, stride_*h, S, 1]`,
  block `[BLOCK_M|BLOCK_N, 1, BLOCK_DMODEL//S, S]`. The `(Lk//S, S)` stick pair is
  adjacent + innermost, so one `reshape` collapses each loaded tile to the logical
  `[BLOCK_*, BLOCK_DMODEL]` for `tl.dot`, and the f32 accumulator is reshaped back
  to `[BLOCK_M, 1, BLOCK_DMODEL//S, S]` before the store. Assumes head_dim is a
  multiple of `S` (true for the 64/128 head dims in use; `next_power_of_2(Lk)`).
- **Layout vs. the device stick-layout spec (unsettled).** The device
  activation layout we were pointed at is `S × D/64 × B × 64` (seq, dim in
  64-stick groups, batch, 64-element stick) — "activations stickified in the
  output dim." We satisfy that core property: head_dim (the QK/PV output dim) is
  stickified with 64 innermost (`Lk//S, S`). But our **beyond-stick order differs**:
  we emit `[SEQ, heads, Lk//S, S]` (heads before the dim stick-groups, no batch
  axis — `batch=1`), whereas the spec puts `D/64` before the batch-like axis. For
  attention the per-matmul batch axis is naturally *heads*, so a faithful mapping
  is likely `S × D/64 × H × 64`, i.e. our `heads` and `Lk//S` axes transposed —
  but this is explicitly **not yet pinned down** (the source "was told" the spec
  and still needs to confirm; real layout work is deferred to `tl.inter_tile`,
  and today's device prefill attention is a fused kernel whose layouts change
  internally). Guidance for now is "assume the layout stays the same throughout,"
  so we keep `[SEQ, heads, Lk//S, S]` and revisit the ordering when `tl.inter_tile`
  lands. Weights (`Dout/64 × Din × 64`, input dim *not* stickified) do not apply
  here — attention has no weights; Q/K/V/scores are all activations.

- **Softmax:** natural `tl.exp` (not `exp2`), so `sm_scale = 1/sqrt(Lk)` carries
  no `RCP_LN2` factor. The online-softmax rescaling is unchanged from the flash
  form.
- **K transpose:** descriptors require a contiguous last dim, so K is loaded as
  `(BLOCK_N, BLOCK_DMODEL)` and `tl.trans`-posed before `tl.dot(q, k)` (as in the
  TD form; the original loaded it pre-transposed via strides).
- **Stripped GPU-only:** `tl.multiple_of(start_n, ...)` dropped. No autotune.
- **Signature:** `Q, K, V, MASK, Out, sm_scale`, then the eight q/k/v/o
  bs+h strides, `num_q_heads, num_kv_heads, batch, num_m_blocks`; constexprs
  `kv_group_num, BLOCK_M, BLOCK_DMODEL, BLOCK_N, Lk, S, SEQ`. Launched by
  `context_attention_fwd_spyre` in wrapper.py (builds a default causal `MASK` and
  drives `num_cores`); the original and TD kernels keep `context_attention_fwd`.
