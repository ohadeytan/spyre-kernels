# `tl.spyre_tensor_layout` — physical stick-layout marker (Proposal 2)

> **Spyre-family only.** `tl.spyre_tensor_layout` is a Triton builtin from
> `torch-spyre/triton` PR #19; it is not in stock PyPI Triton, so KTIR generation
> must use the PR #19 build — see **Dependency** at the bottom.

## What it does

The descriptor stays **logical** (`shape`/`strides`/`block_shape` in math
dimensions). The marker declares how that tensor is **physically stick-tiled** in
device memory. The compiler's `RewriteDescriptorLayout` pass reads the marker and
synthesizes the physical loops (slice sticks, run `linalg.matmul` per tile,
accumulate). `tl.dot` and the descriptor are otherwise untouched — there is **no**
reshape glue (contrast the physical-descriptor variant).

## Syntax

```python
tl.spyre_tensor_layout(desc, [ <one entry per physical dim> ])
```

Entry forms (the OpSpec `device_coordinates` map):

| Entry | Meaning |
|-------|---------|
| `src` (bare int) | identity — this physical dim = logical dim `src`, unchanged |
| `(src, "floordiv", S)` | stick **index**: `logical[src] // S` |
| `(src, "mod", S)` | within-stick **lane**: `logical[src] % S` |

- **`src` is the logical dimension index** being addressed. The same-looking
  marker names a different axis on different operands: K is dim 1 of `A[M,K]` but
  dim 0 of `B[K,N]`.
- **`S = 128 // dtype_bytes`** — the DataStick is 128 bytes, so `S` is
  **64** (fp16/bf16), **32** (fp32), **128** (fp8). Never hard-code 64. In the PR
  fixtures this is `_sticksize = 128 // np.dtype(...).itemsize`.

## Layout convention

`stick-on-X` factors dimension `X` into `(X // S, X % S)` and the physical dim
order is **`[X//S, other, X%S]`** — stick index first, lane last:

```python
# stick-on-X  ->  [(X, "floordiv", S), other_logical, (X, "mod", S)]
tl.spyre_tensor_layout(a_desc, [(0, "floordiv", 64), 1, (0, "mod", 64)])  # A[M,K] stick-on-M
tl.spyre_tensor_layout(b_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])  # B[K,N] stick-on-N
tl.spyre_tensor_layout(c_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])  # C[M,N] stick-on-N
```

## Which matmul "case" you get (a consequence, not a choice)

The pass picks the loop structure from **which axis you stick-tile**, not from any
flag:

- **A stick-on-M (or B/C on N) → Case 1, parallel sticks.** No K reduction loop;
  one inner `linalg.matmul` per output stick. (The example above.)
- **A stick-on-K → Case 2, split-K reduction.** A's K-stick dim drives a reduction
  loop; B's K-flat dim is offset per stick. Marker: `[(1,"floordiv",S), 0, (1,"mod",S)]`
  on `A[M,K]` (K is logical dim 1).

## Batched matmul — an identity batch dim (N-D descriptors)

The marker is not limited to 2D. A descriptor may carry a **leading batch axis**
(or several) that the matmul iterates over independently — e.g. a per-head
attention or a stacked GEMM with descriptors `[BATCH, M, K]` / `[BATCH, K, N]`.
Make the batch axis an **identity dim** (a bare `src` int) so it passes through
the physicalization untouched, and stick-tile only the inner matmul axis:

```python
# A[BATCH, M, K] stick-on-K (contraction axis = logical dim 2); dim 0 is the batch
tl.spyre_tensor_layout(a_desc, [(2, "floordiv", S), 0, 1, (2, "mod", S)])
```

`tl.dot` over the **trailing two dims** of such operands lowers to
`linalg.batch_matmul` (the batch axis becomes the matmul's batch dim). The bare
`0` in the marker is what keeps the batch axis out of the stick factoring — the
stick index / lane entries still name the inner axis (`src = 2` above). This is
how one launch covers every batch/head with no host-side per-batch slicing.

- The batch extents of two operands **need not match**: a grouped case (e.g. GQA,
  where 4 query heads share 2 KV heads) resolves head-sharing at the descriptor
  **load index**, not at the matmul batch dim, so a 4-vs-2 mismatch is one
  `batch_matmul` over length-1 batch tiles — no equal-batch requirement.
- More than one leading batch dim (a *native* `[B, HEADS, …]` descriptor with two
  collapsed batch axes) is **not** currently dispatched — the pass has one batch
  dim. Fold extra batch axes into a single leading dim on the host instead.

## Mark matmul operands only — leave everything else logical

The pass physicalizes a marked descriptor into stick tiles and orients it into
canonical matmul form. That is only correct for **operands that flow into
`tl.dot`**. Three rules follow, all enforced by the pass (a violation is a
compile error or a type mismatch, not a silent bug):

- **A marked operand may not reach `tl.dot` through a transpose.** The pass
  rejects a marked load consumed via `tt.trans`. When a matmul needs one operand
  transposed (e.g. `K^T` in `Q·K^T`), leave that operand **unmarked** and
  transpose it with `tl.trans` — a transpose of an *unmarked* (logical) load is a
  scratchpad operand the pass passes through. (Only the trailing two dims are
  transposed; a batch axis stays in place: `tl.trans(k, (0, 2, 1))` for
  `[batch, N, K]`.)
- **Logical intermediates stay unmarked.** A value produced inside the kernel
  (e.g. a softmax result) has no descriptor and is not marked; it flows straight
  into the next `tl.dot` as a scratchpad operand. Only the *other* operand of that
  `tl.dot` (loaded from memory) carries a marker.
- **Elementwise addends are not matmul operands — do not mark them.** A tensor
  added to the matmul result (a bias, an additive mask) is not stick-tiled the
  same way; marking it physicalizes the load to a stick tile that fails to add to
  the logical result (`arith.addf` type mismatch). Leave it logical.

## Dynamic descriptor extent — a runtime size in `shape`

A descriptor extent may be a **runtime `i32` arg** rather than a `constexpr`, so
one lowered kernel serves any size along that axis (e.g. per-request sequence
length). The runtime value goes in **`shape`** only:

```python
q_desc = tl.make_tensor_descriptor(
    Q, shape=[HEADS, SEQ, Lk], strides=[stride_h, stride_s, 1],  # SEQ is a runtime i32
    block_shape=[1, BLOCK_M, DMODEL],                            # block_shape stays constexpr
)
```

The runtime axis lowers to a `?` (kDynamic) memref dim (`memref<…x?x…>`).
**`strides` and `block_shape` must stay compile-time constant** — only full
extents may be dynamic. A loop bounded by the runtime size (`range(0, SEQ,
BLOCK_N)`) lowers to an `scf.for` with a runtime trip count. Pass the size as an
ordinary arg; a scalar *load* of it (`tl.load(size_ptr + i)`) is a separate
concern with its own lowering caveats.

## Output descriptor drives the store sink

Marking the **output** descriptor (`C`) triggers the store **sink stage** — the
logical matmul result is scattered into the physical stick buffer via
`tensor.insert_slice`. Leave `C` unmarked and the store stays logical (sink is a
no-op).

## Inline-only constraint

The layout list must be an **inline literal** at the call site. Binding it to a
plain local first makes the `@triton.jit` code generator try to tensor-convert the
keyword strings → `CompilationError`:

```python
lay = [(0, "floordiv", 64), 1, (0, "mod", 64)]
tl.spyre_tensor_layout(a_desc, lay)                              # ❌ raises
tl.spyre_tensor_layout(a_desc, [(0,"floordiv",64), 1, (0,"mod",64)])  # ✅ inline
```

Passing the layout as a **`tl.constexpr` kernel argument** is also fine (a
constexpr is not a runtime local) — this is how the PR fixtures parametrize
`A_LAYOUT`/`B_LAYOUT`/`C_LAYOUT` and apply them with
`if A_LAYOUT != 0: tl.spyre_tensor_layout(a_desc, A_LAYOUT)`.

## Dependency

`tl.spyre_tensor_layout` exists only in the `torch-spyre/triton` PR #19 build,
not in stock PyPI Triton. This repo's KTIR generation pins that spyre-Triton rev
in three places (`.github/workflows/ci.yaml`, `scripts/gen_ktir.py`,
`scripts/_spyre/round_trip.py`) — currently commit `92dffea6`, which provides the
builtin, the `RewriteDescriptorLayout` pass, and its `linalg.batch_matmul` /
reduce dispatch. The base tier (`pyproject.toml` `triton>=3.7.0`, PyPI) is
unchanged; the marker only matters on the spyre lowering, so it is exercised via
`gen_ktir.py` + the ktir-cpu tests, not on a GPU.

Source of truth: `torch-spyre/triton` PR #19 and its
`third_party/spyre/test/fixtures/matmul/` fixtures.
