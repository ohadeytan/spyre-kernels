# Tensor-descriptor rules

The `tl.make_tensor_descriptor` API mechanics, shared by the TD and Spyre skill
families. These are **Triton-frontend** facts — they apply the moment you use a
descriptor, independent of which backend you target.

## 1. Replace pointer arithmetic / `block_ptr` with `tl.make_tensor_descriptor`

`tl.make_block_ptr` is deprecated; raw `tt.addptr` loads are legacy.

**Before (raw pointer):**
```python
a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
```

**Before (block_ptr):**
```python
p = tl.make_block_ptr(base, shape, strides, offsets, block_shape=(BM, BN), order=(1, 0))
x = tl.load(p, boundary_check=(0, 1))
p = tl.advance(p, (0, BLOCK_K))
```

**After (descriptor):**
```python
a_desc = tl.make_tensor_descriptor(
    a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
)
a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
```

Key differences:
- `tl.advance` has no equivalent — pass different offsets to each `.load()` / `.store()`.
- `boundary_check` / `padding_option` are implicit — the descriptor handles OOB.
- `mask` is not used — OOB loads return zero by default.
- `order` does not exist — layout is determined by `strides`.
- Descriptors may be any rank (1D–4D). Match the tensor's actual rank to avoid
  `addptr` in the base pointer; pass strides straight from the kernel signature.

Descriptor placement is flexible (top level, inside loops, inside conditionals).
Top-level is preferred — the view is constructed once and reused.

## 2. Drop the tail mask — `shape` replaces it

A raw-pointer kernel *must* mask the partial tail tile (a pointer is just an
address with no notion of where the tensor ends). The descriptor's `shape=[...]`
**is** the boundary: declared once at construction, every `.load()` zero-fills
out-of-range lanes and every `.store()` clamps writes at the boundary. So the
original's `mask` / `tl.where` / `other=` machinery is **redundant** — do not
port it.

**Before (mask mandatory):**
```python
offs = tile * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
x = tl.load(in_ptr + offs, mask=mask, other=0.0)
acc += tl.sum(tl.where(mask, x, 0.0))        # re-mask before reducing
```

**After (no mask):**
```python
x = in_desc.load([tile * BLOCK])             # lanes >= N already zero-filled
acc += tl.sum(x)                             # zero lanes are the sum identity
```

Store side is the same — `out_desc.store([off], y)` clamps at `shape`, so the
tail tile never writes past `N`. Carrying a manual mask re-zeroes already-zero
lanes and forces needless `arange` / `[None, :]` broadcasts inside the loop
(worse on multi-dim batched tiles).

**Caveat — non-identity fills.** Zero-fill is the identity only for *additive*
reductions. If a tail lane must be a non-zero identity (`other=1.0` for a
product, `-inf` for a max), the descriptor's zero-fill is wrong,   and you DO need
to post-process the tail. Dropping the mask *and* needing a non-zero identity is
a correctness bug — check against the original's `other=` value.

## 3. Strides must be expressible at descriptor creation

Strides must be known when the descriptor is created. For row-major contiguous
tensors use computed strides (`[N, 1]` for `[M, N]`). If the original kernel
receives strides as runtime args, pass them directly:
`strides=[stride_row, stride_col]`.

## 4. ≥16 bytes in the last dimension — avoid scalar loads

`tl.make_tensor_descriptor` requires **≥ 16 bytes in the last dimension** — a
Triton frontend constraint, so it applies on **any** backend. A scalar load
(single i32/f32) with `block_shape=[1]` or `block_shape=[..., 1]` is rejected at
trace time.

Scalar descriptor loads must therefore be **avoided** — this is not something to
annotate and leave in place. Some ways to resolve it:

- **Batch the load.** Widen the offending axis so the last dim carries
  ≥ 16 bytes — load `BLOCK_ITEMS` elements at once
  (`BLOCK_ITEMS × dtype_bytes ≥ 16`) instead of one. Batching is a plain
  descriptor technique, not backend-specific; it applies wherever you would
  otherwise issue a per-element load.
- **Restructure the memory layout** so the contiguous last dimension is wide
  enough — e.g. transpose, or fold a scalar-per-row quantity into a vector the
  kernel already loads.

These are examples, not an exhaustive list — whatever makes the last dimension
≥ 16 bytes is fair game. But the constraint must be addressed. If no
restructuring can make the last dimension ≥ 16 bytes, then **the kernel is not
portable to tensor descriptors** — say so and stop; do not ship a
`block_shape=[..., 1]` descriptor with a gap annotation, because it will not
trace.

## 5. Data-dependent gather / scatter — use `desc.gather` / `desc.scatter`

A runtime-indexed access — `x[idx]` where `idx` was loaded from memory, or a
scatter to a runtime-computed position — is **not** a reason to give up on
descriptors. `desc.load([...])` takes only trace-time offsets, but descriptors
also expose:

```python
desc.gather(indices, offset)            # indices: tensor of coords on the indirect dim
desc.scatter(value, indices, offset)    # offset:  scalar coord on the direct dims
```

`indices` supplies the coordinate for **one** (indirect) dimension; `offset`
gives the direct-dimension coordinate. This is the canonical form for paged
KV-cache, MoE expert routing, and per-row index lookups, and it maps 1:1 to the
backend's indirect-access tile (`ktdp.construct_indirect_access_tile`). So the
fix for a data-dependent gather is **express it as `desc.gather` on a
multi-dim descriptor**, not leave it as a raw `tl.load(ptr + idx)`.

**What genuinely cannot be expressed (then, and only then, non-portable):**
- **Multiplicative mixing of the indirect index.** The loaded index must be a
  *standalone* per-dimension coordinate. `X_flat[IDX[m] * K + k]` is illegal —
  the index is scaled and linearized. The fix is usually to *not* pre-flatten:
  keep the tensor multi-dimensional (`X_2d[IDX[m], k]`) so the index stays its
  own coordinate. If you only have a flat pointer and the offset truly mixes the
  loaded index with a stride, it will not lower.
- **Dimension fusion with an indirect dim** — a fused axis where one component is
  indirect cannot be represented; the indirect axis must stay structurally
  separate.
- **Non-delinearizable patterns** — `address = f(loaded_value)` for a
  non-invertible `f` (hash, content-addressable lookup). Outside the IR's
  expressiveness entirely.

Only if the access falls into one of those three (and cannot be restructured out
of it) is the kernel **non-portable** — then say so, name the access, and stop;
do not ship a raw-pointer gather with an annotation. But the common case —
a per-row / per-program index into an otherwise structured tensor — **is**
portable via `desc.gather` / `desc.scatter`. Reach for those before declaring
defeat. (The gathered tile still owes §4: gather a ≥ 16-byte slice, not a single
element.)

## 6. Physical stick layout — annotate with `tl.spyre_tensor_layout` (Proposal 2)

> **Spyre-family only, PR-#19-gated.** For stick-tiled tensors, keep the
> descriptor **logical** and declare the physical layout with a
> `tl.spyre_tensor_layout` marker; the compiler synthesizes the physical loops.
> `tl.dot` is unchanged and there is no reshape glue.

```python
tl.spyre_tensor_layout(a_desc, [(0, "floordiv", 64), 1, (0, "mod", 64)])  # A[M,K] stick-on-M
```

The stick divisor is `S = 128 // dtype_bytes` (64 fp16/bf16, 32 fp32, 128 fp8),
and the list must be **inline** (or a `constexpr` arg). Full coordinate-map
reference, cases, output-sink behavior, and the dependency pin:
[`spyre/tensor-layout-marker.md`](spyre/tensor-layout-marker.md).
