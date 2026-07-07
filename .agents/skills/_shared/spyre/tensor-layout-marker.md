# `tl.spyre_tensor_layout` — physical stick-layout marker (Proposal 2)

> **Spyre-family only, and PR-#19-gated.** `tl.spyre_tensor_layout` is a Triton
> builtin added by `torch-spyre/triton` **PR #19** (not yet merged). It is not in
> stock PyPI Triton. KTIR generation must use the PR #19 build — see
> **Dependency** at the bottom.

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

`tl.spyre_tensor_layout` exists only in the PR #19 Triton build. This repo's KTIR
generation pins the spyre-Triton rev in three places
(`.github/workflows/ci.yaml`, `scripts/gen_ktir.py`, `scripts/_spyre/round_trip.py`);
the Proposal-2 branch repoints all three to the PR #19 fork+SHA
(`git+https://github.com/fabianlim/triton@7814b672…`). The base tier
(`pyproject.toml` `triton>=3.7.0`, PyPI) is unchanged; the marker only matters on
the spyre lowering, so it is exercised via `gen_ktir.py`, not the interim GPU
test path.

Source of truth: `torch-spyre/triton` PR #19 and its
`third_party/spyre/test/fixtures/matmul/` fixtures.
