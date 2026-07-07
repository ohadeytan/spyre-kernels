---
name: spyre-convert
description: "Convert a GPU-shaped Triton kernel into a full Spyre-aware kernel (spyre.py) — tensor descriptors plus the authoring invariants (see _shared/invariants/), scratchpad batching, and Spyre-compiler gap handling. Use when asked to make a kernel Spyre-aware or port to Spyre. For a plain raw->descriptor port with no invariants, use td-convert."
---

# Spyre Kernel Conversion Skill

Convert a GPU-shaped Triton kernel (`original.py`) into a Spyre-aware kernel
(`spyre.py`) that satisfies the authoring invariants for IBM Spyre/AIU.

This **builds on the tensor-descriptor conversion** (`td-convert`): first the
memory accesses become descriptors, then this skill adds the Spyre-specific
distribution, scratchpad, and gap handling. If you only need the descriptor port
(no invariants), use `td-convert`.

## Trigger

Use when asked to convert a kernel to Spyre-aware form, port a kernel to Spyre,
or produce a `spyre.py` from an `original.py`.

## Pre-flight

Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md).

## Inputs

- **kernel_name**: directory under `kernels/`
- Source: `kernels/<name>/original.py`
- Output: `kernels/<name>/spyre.py`, kernel `_<name>_kernel_spyre`

## The invariants

Every Spyre-aware kernel must satisfy the invariants in
[`../_shared/invariants/`](../_shared/invariants/) — one file each. **Read that
directory** and apply every file you find; the set may grow or shrink, so do not
rely on a fixed list here.

## Procedure

1. **Descriptors first.** Apply
   [`../_shared/descriptor-rules.md`](../_shared/descriptor-rules.md) in full —
   replace pointer arithmetic, drop redundant masks, respect the 16-byte
   minimum. (Same as `td-convert` step 1–3.)

2. **Satisfy every invariant.** Read
   [`../_shared/invariants/`](../_shared/invariants/) and apply each file's rule
   and canonical pattern — do not work from a fixed list, as the set may change.
   Typical work this entails: replacing the unbounded GPU grid with a
   `tl.program_id` / `tl.num_programs` distribution loop, using fixed constexpr
   tiles with a loop over the dim (never `BLOCK = next_power_of_2(N)`), and
   keeping problem sizes as runtime args with `tl.cdiv` / `tl.minimum` tail
   handling. The invariant files are authoritative for the details.

3. **Strip `@triton.autotune`** and other GPU-only helpers (`tl.assume`,
   `tl.multiple_of`, CUDA/HIP config). *Why autotune specifically:* on GPU,
   autotune times each `BLOCK_*` config and caches the fastest per key — it
   relies on GPU-style tight kernel-launch timing. Spyre does not work that way:
   tile sizes interact with the 2 MB scratchpad budget and OpSpec-level
   scheduling (LoopSpec, tiled_symbols, device_size), so the right sizes come
   from compiler-level analysis, not random search. (Read this as a hard
   constraint, not a style preference.)

4. **Preserve the signature when possible** so the wrapper can call both forms
   with minimal branching. Recover grid-derived values from existing args
   before adding parameters; update the wrapper if you must add any.

5. **Utilize the scratchpad — batch work items.** After the single-item
   conversion, evaluate whether the distribution loop processes one item per
   iteration with a small accumulator. If so, batch `BLOCK_ITEMS` per iteration
   per [`../_shared/spyre/scratchpad-batching.md`](../_shared/spyre/scratchpad-batching.md).
   This also documents the **16-byte escape** (a scalar load widened to a
   `[BLOCK_ITEMS]` vector clears the minimum) and the divergent-control-flow
   masking trade-off.

6. **Handle Spyre-compiler descriptor gaps the right way.** Write the cleanest
   descriptor-first form even if it does not compile yet; if it hits a gap,
   **mark the site with a `# [gap]` annotation** rather than baking a workaround
   into the kernel. See [`../_shared/spyre/gap-handling.md`](../_shared/spyre/gap-handling.md)
   for the annotation contract and the live sources (KB, `torch-spyre/triton`
   issues/docs) where the current gaps are tracked, so the target form is clear
   once the compiler catches up.

## Layout awareness (write logical; annotate the physical layout with a marker)

Write descriptors in **logical** shape — the tensor's math dimensions — see the
worked [`../_shared/examples/matmul-logical.md`](../_shared/examples/matmul-logical.md).
You do **not** hand-write the physical device layout in the shape/strides; the
compiler tiles it.

> **Proposal 2.** When a tensor is stick-tiled, make the physical layout
> **explicit** with a `tl.spyre_tensor_layout` marker on its descriptor. The
> descriptor stays logical; the marker declares the stick factoring; the compiler's
> `RewriteDescriptorLayout` pass synthesizes the physical loops. `tl.dot` is
> untouched and there is **no** reshape glue (contrast the physical variant).

For each stick-tiled descriptor, immediately after constructing it:

```python
a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                    block_shape=[BLOCK_M, BLOCK_K])
tl.spyre_tensor_layout(a_desc, [(0, "floordiv", 64), 1, (0, "mod", 64)])  # stick-on-M
```

- **One entry per physical dim**, in the order `[stick-index, other…, lane]`.
  Convention: `stick-on-X` → `[(X, "floordiv", S), other, (X, "mod", S)]`. Bare
  int = identity on that logical dim; `(src,"floordiv",S)` = stick index;
  `(src,"mod",S)` = within-stick lane.
- **`src` is the logical dim index** being stick-tiled (K for `A[M,K]` is dim 1;
  K for `B[K,N]` is dim 0 — same-looking markers name different axes).
- **`S = 128 // dtype_bytes`** (64 fp16/bf16, 32 fp32, 128 fp8) — not hard-coded.
- **Inline only.** The list must be a literal at the call site; binding it to a
  local makes the jit try to tensor-convert the keyword strings (compile error).
- **Mark the output descriptor** too when the store must scatter into sticks.

Which matmul "case" results (parallel M/N split vs K-reduction) is a *consequence*
of which axis you mark — the compiler decides, not you. See
[`../_shared/spyre/tensor-layout-marker.md`](../_shared/spyre/tensor-layout-marker.md)
for the full coordinate-map reference.

> **Dependency:** `tl.spyre_tensor_layout` requires the PR #19 Triton build
> (`torch-spyre/triton` #19). KTIR generation is pinned to that fork/SHA — see
> [`../_shared/spyre/tensor-layout-marker.md`](../_shared/spyre/tensor-layout-marker.md).

## Output file structure

```python
# SPDX-License-Identifier: <same as original>
# SPDX-FileCopyrightText: <same as original>
#
# Spyre-aware conversion of <kernel_function_name>.
# Original: <source path>
# Changes summarized in kernels/<name>/conversion-notes.md.

import triton
import triton.language as tl


@triton.jit
def _<name>_kernel_spyre(..., BLOCK_...: tl.constexpr):
    """<Brief description.>"""
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    # descriptors ...
    # distribution loop ...
```

## Document the conversion

After writing the kernel, record what changed in
`kernels/<name>/conversion-notes.md` — the single file shared with `td-convert`.
Append a `## Spyre-aware conversion` section; if `td-convert` already wrote a
`## Tensor-descriptor conversion` section, leave it intact and add yours below.
Create the file with both headings if it does not exist yet.

```markdown
# <name> conversion notes

## Tensor-descriptor conversion
- (written by td-convert, if it ran — leave intact)

## Spyre-aware conversion

- Source: original.py → spyre.py
- Invariants: <key per-invariant decisions, e.g. which axis is the distribution loop>
- Scratchpad batching: <BLOCK_ITEMS choice and why, or "none — already dense">
- Gaps: <gap sites + `# [gap]` annotation, or "none">
- <any signature change or other non-obvious decision>
```

Keep it short — the *why* behind non-obvious choices, not a line-by-line diff.

## Checklist

1. [ ] All memory accesses use descriptors (no raw `tl.load`/`tl.store`,
       `make_block_ptr`, `advance`)
2. [ ] Scalar descriptors resolved (last dim ≥ 16 bytes) or kernel declared
       non-portable, per `descriptor-rules.md` §4 — not annotated and shipped.
       Spyre-compiler gaps written in clean form + `# [gap]` annotation — not
       worked around (see `gap-handling.md`)
3. [ ] `@triton.autotune` removed, with the why understood (compiler-level
       tiling, not launch-timing search)
4. [ ] Every invariant in `_shared/invariants/` satisfied (read the directory —
       do not assume a fixed set)
5. [ ] Scratchpad utilization evaluated; batched where beneficial
6. [ ] Output at `kernels/<name>/spyre.py`, kernel `_<name>_kernel_spyre`
7. [ ] `kernels/<name>/conversion-notes.md` updated with the Spyre-aware section
