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

## Layout awareness (emit the physical device layout yourself)

> **Proposal 1.** In this variant the kernel carries the **physical stick-tiled
> layout directly** in the descriptor's `shape`/`strides`/`block_shape` — stock
> `tl.make_tensor_descriptor`, no marker op, no compiler layout pass. You compute
> the stick factoring and the reshape glue; the compiler sees an ordinary strided
> tensor. See the worked
> [`../_shared/examples/matmul-physical.md`](../_shared/examples/matmul-physical.md).

For every stick-tiled tensor:

1. **Elements per stick is dtype-dependent.** A DataStick is **128 bytes**, so
   `S = 128 // dtype_bytes` — **64** for fp16/bf16, **32** for fp32, **128** for
   fp8. Do **not** hard-code 64.

2. **Factor the stick dimension.** The innermost matrix dim is stick-tiled
   (K for A, N for B, N for C in a matmul). Logical `[..., D, ...]` becomes a
   physical shape with the stick dim split into `(D // S, S)`.

3. **Permute so the stick pair is adjacent and innermost.** `tl.dot` needs a 2D
   tile, so the `(D // S, S)` pair must sit last (leading matrix/batch dims ahead)
   and collapse with one `reshape`. For `A[M, K]` stick-on-K the operand order is
   **`[M, K // S, S]`** (→ reshape `[M, K]`), **not** `[K // S, M, S]`.

4. **Row-major strides over the physical shape.** For `[M, K // S, S]` that is
   `[K, S, 1]`.

5. **Reshape at the boundaries.** After each `.load()` collapse the stick pair to
   the logical 2D tile before `tl.dot`; before each `.store()` reshape the 2D
   accumulator back to the physical shape.

The [`../_shared/descriptor-rules.md`](../_shared/descriptor-rules.md) "Physical
stick layouts" section gives the factoring/stride/reshape mechanics in full.

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
