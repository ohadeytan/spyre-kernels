---
name: spyre-review
description: "Review a Spyre-aware kernel (spyre.py) for compliance with the authoring invariants (see _shared/invariants/), scratchpad utilization, Spyre-compiler descriptor patterns, and correctness vs original. Use when asked to review/verify a Spyre kernel. For a plain descriptor-API review (no invariants), use td-review."
---

# Spyre Kernel Review Skill

Review a Spyre-aware kernel for the authoring invariants, scratchpad
utilization, Spyre-compiler descriptor patterns, and equivalence to the
original. For a descriptor-API-only review (no invariants), use `td-review`.

## Trigger

Use when asked to review/verify/validate a `spyre.py`.

## Pre-flight

Run the KB consult in [`../_shared/preflight.md`](../_shared/preflight.md). If
the KB documents constraints newer than this procedure, flag the discrepancy in
the report.

## Inputs

- **kernel_name**: directory under `kernels/`
- Review: `kernels/<name>/spyre.py`; compare against `kernels/<name>/original.py`

## Review procedure

### Step 1 — Invariants

Read [`../_shared/invariants/`](../_shared/invariants/) and check the kernel
against **every** file in it. Do not work from a fixed list — the set may grow
or shrink, so enumerate the directory each time. Each invariant file states its
own rule, verification steps, and red flags; apply them as written and report
PASS/FAIL per invariant.

### Step 1b — Scratchpad utilization (WARN, not FAIL)

After the hard constraint, assess under-use per
[`../_shared/spyre/scratchpad-batching.md`](../_shared/spyre/scratchpad-batching.md).
If peak live bytes are below ~10% of 2 MB and a loop axis processes one item per
iteration with an independent reduction, recommend batching `BLOCK_ITEMS`, give a
value, and note:
- the **16-byte escape** side benefit (cross-reference any unresolved scalar
  descriptor from Step 2 — batching is one way to resolve it, per
  `descriptor-rules.md` §4);
- the **divergent-control-flow** trade-off (feasible-with-masking, not a
  blocker);
- the **distribution-granularity** trade-off (`cdiv(total_work, BLOCK_ITEMS) ≥ 32`).

### Step 2 — Descriptor API usage

Per [`../_shared/descriptor-rules.md`](../_shared/descriptor-rules.md): no
`make_block_ptr`/`advance`/`order`/`boundary_check`; valid shape/strides/
block_shape; runtime `shape`, constexpr `block_shape`; offsets are block
multiples. Flag **redundant tail masking** as WARN (with the non-identity-fill
exception — a missing non-zero identity is a correctness FAIL; check the
original's `other=`). Per `descriptor-rules.md` §4 a scalar descriptor must be
resolved (last dim ≥ 16 bytes) or the kernel declared non-portable — a surviving
scalar descriptor or stale gap annotation is a FAIL.

### Step 2b — Physical layout markers (Proposal 2)

For stick-tiled tensors, verify the `tl.spyre_tensor_layout` markers (see
[`../_shared/spyre/tensor-layout-marker.md`](../_shared/spyre/tensor-layout-marker.md)):
- **Present** on each stick-tiled descriptor; **inline literal** or a `constexpr`
  arg (a list bound to a plain local is a compile error — FAIL).
- **Well-formed**: one entry per physical dim; `stick-on-X` form
  `[(X,"floordiv",S), other, (X,"mod",S)]`; `src` indices in range for the
  logical rank.
- **Stick divisor** `S = 128 // dtype_bytes` (64 fp16/bf16, 32 fp32, 128 fp8) —
  a hard-coded `64` on a non-2-byte dtype is a FAIL.
- **Axis matches intent**: the marked axis is the one actually stick-tiled; A
  stick-on-K vs stick-on-M changes the synthesized case.
- **Output descriptor** carries a marker when the store must scatter into sticks.
- Because PR #19's `RewriteDescriptorLayout` closes the layout gap, a *missing*
  marker where one is needed is a **FAIL**, not a `# [gap]` (reconcile with
  Step 3 / `gap-handling.md`).

### Step 3 — Spyre-compiler descriptor patterns

The set of compiler descriptor gaps changes as the lowering evolves, so this
step does not enumerate them — see
[`../_shared/spyre/gap-handling.md`](../_shared/spyre/gap-handling.md)
for the contract and the live sources (KB, `torch-spyre/triton` issues/docs)
where the current gaps are tracked.

What to verify is the **handling**, not a fixed list: every site that hits a
compiler gap stays in clean descriptor-first form with a `# [gap]` annotation,
**not** a baked-in raw-pointer workaround or invented layout. A workaround
shipped in place of the clean form is the FAIL.

### Step 4 — Removed GPU patterns

- [ ] No `@triton.autotune`, `tl.assume`, `tl.multiple_of`, CUDA/HIP config
- [ ] No `order=`, `boundary_check=`, `padding_option=`

### Step 5 — Correctness vs original

Same math, accumulator dtype, output dtype conversion, reduction structure; no
dropped operations; all original tensors handled.

### Step 6 — Conversion notes (WARN, not FAIL)

`spyre-convert` requires `kernels/<name>/conversion-notes.md` with a
`## Spyre-aware conversion` section recording the conversion decisions
(invariant handling, scratchpad batching, gap sites, signature changes). Verify
it exists and reflects the kernel as reviewed. Flag a **WARN** if the file is
missing, lacks the Spyre-aware section, or is stale relative to the code —
documentation hygiene, not a correctness defect, so it does not by itself make
the kernel non-compliant.

## Report format

```
## Spyre Compliance Review: <kernel_name>

### Invariants
- <one line per invariant file in `_shared/invariants/`>: PASS/WARN/FAIL — <evidence, verdict>

### Scratchpad utilization
**Status:** OK / WARN
- Peak live bytes / 2MB: <%>; batchable axis: <name/none>;
  recommended BLOCK_ITEMS: <value>; side benefits/caveats: <...>

### Descriptor API usage
**Status:** PASS / WARN / FAIL — <block_ptr, redundant masking, unresolved scalar descriptors>

### Spyre-compiler descriptor patterns
**Status:** PASS / FAIL — <gaps in clean form + `# [gap]` annotation, or baked-in workaround>

### Correctness vs original
**Status:** PASS / FAIL — <discrepancies>

### Conversion notes
**Status:** OK / WARN — <present and current / missing / stale>

### Overall: COMPLIANT / NON-COMPLIANT
```

Conversion is `spyre-convert`; numerical tests are `spyre-test`.
