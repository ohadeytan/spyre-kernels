# SPDX-License-Identifier: Apache-2.0
"""KTIR lowering driver for the prefill_attention kernel variants.

Consumed by ``scripts/gen_ktir.py``, which lowers each entry in ``VARIANTS``
to ``kernels/prefill_attention/<variant>.ktir``.

``VARIANTS`` maps a **variant name** (also the source ``.py`` module name and
the output ``.ktir`` stem) to the four things the round-trip lowering needs:

    KERNEL      : the @triton.jit function to lower
    SIGNATURE   : dict[str, str]  arg name -> Triton type ("*fp16", "i32", ...)
    CONSTEXPRS  : dict[str, value] for every constexpr arg
    GRID        : optional list, forwarded to SpyreOptions.grid

There is a single kernel, ``_prefill_attention_kernel_spyre``: the all-heads
form where the head is the *batch* axis of 3D ``[HEADS, SEQ, Lk]`` descriptors
(QK^T / P·V lower to ``linalg.batch_matmul``) and the causal/validity mask is an
additive host tensor — so the lowered kernel has live ``tl.spyre_tensor_layout``
markers, no ``tl.arange``, and no ``tt.addptr`` / ``tt.reshape`` glue.

``SEQ`` and ``SEQ_KV`` are **runtime i32 args**, not constexpr, so one lowered
kernel serves any per-request length (dynamic descriptor extent + runtime
KV-loop bound). Each variant below is the *same* kernel lowered with different
constexprs (head counts, head dim) or grid — GQA, head-dim padding, fixed batch,
and the distribution grids are all lowerings of that one kernel. The per-request
length is supplied at *execution* time by the test, not baked at lowering.
"""

from kernels.prefill_attention.spyre import _prefill_attention_kernel_spyre

# Concrete shapes match tests/ktir/test_prefill_attention.py: head_dim 64, fp16,
# multiple heads in one launch (batched matmul). SEQ is a runtime arg, so it is
# not fixed at lowering — the test drives several lengths through one .ktir.
_HEADS = 4
_HEAD_DIM = 64
_BLOCK = 128
_STICK = 128 // 2  # fp16 -> 64 elems per 128-byte DataStick

# SEQ / SEQ_KV are runtime i32 args (right after Out, before the strides),
# matching _prefill_attention_kernel_spyre's parameter order. Everything that is
# fixed at compile time (head counts, tile sizes, head dim) stays constexpr.
_SPYRE_SIGNATURE = {
    "Q": "*fp16",
    "K": "*fp16",
    "V": "*fp16",
    "Mask": "*fp32",
    "sm_scale": "fp32",
    "Out": "*fp16",
    "SEQ": "i32",       # runtime sequence length
    "SEQ_KV": "i32",    # runtime padded key length = cdiv(SEQ, BLOCK_N) * BLOCK_N
    "stride_qh": "i32",
    "stride_qbs": "i32",
    "stride_kh": "i32",
    "stride_kbs": "i32",
    "stride_vh": "i32",
    "stride_vbs": "i32",
    "stride_oh": "i32",
    "stride_obs": "i32",
    "stride_mm": "i32",
    "HEADS": "i32",
    "KV_HEADS": "i32",
    "BLOCK_M": "i32",
    "BLOCK_N": "i32",
    "STICK": "i32",
    "Lk": "i32",
    "DMODEL": "i32",
}

# ── Distribution-invariance variants ────────────────────────────────────────
# The GRID (core count) is baked into the .ktir at lowering time, so verifying
# the distribution loop is partition-independent means lowering the SAME kernel
# at several grids and asserting ktir-cpu produces the same result for each.
#
# SEQ is a runtime arg, so the work-space size (HEADS * cdiv(SEQ, BLOCK_M)) is
# computed at runtime; the test drives SEQ=256 (two BLOCK=128 query-blocks) with
# HEADS=4 so the work space (8 items) is non-trivial to partition: grid [1] runs
# all 8 on one core, [4] gives each core 2, [16]/[32] leave most cores idle.
# Tests: tests/ktir/test_prefill_attention.py::TestPrefillAttentionDistribution.
_DIST_HEADS = 4


def _dist_variant(grid: int) -> dict:
    return {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "HEADS": _DIST_HEADS,
            "KV_HEADS": _DIST_HEADS,  # 1:1 heads (no GQA) in the dist variants
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": _HEAD_DIM,
            "DMODEL": _HEAD_DIM,  # Lk is a stick multiple -> no head-dim padding
        },
        "GRID": [grid],
    }


# NOTE: gen_ktir.py enumerates variants by statically parsing this dict's
# *literal keys* (ast, no import), so every variant must appear as an explicit
# key here — a loop that inserts keys afterward would be invisible to it.
VARIANTS = {
    # Base: 4 heads, 1:1 query:KV (multi-head attention), head dim 64. SEQ is a
    # runtime arg, so this one .ktir serves any per-request length.
    "spyre": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "HEADS": _HEADS,
            "KV_HEADS": _HEADS,  # 1:1 query:KV heads (multi-head attention)
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": _HEAD_DIM,
            "DMODEL": _HEAD_DIM,  # Lk is a stick multiple -> no head-dim padding
        },
        # 32 cores; the kernel's distribution loop covers this request's
        # (head, query-block) work space regardless of core count.
        "GRID": [32],
    },
    # GQA: 4 query heads share 2 KV heads (group size 2). K/V descriptors carry
    # KV_HEADS on the batch axis; the kernel reads KV head `head // 2`. Exercises
    # a num_q_heads != num_kv_heads batch_matmul (head-sharing at the load index).
    "spyre_gqa": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "HEADS": _HEADS,        # 4 query heads
            "KV_HEADS": _HEADS // 2,  # 2 KV heads -> group size 2
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": _HEAD_DIM,
            "DMODEL": _HEAD_DIM,
        },
        "GRID": [32],
    },
    # Head-dim padding: a logical head dim Lk=48 (not a stick multiple) padded to
    # DMODEL=64 (one full STICK). Descriptors carry Lk in `shape`; block_shape /
    # marker / tiles use DMODEL. The host must pad+zero the physical head dim to
    # DMODEL (see the kernel docstring); the test honors that contract.
    "spyre_pad": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "HEADS": _HEADS,
            "KV_HEADS": _HEADS,
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": 48,          # ragged head dim (not a power of 2 / stick multiple)
            "DMODEL": _STICK,  # padded to one full stick (64 for fp16)
        },
        "GRID": [32],
    },
    # Fixed multi-request batch, folded into the head axis. A batch of B uniform-
    # length requests is B*HEADS independent attention problems, so the host lays
    # Q/K/V/O out as [B*HEADS, SEQ, Lk] and the unchanged kernel handles it. Here
    # B=2 × 4 heads = 8 "heads". Valid when every request shares the mask
    # (uniform-length prefill).
    "spyre_batch": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "HEADS": 2 * _HEADS,      # BATCH(2) * HEADS(4) folded into the head axis
            "KV_HEADS": 2 * _HEADS,   # 1:1 within the fold (no GQA here)
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": _HEAD_DIM,
            "DMODEL": _HEAD_DIM,
        },
        "GRID": [32],
    },
    "spyre_dist_c1": _dist_variant(1),
    "spyre_dist_c4": _dist_variant(4),
    "spyre_dist_c16": _dist_variant(16),
    "spyre_dist_c32": _dist_variant(32),
}
