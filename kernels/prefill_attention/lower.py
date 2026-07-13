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

The spyre variant is the single-(request, head) form: ``SEQ`` is a compile-time
constant, one head per launch (2D descriptors), and the causal/validity mask is
an additive host tensor — so the lowered kernel has live
``tl.spyre_tensor_layout`` markers, no ``tl.arange``, and no ``tt.addptr`` /
``tt.reshape`` glue.
"""

from kernels.prefill_attention.spyre import _prefill_attention_kernel_spyre

# Concrete shapes match tests/ktir/test_prefill_attention.py: a single 128-token
# sequence, head_dim 64, fp16, one head per launch.
_SEQ = 128
_HEAD_DIM = 64
_BLOCK = 128
_STICK = 128 // 2  # fp16 -> 64 elems per 128-byte DataStick

_SPYRE_SIGNATURE = {
    "Q": "*fp16",
    "K": "*fp16",
    "V": "*fp16",
    "Mask": "*fp32",
    "sm_scale": "fp32",
    "Out": "*fp16",
    "stride_qbs": "i32",
    "stride_kbs": "i32",
    "stride_vbs": "i32",
    "stride_obs": "i32",
    "stride_mm": "i32",
    "SEQ": "i32",
    "SEQ_KV": "i32",
    "BLOCK_M": "i32",
    "BLOCK_N": "i32",
    "STICK": "i32",
    "Lk": "i32",
}

# ── Distribution-invariance variants ────────────────────────────────────────
# The GRID (core count) is baked into the .ktir at lowering time, so verifying
# the distribution loop is partition-independent means lowering the SAME kernel
# at several grids and asserting ktir-cpu produces the same result for each.
#
# These use SEQ=256 (two BLOCK=128 query-blocks) so the partition is non-trivial:
# grid [1] runs both blocks on one core, [4] gives cores 0/1 one block each with
# 2 idle, [16]/[32] leave most cores idle. With the primary spyre variant's
# SEQ=128 (a single query-block) every grid is degenerate and tests nothing.
# Tests: tests/ktir/test_prefill_attention.py::TestPrefillAttentionDistribution.
_DIST_SEQ = 256


def _dist_variant(grid: int) -> dict:
    return {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "SEQ": _DIST_SEQ,
            "SEQ_KV": _DIST_SEQ,  # SEQ is a BLOCK multiple -> no key padding
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": _HEAD_DIM,
        },
        "GRID": [grid],
    }


# NOTE: gen_ktir.py enumerates variants by statically parsing this dict's
# *literal keys* (ast, no import), so every variant must appear as an explicit
# key here — a loop that inserts keys afterward would be invisible to it.
VARIANTS = {
    "spyre": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "SEQ": _SEQ,
            "SEQ_KV": _SEQ,  # SEQ is a BLOCK multiple here -> no key padding
            "BLOCK_M": _BLOCK,
            "BLOCK_N": _BLOCK,
            "STICK": _STICK,
            "Lk": _HEAD_DIM,
        },
        # 32 cores; the kernel's distribution loop covers this head's
        # query-blocks regardless of core count.
        "GRID": [32],
    },
    "spyre_dist_c1": _dist_variant(1),
    "spyre_dist_c4": _dist_variant(4),
    "spyre_dist_c16": _dist_variant(16),
    "spyre_dist_c32": _dist_variant(32),
}
