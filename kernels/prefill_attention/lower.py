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

The concrete shape frozen into the KTIR matches ``tests/ktir/test_prefill_attention.py``:
one request (seq_len 128), 4 query / 4 kv heads, head_dim 64, fp16, causal. That
is a single Q-tile at ``BLOCK_M = 64`` over two KV tiles. ``S = 64`` is the fp16
stick size (128 bytes / 2), and ``head_dim`` (64) is a multiple of ``S`` as the
physical stick layout requires. The causal mask is precomputed on the host and
passed as the additive ``MASK`` tensor (no ``tl.arange`` -> no ``tt.make_range``),
so the KTIR lowers fully and executes on ktir-cpu.

The ``spyre_dist_c{1,4,16,32}`` variants lower the same kernel at different core
counts to verify the distribution loop is partition-independent (see the test's
``TestPrefillAttentionDistribution``).
"""

from kernels.prefill_attention.spyre import _prefill_attention_kernel_spyre

# Concrete shapes match tests/ktir/test_prefill_attention.py.
_SPYRE_SIGNATURE = {
    "Q": "*fp16",
    "K": "*fp16",
    "V": "*fp16",
    "MASK": "*fp32",
    "Out": "*fp16",
    "sm_scale": "fp32",
    "stride_qbs": "i32",
    "stride_qh": "i32",
    "stride_kbs": "i32",
    "stride_kh": "i32",
    "stride_vbs": "i32",
    "stride_vh": "i32",
    "stride_obs": "i32",
    "stride_oh": "i32",
    "num_q_heads": "i32",
    "num_kv_heads": "i32",
    "batch": "i32",
    "num_m_blocks": "i32",
}

_SPYRE_CONSTEXPRS = {
    "kv_group_num": 1,       # num_q_heads // num_kv_heads = 4 // 4
    "BLOCK_M": 64,
    "BLOCK_DMODEL": 64,      # next_power_of_2(Lk=64)
    "BLOCK_N": 64,
    "Lk": 64,
    "S": 64,                 # fp16 stick size: 128 bytes / 2
    "SEQ": 128,              # single-request seq_len (batch == 1)
}

# ── Distribution-invariance variants ────────────────────────────────────────
# The GRID (core count) is baked into the .ktir at lowering time, so verifying
# the distribution loop is partition-independent means lowering the SAME kernel
# at several grids and asserting ktir-cpu produces the same result for each.
#
# The primary shape already partitions non-trivially: the work space is
# batch * num_q_heads * num_m_blocks = 1 * 4 * 2 = 8 items (SEQ=128, BLOCK_M=64
# -> 2 m_blocks; 4 heads). grid [1] runs all 8 on one core, [4] gives 2 items
# per core, [16]/[32] leave most cores idle (more cores than work). Only the
# GRID differs across variants; SIGNATURE and CONSTEXPRS are identical.
# Tests: tests/ktir/test_prefill_attention.py::TestPrefillAttentionDistribution.


def _dist_variant(grid: int) -> dict:
    return {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": _SPYRE_CONSTEXPRS,
        "GRID": [grid],
    }


# NOTE: gen_ktir.py enumerates variants by statically parsing this dict's
# *literal keys* (ast, no import), so every variant must appear as an explicit
# key here — a loop that inserts keys afterward would be invisible to it.
VARIANTS = {
    "spyre": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": _SPYRE_CONSTEXPRS,
        # 32-core distribution grid, matching the wrapper's fixed NUM_CORES.
        "GRID": [32],
    },
    "spyre_dist_c1": _dist_variant(1),
    "spyre_dist_c4": _dist_variant(4),
    "spyre_dist_c16": _dist_variant(16),
    "spyre_dist_c32": _dist_variant(32),
}
