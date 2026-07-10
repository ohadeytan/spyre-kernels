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

VARIANTS = {
    "spyre": {
        "KERNEL": _prefill_attention_kernel_spyre,
        "SIGNATURE": _SPYRE_SIGNATURE,
        "CONSTEXPRS": {
            "kv_group_num": 1,       # num_q_heads // num_kv_heads = 4 // 4
            "BLOCK_M": 64,
            "BLOCK_DMODEL": 64,      # next_power_of_2(Lk=64)
            "BLOCK_N": 64,
            "Lk": 64,
            "S": 64,                 # fp16 stick size: 128 bytes / 2
            "SEQ": 128,              # single-request seq_len (batch == 1)
        },
        # 32-core distribution grid, matching the wrapper's fixed NUM_CORES.
        "GRID": [32],
    },
}
