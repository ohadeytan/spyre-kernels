# Worked example — matmul (physical stick layout)

> **Physical shape (Proposal 1).** This is the form you write when the kernel
> carries the stick-tiled device layout itself: descriptors in the **physical**
> shape, with reshape glue around `tl.dot`. Contrast
> [`matmul-logical.md`](matmul-logical.md), where you write the logical shape and
> the compiler tiles it. This file is the template you copy for the physical
> variant.

**Input** (`kernels/matmul/original.py`): GPU-shaped matmul with autotune,
pointer arithmetic, unbounded grid.

**Output** (`kernels/matmul/spyre.py`):

```python
@triton.jit
def matmul_kernel_spyre(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    S: tl.constexpr,   # elements per stick = 128 // dtype_bytes (64 for fp16/bf16)
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # Physical descriptors: the stick dim is factored (D -> D//S, S) and the
    # stick pair is placed adjacent + innermost so a single reshape collapses it
    # into the matrix dim for tl.dot.
    #   A[M, K] stick-on-K -> [M, K//S, S], row-major strides [K, S, 1]
    #   B[K, N] stick-on-N -> [K, N//S, S], row-major strides [N, S, 1]
    #   C[M, N] stick-on-N -> [M, N//S, S], row-major strides [N, S, 1]
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K // S, S], strides=[K, S, 1],
        block_shape=[BLOCK_M, BLOCK_K // S, S],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N // S, S], strides=[N, S, 1],
        block_shape=[BLOCK_K, BLOCK_N // S, S],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N // S, S], strides=[N, S, 1],
        block_shape=[BLOCK_M, BLOCK_N // S, S],
    )

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
    m_start = pid * m_blocks_per_core
    m_end = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(n_blocks):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(k_tiles):
                # Load physical 3D tiles; offsets index the factored dims.
                a_tile = a_desc.load([m * BLOCK_M, k * (BLOCK_K // S), 0])
                b_tile = b_desc.load([k * BLOCK_K, n * (BLOCK_N // S), 0])

                # Collapse the (stick, lane) pair -> logical 2D tile for tl.dot.
                a_tile = a_tile.reshape(BLOCK_M, BLOCK_K)   # [M, K//S, S] -> [M, K]
                b_tile = b_tile.reshape(BLOCK_K, BLOCK_N)   # [K, N//S, S] -> [K, N]

                acc = tl.dot(a_tile, b_tile, acc)

            # Reshape the 2D accumulator back to physical before the store.
            acc_phys = acc.to(tl.float16).reshape(BLOCK_M, BLOCK_N // S, S)
            c_desc.store([m * BLOCK_M, n * (BLOCK_N // S), 0], acc_phys)
```

Key decisions:
- **Stick size `S` is dtype-dependent**: `S = 128 // dtype_bytes` (64 for
  fp16/bf16, 32 for fp32, 128 for fp8). Passed as a `constexpr`, not hard-coded.
- **Physical shape factors the stick dim** `D -> (D // S, S)` and places the pair
  **innermost**, so `reshape` to the 2D tile is a free view (no data movement).
- **Strides are row-major over the physical shape** (`[K, S, 1]` for A's
  `[M, K//S, S]`).
- **Reshape at both boundaries**: 3D->2D after each load (for `tl.dot`), 2D->3D
  before the store.
- **M** is the distributed axis; **N**, **K** are inner loops — same distribution
  as the logical variant.
- Accumulator is f32 for the K reduction; the reshape/store happen after the
  down-cast.

Contrast with the logical variant: there the descriptors are `[M, K]` / `[K, N]`
/ `[M, N]`, `tl.dot` runs directly on loaded tiles, and there is **no** reshape —
the compiler derives the physical layout. Here the kernel owns that layout.
