---
icon: octicons/file-symlink-file-16
---
# IREE Lowering Configs

## Overview

The lowering config is an attribute that is used to correctly and optimally
lower operations within a dispatch from the tensor level down to the vector
level. They are determined by:

1. The type of computation being performed (e.g., matmul, reduction,
   convolution)
2. Hardware attributes (e.g., subgroup size, memory bandwidth, compute units)
3. Optional tuner refinements for performance optimization

IREE provides multiple variants of lowering configs depending on the desired
backend and type of computation.

---

## LLVMGPU Vector Distribute Pipeline

### Reduction

This configuration adopts the broader reduction strategy used in memory-bound
kernels, drawing inspiration from the high-level approach described in Harris’s
[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

#### Relevant lowering config attributes

* `workgroup` tile sizes
* `thread` tile sizes
* `partial_reduction` tile sizes
* `lane_basis` (thread distribution within a subgroup)
* `subgroup_basis` (subgroup distribution within a workgroup)
* `expand_dims` reassociation list

#### Tile sizes

Tile sizes are expressed as arrays of integers, one per dimension of the
iteration space. A zero indicates that the tiling level does not apply to that
dimension.

The three relevant tiling levels for this pipeline are: **workgroup**,
**thread** and **partial reduction**.

Workgroup- and thread-level tilings directly describe the tile sizes at their
respective levels.

**Example:**

```mlir
workgroup = [16, 0]

Dimension 0: Each workgroup produces 16 output elements in d0.
```

Partial reduction tiling is slightly less straightforward and is described as
follows:

##### `partial_reduction` tile sizes

**Applies to:** Reduction dimensions only.

**Tiling strategy:** The reduction dimension `r` is tiled such that
`r -> r_outer, r_partial`, where we create a serial loop over `r_outer` with
step size equal to `r_partial`. Within each iteration, threads maintain
`r_partial` partial accumulators across the reduction dimension. At the end,
partial results are merged.

**Semantics:**

* `partial_reduction[d] = 0`: Dimension `d` is not a reduction dimension.
* `partial_reduction[d] = S`: Tile the reduction dimension into chunks of size
  `S`.

**Number of iterations:**

```text
iterations = ⌈reduction_size / partial_reduction[d]⌉
```

**Special case:** If `reduction_size / partial_reduction[d] = 1`, there is only
one iteration and the outer loop can be elided.

**Example:**

```mlir
partial_reduction = [0, 512]

Dimension 0: Not a reduction dimension.
Dimension 1: Process the reduction in tiles of 512 elements.

For a reduction of size 16384:
- Loop iterations: 16384 / 512 = 32.
- Each iteration: threads within the subgroup process 512 elements along
  dimension 1.
```

> **Tip:**
> The total number of elements each thread processes per iteration along a
> reduction dimension `d` is:
> **`partial_reduction[d] * thread[d]`**

---

#### `Basis` attributes

Basis attributes describe how a particular resource is distributed within the
iteration space.

There are two basis attributes:

* **Lane basis** — describes how threads within a subgroup are distributed
  within the specified iteration space
* **Subgroup basis** — describes how subgroups within a workgroup are
  distributed within the specified iteration space

**Format:** `[[counts], [mapping]]`

* `counts`: Array of thread counts per basis dimension; i.e, the shape of the
  conceptual grid of resources onto `mapping`.
* `mapping`: Permutation array mapping basis coordinates to iteration
  dimensions.

##### The `counts` Array

**Definition:** Number of threads/subgroups along each basis axis.

**Constraint:** The product of all counts equals the subgroup size
(for `lane_basis`) or number of subgroups (for `subgroup_basis`).

**Example:**

```mlir
lane_basis = [[16, 4], [1, 0]]
counts = [16, 4]
For a subgroup of 64 threads:
* 16 × 4 = 64
* This forms a conceptual 16×4 grid of threads in basis space.
```

##### The `mapping` Array

**Definition:** A permutation that maps basis coordinates to iteration space
dimensions.

**Semantics:**

```text
mapping[j] = i  means:  iteration_dim[i] ← basis_digit d_j
```

**Example:**

```mlir
mapping = [1, 0]
```

```text
This swaps/transposes the coordinates:

- Basis digit d₀ maps to iteration dimension 1.
- Basis digit d₁ maps to iteration dimension 0.
```

##### Computing thread position based on lane_basis (Step by step)

Given a thread ID `x`, compute its position in the iteration space:

**Step 1: Delinearize** `x` using `counts`.

Let the counts be `B₀, B₁, …, Bₙ₋₁`, and let `N = Π Bᵢ`.

P<sub>i</sub> = ∏<sub>k=i</sub><sup>n−1</sup> B<sub>k</sub>

The basis digits (coordinates) are:

```text
dᵢ = ⌊ (x mod Pᵢ) / Pᵢ₊₁ ⌋     for i = 0..n-1 where each digit ranges 0 <= dᵢ < bᵢ
```

**Step 2: Apply the mapping** to get iteration-space coordinates.

```text
iteration_dim[mapping[i]] = dᵢ   for i = 0..n-1
```

##### Concrete Example: Thread 42 with `[[16, 4], [1, 0]]`

**Step 1: Delinearize(42, [16, 4])**

```text
Basis counts: [16, 4]

Products:
  P₂ = 1
  P₁ = 4
  P₀ = 64

Digits:
  d₀ = ⌊(42 mod 64) / 4⌋ = ⌊42 / 4⌋ = 10
  d₁ = ⌊(42 mod 4)  / 1⌋ = ⌊2  / 1⌋ = 2

Basis digits: [d₀, d₁] = [10, 2]
```

**Step 2: Apply mapping [1, 0]**

```text
mapping[0] = 1  →  iteration_dim[1] = d₀ = 10
mapping[1] = 0  →  iteration_dim[0] = d₁ = 2

Coordinates: [dim0 = 2, dim1 = 10]
```

**Result:** Thread 42 works at position `[d0 = 10, d1 = 2]` in the
iteration space.

**Visual interpretation:**

```text
Threads form a 16×4 grid in basis space:
       col0 col1 col2 col3
row0:   T0   T1   T2   T3
row1:   T4   T5   T6   T7
...
row10:  T40  T41  T42  T43  ← Thread 42 at (row = 10, col = 2)
...
```

Subgroups distribute work identically to how lane basis distributes lanes.
If there is more than one subgroup, results require workgroup-level
synchronization.

---

#### Dimension Expansion (`expand_dims`)

**Applies to:** Reduction dimensions only.

**Purpose:** Expand (split) a reduction dimension into multiple dimensions in
the iteration space so threads can accumulate at a finer granularity across
the reduction loop. Without `expand_dims`, each thread typically keeps a full
vector accumulator across the entire reduction (e.g., `vector<8xf16>`) and
reduces it at the end; with `expand_dims`, the reduction is split so each
thread can reduce per inner chunk (e.g., `vector<1xf16>`), reducing register
pressure while preserving the same logical result.

**Semantics:** The attribute follows the same reassociation model as
`tensor.expand_shape`, with two parameters:

* `reassociations`: Maps original iterator dimensions to expanded dimensions.
  For example, `[[0], [1], [2, 3]]` keeps dimensions 0 and 1 unchanged and
  splits dimension 2 into dimensions 2 and 3.
* `output_shape`: Sizes of the expanded dimensions. Use `?` to indicate a
  dynamic size, which is inferred from the original dimension and the other
  static factors in the same reassociation group (at most one `?` per group).

**Applicability:** Expansion is only performed when it is statically valid
(e.g., the original size is known and divisible by the static factors).
Otherwise, it is ignored.

**Example:**

```mlir
#iree_gpu.expand_dims<[[0], [1], [2, 3]], output_shape = [?, ?, ?, 8]>
```

This keeps `d0` and `d1` unchanged and splits `d2` into `d2` and `d3`, where
`d3 = 8` and `d2 = extent(d2) / 8`.

---

## Summary: Reduction Config Attributes Quick Reference

| Attribute           | Key Semantic                                                          |
| ------------------- | --------------------------------------------------------------------- |
| `workgroup`         | Workgroup tile size along each dimension                              |
| `thread`            | Thread tile size along each dimension (e.g., load width per thread)   |
| `partial_reduction` | Tile size of the reduction dimension(s) processed by the workgroup    |
| `lane_basis`        | Distribution of threads within a subgroup onto the iteration space    |
| `subgroup_basis`    | Distribution of subgroups within a workgroup onto the iteration space |
| `expand_dims`       | Split reduction dimensions to enable finer-grain accumulation         |
