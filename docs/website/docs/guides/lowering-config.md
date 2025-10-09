---
icon: octicons/file-symlink-file-16
---
# Lowering Configs

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

#### Partial Reduction

This configuration is set when an operation contains at least one reduction
dimension and targets efficient reduction strategies.

#### Relevant lowering config attributes

- `workgroup` tile sizes
- `thread` tile sizes
- `partial_reduction` tile sizes
- `lane_basis` (thread distribution within a subgroup)
- `subgroup_basis` (subgroup distribution within a workgroup)

---

#### `workgroup` Tile Sizes

**Applies to:** Parallel dimensions.

**Definition:** The output tile that each workgroup computes.

**Format:** Array of integers, one per iteration dimension.

**Semantics:**

* `workgroup[d] > 0`: Each workgroup produces this many elements in dimension
  `d`.

**Example:**

```mlir
workgroup = [16, 0]

Dimension 0 (parallel): Each workgroup produces 16 output elements in d0.
```

---

#### `thread` Tile Sizes

**Applies to:** Reduction dimensions.

**Definition:** The number of elements each thread processes per load per
iteration along reduction dimensions.

**Format:** Array of integers, one per iteration dimension.

**Semantics:**

* `thread[d] = N`: Each thread loads `N` elements per iteration of the reduction
  loop along `d`.

**Example:**

```mlir
thread = [0, 8]

Dimension 1 (reduction): Each thread loads 8 elements per loop iteration
along d1.
```

---

#### `partial_reduction` Tile Sizes

**Applies to:** Reduction dimensions.

**Tiling strategy:** We use `PartialReductionOuterReduction` at the partial
reduction level. This tiles the reduction dimension as `r -> r_outer, r_partial`,
where we create a serial loop over `r_outer` with step size equal to
`r_partial`. Within each iteration, threads maintain `r_partial`
partial accumulators across the reduction dimension. At the end, partial results
are merged.

This config specifies the chunk size `S` for `PartialReductionOuterReduction`
(i.e., `r -> r/S, S`) in the reduction dimension.

**Format:** Array of integers, one per iteration dimension.

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
```

```text
Dimension 0: Not a reduction dimension.
Dimension 1: Process the reduction in chunks of 512 elements.

For a reduction of size 16384:
- Loop iterations: 16384 / 512 = 32.
- Each iteration: threads within the subgroup process 512 elements along
  dimension 1.
```

---

#### `lane_basis` (Thread Distribution within Subgroup)

**Purpose:** Defines how threads within a single subgroup are distributed across
the iteration space.

**Format:** `[[counts], [mapping]]`

* `counts`: Array of thread counts per basis dimension.
* `mapping`: Permutation array mapping basis coordinates to iteration
  dimensions.

##### The `counts` Array

**Definition:** Number of threads spanning each basis dimension.

**Constraint:** The product of all counts equals the subgroup size.

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

##### Computing Thread Position (Step by Step)

Given a thread ID `x`, compute its position in the iteration space.

**Step 1: Delinearize** `x` into mixed-radix digits using `counts`.

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

Iteration coordinates: [dim0 = 2, dim1 = 10]
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

---

#### `subgroup_basis` (Subgroup Distribution within Workgroup)

**Purpose:** Defines how multiple subgroups within a workgroup are distributed
across the iteration space.

**Format:** Same as `lane_basis`: `[[counts], [mapping]]`

**Constraint:**

```text
product(counts) = number_of_subgroups_per_workgroup
```

**Example:**

```mlir
subgroup_basis = [[1, 2], [0, 1]]
subgroup_size = 64
```

```text
Number of subgroups: 1 × 2 = 2
Workgroup size: 64 × 2 = 128 threads
```

**Interpretation:**

* Delinearization works identically to `lane_basis` but applies to subgroup IDs.
* Subgroup 0 and Subgroup 1 handle different chunks of the iteration space.
* Distribution is determined by the same delinearize-then-map process.
* At the end of the computation, subgroups typically require workgroup-level
  synchronization (barriers/shared memory) to combine partial results.

---

## Complete Worked Examples

### Example 1

**Iteration space:** `[d0 = parallel(4), d1 = parallel(6656), d2 = reduction(16384)]`

**Configuration:**

```mlir
lane_basis = [[1, 1, 64], [0, 1, 2]]
partial_reduction = [0, 0, 512]
subgroup_basis = [[1, 1, 1], [0, 1, 2]]
thread = [0, 0, 8]
workgroup = [4, 1, 0]
```

**Analysis:**

**Lane basis `[[1, 1, 64], [0, 1, 2]]`:** All 64 threads have `c₁ = 0` and
`c₂ = 0` (counts of 1 mean no distribution on the first two dimensions).
Only `c₃` varies from 0 to 63, distributing the 64 threads. With identity
mapping `[0, 1, 2]`, threads are distributed along `d2` (the reduction
dimension).

**Partial reduction `[0, 0, 512]`:** We tile dimension 2 (reduction) into
chunks of 512 elements, creating `16384 / 512 = 32` loop iterations. In each
iteration the workgroup processes 512 elements of partial accumulators along
`d2`.

**Thread `[0, 0, 8]`:** Each thread loads 8 elements per iteration in `d2`. With
64 threads, this yields `64 × 8 = 512` elements per iteration.

**Workgroup `[4, 1, 0]`:** The workgroup produces a `4 × 1` output tile. The
reduction dimension is handled via the partial reduction loop.

**Subgroup basis `[[1, 1, 1], [0, 1, 2]]`:** With product `1 × 1 × 1 = 1`,
there is a single 64-thread subgroup per workgroup.

---

### Example 2

**Iteration space:** `[d0 = parallel(1152), d1 = reduction(384)]`

**Configuration:**

```mlir
lane_basis = [[16, 4], [1, 0]]
partial_reduction = [0, 32]
subgroup_basis = [[1, 2], [0, 1]]
thread = [0, 1]
workgroup = [16, 0]
```

**Analysis:**

**Lane basis `[[16, 4], [1, 0]]`:** The 64 threads form a 16×4 grid in basis
space. Mapping `[1, 0]` swaps coordinates: 16 threads (`c₁`) map to iteration
dimension 1 (reduction) and 4 threads (`c₂`) map to iteration dimension 0
(parallel).

**Partial reduction `[0, 32]`:** We tile dimension 1 (reduction) into chunks of
32 elements, creating `1152 / 32 = 36` loop iterations. Each workgroup processes
32 elements of partial accumulators along `d1` per iteration.

**Thread `[0, 1]`:** Each thread loads 1 element per iteration along reduction
dimension `d1`.

**Workgroup `[16, 0]`:** The workgroup produces a 16 element tile on `d0`.

**Subgroup basis `[[1, 2], [0, 1]]`:** With product `1 × 2 = 2`, there are two
subgroups per workgroup, yielding a workgroup size of `64 × 2 = 128` threads.
The subgroups handle consecutive chunks of the reduction dimension: Subgroup 0
processes positions 0–15 per iteration; Subgroup 1 processes positions 16–31.

---

### Example 3

**Iteration space:** `[d0 = parallel(4096), d1 = reduction_1(32),
d2 = reduction_2(128)]`

**Configuration:**

```mlir
lane_basis = [[1, 1, 64], [0, 1, 2]]
partial_reduction = [0, 1, 128]
subgroup_basis = [[1, 1, 1], [0, 1, 2]]
thread = [0, 1, 2]
workgroup = [8, 0, 0]
```

**Analysis:**

**Lane basis `[[1, 1, 64], [0, 1, 2]]`:** All 64 threads are distributed along
`d2` only. The identity mapping gives a straightforward distribution.

**Partial reduction `[0, 1, 128]`:** Dimension 1 has tile size 1 (iterated
serially without chunking). Dimension 2 has tile size 128, producing chunks of
128 elements. Since `d2` has extent 128, the loop is elided (processed once).
Each iteration therefore has step size 1 and processes 128 elements
of partial accumulators along `d2` per workgroup.

**Thread `[0, 1, 2]`:** Each thread loads 1 element in `d1` and 2 elements in
`d2` per iteration, for 2 elements per thread per iteration. With 64 threads,
that yields `64 × 2 = 128` elements total.

**Workgroup `[8, 0, 0]`:** The workgroup produces an 8 element tile along `d0`.

**Subgroup basis `[[1, 1, 1], [0, 1, 2]]`:** With product `1 × 1 × 1 = 1`,
there is a single subgroup per workgroup.

---

## Summary: Reduction Config Attributes Quick Reference

| Attribute           | Key Semantic                                             |
| ------------------- | -------------------------------------------------------- |
| `workgroup`         | Output tile per workgroup                                |
| `thread`            | Elements per thread load per iteration along dimension   |
| `partial_reduction` | Chunk size for `PartialReductionOuterReduction`          |
| `lane_basis`        | Thread distribution: where each thread works in subgroup |
| `subgroup_basis`    | Subgroup distribution: where each subgroup works in      |
