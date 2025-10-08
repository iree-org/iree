---
icon: octicons/file-symlink-file-16
---

# Demystifying IREE Lowering Configs

## Overview

Lowering configs are attributes that guide IREE in effectively lowering operations from the tensor level down to the vector level. They are determined by:
1. The type of computation being performed (e.g., matmul, reduction, convolution)
2. Hardware attributes (e.g., subgroup size, memory bandwidth, compute units)
3. Optional tuner refinements for performance optimization

---

## LLVMGPU Vector Distribute Pipeline

### Reduction
This pipeline is used when a dispatch contains at least one reduction operation and targets efficient reduction strategies.

---

## Relevant lowering config attributes

### 1. `workgroup` Tile Sizes

**Definition:** The output tile that each workgroup will compute.

**Format:** Array of integers, one per iteration dimension.

**Semantics:**
- `workgroup[d] > 0`: Each workgroup produces this many elements in dimension `d`

**Example:**
```mlir
workgroup = [16, 0]
```
- Dimension 0 (parallel): Each workgroup produces 16 output elements

---

### 2. `thread` Tile Sizes

**Definition:** The number of elements each thread processes per load per iteration along reduction dimensions.

**Format:** Array of integers, one per iteration dimension.

**Semantics:**
- `thread[d] = N`: Each thread loads `N` elements per iteration of the reduction loop

**Example:**
```mlir
thread = [0, 8]
```
- Dimension 1 (reduction): Each thread loads 8 elements per loop iteration along dimension 1.

---

### 3. `partial_reduction` Tile Sizes

**Applies to:** Reduction dimensions only.

**Tiling strategy:** We use `PartialReductionOuterReduction` as our reduction tiling strategy at the partial reduction level. This strategy tiles the reduction dimension according to `r -> r_outer, r_partial`, where we create a serial loop over `r_outer` with step size equal to `partial_reduction[d]`. Within each iteration, threads maintain partial accumulators as they process `r_partial` elements. At the end, partial results are merged.

This config informs us of the chunk size `S` we tile with respect to `PartialReductionOuterReduction` (r -> r/S, S) in the reduction dimension.

**Format:** Array of integers, one per iteration dimension.

**Semantics:**
- `partial_reduction[d] = 0`: Dimension `d` is not a reduction dimension
- `partial_reduction[d] = S`: Tile the reduction dimension into chunks of size `S`

**Number of iterations:**
```
iterations = ⌈reduction_size / partial_reduction[d]⌉
```

**Special case:** If `reduction_size / partial_reduction[d] = 1`, there is only one iteration and the outer loop can be elided.

**Example:**
```mlir
partial_reduction = [0, 512]
```
- Dimension 0: Not a reduction dimension
- Dimension 1: Process reduction in chunks of 512 elements

For a reduction of size 16384:
- Loop iterations: 16384 / 512 = 32
- Each iteration: threads within the subgroup process 512 elements per iteration along dimension 1.

---

### 4. `lane_basis` (Thread Distribution within Subgroup)

**Purpose:** Defines how threads within a single subgroup are distributed across the iteration space.

**Format:** `[[counts], [mapping]]`
- `counts`: Array of thread counts per basis dimension
- `mapping`: Permutation array mapping basis coordinates to iteration dimensions

#### The `counts` Array

**Definition:** Number of threads spanning each basis dimension.

**Constraint:** The product of all counts equals the subgroup size.

**Example:**
```mlir
lane_basis = [[16, 4], [1, 0]]
counts = [16, 4]
```

For a subgroup of 64 threads:
- 16 × 4 = 64
- This creates a conceptual 16×4 grid of threads in basis space

**Why 1 is the default, not 0:**
- Counts are multiplicative (used in product calculations)
- `count = 1` means "no distribution along this dimension" (acts as multiplicative identity)
- `count = 0` would make the product zero, which is invalid

#### The `mapping` Array

**Definition:** A permutation that maps basis coordinates to iteration space dimensions.

**Semantics:**
```
mapping[j] = i  means:  iteration_dim[i] ← basis_coordinate[j+1]
```

**Example:**
```mlir
mapping = [1, 0]
```
This swaps/transposes the coordinates:
- Basis coordinate c₁ maps to iteration dimension 1
- Basis coordinate c₂ maps to iteration dimension 0

#### Computing Thread Position (Step-by-Step)

For a given thread ID `x`, we can calculate its position within the iteration space using the following process:

**Step 1: Delinearize** the thread ID using counts to get basis coordinates.

**Formula**:

For basis `[B₀, B₁, ..., Bₙ₋₁]` producing `n+1` results `(c₀, c₁, ..., cₙ)`:

Let `P_i` = product of basis elements from position `i` onward:
```
P_n = 1
P_{n-1} = B_{n-1}
P_{n-2} = B_{n-2} × B_{n-1}
...
P_0 = B_0 × B_1 × ... × B_{n-1}
```

Then:
```
c₀ = x ÷ P_0                    (outer bound, typically ignored)
c_i = (x mod P_{i-1}) ÷ P_i    for i = 1, 2, ..., n
```

**Step 2: Apply mapping** to get iteration space coordinates.

```
For each basis coordinate c_j (j = 1, 2, ..., n):
  iteration_dim[mapping[j-1]] = c_j
```

#### Concrete Example: Thread 42 with `[[16, 4], [1, 0]]`

**Step 1: Delinearize(42, [16, 4])**
```
Basis: [16, 4]

Calculate products:
  P_2 = 1
  P_1 = 4 × 1 = 4
  P_0 = 16 × 4 = 64

Apply formula:
  c₀ = 42 ÷ 64 = 0       (outer bound, typically ignored)
  c₁ = (42 mod 64) ÷ 4 = 42 ÷ 4 = 10
  c₂ = (42 mod 4) ÷ 1 = 2

Basis coordinates: [c₁, c₂] = [10, 2]
```

**Step 2: Apply mapping [1, 0]**
```
mapping[0] = 1  →  iteration_dim[1] = c₁ = 10
mapping[1] = 0  →  iteration_dim[0] = c₂ = 2

Iteration coordinates: [dim0=2, dim1=10]
```

**Result:** Thread 42 works at position [parallel=2, reduction=10] in the iteration space.

**Visual interpretation:**
```
Threads form a 16×4 grid in basis space:
       col0 col1 col2 col3
row0:   T0   T1   T2   T3
row1:   T4   T5   T6   T7
...
row10:  T40  T41  T42  T43  ← Thread 42 at (row=10, col=2)
...

```

---

### 5. `subgroup_basis` (Subgroup Distribution within Workgroup)

**Purpose:** Defines how multiple subgroups within a workgroup are distributed across the iteration space.

**Format:** Same as `lane_basis`: `[[counts], [mapping]]`

**Constraint:**
```
product(counts) = number_of_subgroups_per_workgroup

Workgroup size = subgroup_size × product(subgroup_basis.counts)
```

**Example:**
```mlir
subgroup_basis = [[1, 2], [0, 1]]
subgroup_size = 64

Number of subgroups: 1 × 2 = 2
Workgroup size: 64 × 2 = 128 threads
```

**Interpretation:**
- Delinearization works identically to `lane_basis`, but is applied to subgroup IDs instead of thread IDs
- Subgroup 0 and Subgroup 1 handle different chunks of the iteration space
- Distribution is determined by applying the same delinearize-then-map process
- At the end of the computation, subgroups typically require workgroup-level synchronization (barriers and shared memory) to combine their partial results

---

## Complete Worked Examples

### Example 1:

**Iteration space:** `[d0=batch(4), d1=parallel(6656), d2=reduction(16384)]`

**Configuration:**
```mlir
lane_basis = [[1, 1, 64], [0, 1, 2]]
partial_reduction = [0, 0, 512]
subgroup_basis = [[1, 1, 1], [0, 1, 2]]
thread = [0, 0, 8]
workgroup = [4, 1, 0]
```

**Analysis:**

**Lane basis `[[1, 1, 64], [0, 1, 2]]`:**
All 64 threads have c₁=0 and c₂=0 (counts of 1 mean no distribution on first two dimensions). Only c₃ varies from 0 to 63, distributing the 64 threads. With identity mapping [0,1,2], threads are distributed along d2 (the reduction dimension).

**Partial reduction `[0, 0, 512]`:**
We tile dimension 2 (reduction) into chunks of 512 elements. This creates 16384 / 512 = 32 loop iterations, where for each iteration the subgroup processes 512 elements worth of partial accumulators along d2.

**Thread `[0, 0, 8]`:**
Each thread loads 8 elements per iteration in d2. With 64 threads, this gives us 64 × 8 = 512 elements total per iteration.

**Workgroup `[4, 1, 0]`:**
The workgroup produces a 4×1 output tile, handling 4 batch elements and 1 parallel element. The reduction dimension is handled via the partial reduction loop.

**Subgroup basis `[[1, 1, 1], [0, 1, 2]]`:**
With product 1×1×1 = 1, we have a single 64-thread subgroup per workgroup.

---

### Example 2: 

**Iteration space:** `[d0=parallel(384 columns), d1=reduction(1152 rows)]`

**Configuration:**
```mlir
lane_basis = [[16, 4], [1, 0]]
partial_reduction = [0, 32]
subgroup_basis = [[1, 2], [0, 1]]
thread = [0, 1]
workgroup = [16, 0]
```

**Analysis:**

**Lane basis `[[16, 4], [1, 0]]`:**
The 64 threads are arranged as a 16×4 grid in basis space. The mapping [1, 0] swaps coordinates, so 16 threads (c₁) map to iteration dimension 1 (reduction) and 4 threads (c₂) map to iteration dimension 0 (parallel).

**Partial reduction `[0, 32]`:**
We tile dimension 1 (reduction) into chunks of 32 elements, creating 1152 / 32 = 36 loop iterations. Within each iteration we process 32 elements worth of partial accumulators along d1. 

**Thread `[0, 1]`:**
Each thread loads 1 element per iteration along the reduction dimension 1.

**Workgroup `[16, 0]`:**
The workgroup produces 16 output elements (one per parallel dimension element).

**Subgroup basis `[[1, 2], [0, 1]]`:**
With product 1×2 = 2, we have 2 subgroups per workgroup, giving a total workgroup size of 64 × 2 = 128 threads. The subgroups handle consecutive chunks of the reduction dimension: Subgroup 0 processes reduction positions 0-15 in each iteration, while Subgroup 1 processes positions 16-31.

---

### Example 3: 

**Iteration space:** `[d0=parallel(4096), d1=reduction_1(32), d2=reduction_2(128)]`

**Configuration:**
```mlir
lane_basis = [[1, 1, 64], [0, 1, 2]]
partial_reduction = [0, 1, 128]
subgroup_basis = [[1, 1, 1], [0, 1, 2]]
thread = [0, 1, 2]
workgroup = [8, 0, 0]
```

**Analysis:**

**Lane basis `[[1, 1, 64], [0, 1, 2]]`:**
All 64 threads are distributed along d2 only. The identity mapping means straightforward distribution.

**Partial reduction `[0, 1, 128]`:**
Dimension 1 has tile size 1 (iterate serially with no chunking). Dimension 2 has tile size 128, creating chunks of 128 elements. Since the dimensionality of d2 is 128, the loop is elided as we only process this once. So each iteration we process 128 elements worth of partial accumulators along the reduction dimension per workgroup.

**Thread `[0, 1, 2]`:**
Each thread loads 1 element in d1 and 2 elements in d2 per iteration, giving 2 elements total per thread per iteration. With 64 threads, this gives us 64 × 2 = 128 elements total.

**Workgroup `[8, 0, 0]`:**
The workgroup produces 8 output elements.

**Subgroup basis `[[1, 1, 1], [0, 1, 2]]`:**
With product 1×1×1 = 1, we have a single subgroup per workgroup.

---

## Summary: Config Attributes Quick Reference

| Attribute | Key Semantic |
|-----------|--------------|
| `workgroup` | Output tile per workgroup |
| `thread` | Elements per thread per load per iteration |
| `partial_reduction` | Chunk size for `PartialReductionOuterReduction` tiling strategy |
| `lane_basis` | Thread distribution: Where each thread works within subgroup |
| `subgroup_basis` | Subgroup distribution: Where each subgroup works within workgroup |