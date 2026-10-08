---
icon: octicons/cpu-16
---
# AMDGPU LDS Transpose Reads

## Overview

AMD GPUs have a dedicated shared memory (LDS, "Local Data Share") load
family that reads a matrix from LDS and returns the **transposed** tile to the
lanes that issued the read: the `ds_read_tr` family of instructions on CDNA
(referred to as the `ds_read_b{N}_tr_b{M}` instructions in the ISA manual), and
the `ds_load_tr` family on newer (gfx1250+) chips.

These instructions matter for GEMM-style kernels. When an operand is staged in
LDS in a row-major layout, computing along the other dimension requires each
lane to read a *column* of the buffer. Naive column reads issue one scalar LDS
load per lane and hit repeated LDS bank conflicts. A transpose read instead lets
a group of lanes collectively fetch the tile with a single instruction per lane,
with the hardware performing the transpose, avoiding both the extra
instructions and the bank conflicts.

Neither the ISA manual nor the MLIR `AMDGPU` dialect documentation spells out
the exact semantics that a compiler needs to respect, so this page documents
what IREE relies on: the instruction shapes, the MLIR wrapper op, and the pass
that generates the ops from ordinary `vector.transfer_read` code.

## Instruction shapes

The instructions operate on **groups of 16 consecutive lanes**. Each lane of the
group issues one instruction that reads a vector of elements starting at that
lane's address, and the 16 lanes collectively receive the transposed tile. The
valid element type / element count combinations, and the ROCDL intrinsics they
lower to (see `TransposeLoadOpLowering` in LLVM's `AMDGPUToROCDL`
conversion pass), are:

| Element bit width | Elements per lane | gfx950 intrinsic   | gfx1250+ intrinsic  |
| ----------------- | ----------------- | ------------------ | ------------------- |
| 4                 | 16                | `ds_read_tr4_b64`  | `ds_load_tr4_b64`   |
| 6                 | 16                | `ds_read_tr6_b96`  | `ds_load_tr6_b96`   |
| 8                 | 8                 | `ds_read_tr8_b64`  | `ds_load_tr8_b64`   |
| 16                | 4                 | `ds_read_tr16_b64` | `ds_load_tr16_b128` |

A transpose load is only usable when the access pattern of the whole 16-lane
group is known up front: all lanes of the group must read from the same
transposed tile, meaning the row index must be uniform across the 16-lane group
while the column indices advance by one per lane. Any subgroup size that is not
a multiple of 16 therefore cannot be tiled with these instructions. The
requirements IREE enforces are listed
[below](#matching-requirements).

## MLIR representation: `amdgpu.transpose_load`

IREE models the instruction with the MLIR `AMDGPU` dialect op
`amdgpu.transpose_load` (from [llvm-project](https://github.com/llvm/llvm-project),
`mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPUOps.td`):

```mlir
%0 = amdgpu.transpose_load %src[%row, %col] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
```

Per the dialect documentation, the op represents "a subgroup load from LDS
memory, where the subgroup of threads collectively reads a matrix from the
source memref, with each thread reading a vector of the matrix, and gets a
transposed matrix as the result. That is, each thread reads a vector of the
col-major matrix at different indices, and the thread's read result is a vector
of the corresponding row of the transposed matrix."

The op is lowered to the intrinsics listed above by LLVM's
`AMDGPUToROCDL` pass; IREE never emits the intrinsics directly.

## How IREE generates transpose loads

The pass `iree-rocdl-load-to-transpose-load`
(`ROCDLLoadToTransposeLoadPass`, defined in
`compiler/src/iree/compiler/Codegen/LLVMGPU/ROCDLLoadToTransposeLoad.cpp`)
rewrites `vector.transfer_read` operations into `amdgpu.transpose_load`
operations when the access pattern matches the instruction requirements.

The pass runs as **step 8** of the tile-and-fuse codegen pipeline (see
`addGPUTileAndFusePassPipeline` in
`compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`, enabled only for ROCDL
targets), i.e. after
`UnrollToIntrinsicsPass` but before memref flattening and before
non-contiguous vector loads are unrolled to scalar loads. This placement
matters: the analysis needs the original memref shapes and must see column
accesses as single vector loads rather than scalarized loads. It can be
disabled for testing with `--iree-llvmgpu-test-load-to-transpose-load=false`.

### Lane index hints

The pass must prove, per lane, how each index of a load varies across the
subgroup. This information is carried by `iree_codegen.index_hint` ops wrapping
indices, with `#iree_gpu.lane_constant<N>` (constant within groups of `N`
consecutive lanes) or `#iree_gpu.lane_increment<N, step, aligned>` (increments
by `step` per lane, wrapping at `N`) attributes.

The pass first **seeds** hints on `gpu.thread_id` ops from the known workgroup
size (`seedThreadIdHints`):

* `thread_id x` &rarr; `lane_increment<wgSizeX, step = 1, aligned>`
* `thread_id y` &rarr; `lane_constant<wgSizeX>`
* `thread_id z` &rarr; `lane_constant<wgSizeX * wgSizeY>`

and then **propagates** them through `affine.delinearize_index` and simple
`arith`/`affine` arithmetic (`PropagateHintThroughDelinearize`), all in one
greedy rewrite driver together with the conversion pattern below, so hints reach
every derived index at fixpoint. The pass-local hints are removed again after
the patterns are applied.

### Matching requirements

A `vector.transfer_read` is converted to `amdgpu.transpose_load`
(`TransferReadToTransposeLoad` / `analyzeTransferReadForTransposeLoad`) only
when **all** of the following hold:

* The source memref is in the **workgroup** (LDS) address space, and the value
  read is a full view of the allocation: a `memref.alloc`, optionally followed
  by `memref.expand_shape`/`memref.collapse_shape` ops. Subviews are not
  supported, since their index arithmetic is not analyzed.
* The result vector's **innermost dimension has size 1** — that dimension is
  the column being read, and the transpose hardware produces the column data
  across lanes.
* The permutation map is a projected permutation.
* The **column index** is defined by an `index_hint` with `lane_increment`
  where the group size is a multiple of 16 and `step = 1` (consecutive lanes
  read consecutive columns).
* The **column memref dimension has stride 1** (contiguous).
* Every **row index** is uniform within each 16-lane group: it must derive only
  from constants and `lane_constant` hints (group size a multiple of 16),
  combined through `arith`/`affine` ops. Indices of unknown provenance (e.g.
  block arguments) are rejected.
* The element type is 8- or 16-bit wide (4/6-bit types are not yet handled), and
  the total row size is a multiple of the per-lane element count (8 for 8-bit,
  4 for 16-bit).

### The rewrite

When the pattern matches, the load is replaced by one or more
`amdgpu.transpose_load` ops (`generateTransposeLoads`):

* Lane indices are remapped to the instruction's addressing scheme:
  `rowGroupIdx = (lane_id % 16) / (16 / elemCount)` selects the row offset for
  this lane's instruction, and the column index becomes
  `col - (lane_id % 16) + ((lane_id % 16) % (16 / elemCount)) * elemCount`.
* Larger row vectors are unrolled into `elemCount`-wide transpose loads and the
  pieces are combined with `vector.insert_strided_slice` and
  `vector.shape_cast` back into the original result shape.

For example (from
`compiler/src/iree/compiler/Codegen/LLVMGPU/test/rocdl_load_to_transpose_load.mlir`):

```mlir
// Input: each lane reads column `tid` of a row-major LDS buffer
%row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
%col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
%0 = vector.transfer_read %src[%row, %col], %cst
     {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
     : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>

// Output: hardware transpose read, 4 f16 elements per lane
//   row_offset = (lane_id % 16) / 4
//   new_col    = (col - (lane_id % 16)) + ((lane_id % 16) % 4) * 4
%lane_id = gpu.lane_id
%rem16 = arith.remui %lane_id, %c16 : index
%row_offset = arith.divui %rem16, %c4 : index
%sub = arith.subi %col, %rem16 : index
%rem4 = arith.remui %rem16, %c4 : index
%mul = arith.muli %rem4, %c4 : index
%new_col = arith.addi %sub, %mul : index
%new_row = arith.addi %row, %row_offset : index
%load = amdgpu.transpose_load %src[%new_row, %new_col]
        : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
%1 = vector.shape_cast %load : vector<4xf16> to vector<4x1xf16>
```

## When the pass applies

The pass is a no-op unless the target chipset supports the instructions. As of
this writing:

* **gfx950 (CDNA4, e.g. MI350)**: LDS transpose loads (`ds_read_tr` family)
  are enabled, for reads from workgroup memory as described above.
* **RDNA4 (gfx1200/gfx1201)**: the same pass only rewrites a *global memory*
  pattern into `amdgpu.global_transpose_load` (see below); the LDS path is
  gfx950-only.
* **gfx1250+**: supported by the `AMDGPU` dialect (`ds_load_tr` family) but not
  yet enabled in IREE.

Because each instruction covers a 16-lane group, the pass relies on subgroup
behavior that only holds when the workgroup size along the varying dimension is
a multiple of 16; workgroups smaller than that cannot use transpose reads.

## Related: global memory transpose loads (`global_load_tr`)

The same pass also matches RDNA4's `global_load_tr` instructions (the
`amdgpu.global_transpose_load` op, gfx1200+), which perform the analogous
transpose for reads straight from **global** memory
(`TransferReadTransposeToGlobalTransposeLoad` in the same pass file). The
matched pattern is a `vector<Nx1>` global `transfer_read` (the per-lane
`K`-element row) followed by a `vector.transpose` with permutation `[1, 0]` and
a `transfer_write` into workgroup memory: instead of the N scalar global loads
plus a software transpose, a single `global_load_tr` per lane loads the
transposed tile directly (8 x 8 tiles of 8-bit elements, 8 lanes per tile).

## Testing and debugging

* Lit tests:
  `compiler/src/iree/compiler/Codegen/LLVMGPU/test/rocdl_load_to_transpose_load.mlir`
  (unit tests of the pass, including all rejection cases), plus
  `.../test/ROCDL/pipeline_tile_and_fuse_gfx950.mlir` for the full pipeline.
* Run just this pass with:
  `iree-opt --iree-gpu-test-target=gfx950 --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))'`.
* Pass failures during pattern matching are logged with
  `--debug-only=iree-rocdl-load-to-transpose-load` (the `LDBG()` traces in
  `ROCDLLoadToTransposeLoad.cpp` explain exactly which requirement failed).
