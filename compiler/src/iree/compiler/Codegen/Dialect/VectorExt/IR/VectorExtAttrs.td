// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_ATTRS
#define IREE_DIALECT_VECTOREXT_ATTRS

include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtBase.td"

//===---------------------------------------------------------------------===//
// Vector layout attributes
//===---------------------------------------------------------------------===//

def NestedLayoutAttr : IREEVectorExt_Attr<"NestedLayout",
      [ DeclareAttrInterfaceMethods<VectorLayoutInterface> ]> {
  let mnemonic = "nested_layout";
  let summary = [{A layout representing a mapping from GPU thread hierarchy to a shape}];
  let description = [{
    This layout explicitly defines how a shape is mapped to a compute
    hierarchy. We consider the following levels of hierarchy, inspired by GPUs:

    1. Subgroups per Workgroup
    2. Threads per Subgroup
    3. Elements per Thread

    Conceptually, each higher level of hierarchy can be viewed as multiple
    tiles of the lower level of hierarchy; each lower level of hierarchy is
    nested in the higher level of hierarchy. The last level represents the
    final elements in memory.

    The conceptual mapping is leveraged during compilation for tiling and
    distributing to hardware for parallel computation. Concretely, the mapping
    is done on each dimension of the original vector shape. For example, for
    vector shape 16x16x16, we have 3 dimensions, so at each level of the
    hierarchy we would have 3 tile sizes. Similarly for vector shape 32x32, we
    would have 2-D tile sizes per compute hierarchy level.

    We now describe each level of tiling. Each level of tiling represents a
    count of tiles over the next level (rather than a list of tile sizes).

    1. Subgroups per Workgroup

    This level of tiling is also known as "subgroup/warp distribution". It
    represents how subgroups are distributed in a workgroup.

    The subgroups are placed contiguously with their shape and ordering
    determined by:
      - `subgroup_tile`: Sizes of this level of tiling
      - `subgroup_order`: Ordering of dimensions, from outermost to innermost

    For example, subgroup_tile=[4, 2], subgroup_order=[1, 0] will
    arrange the subgroups in the order:

    0 4
    1 5
    2 6
    3 7

    The total number of subgroups used (computed by multiplying each dim in
    subgroup_tile) should be a multiple of number of subgroups in the
    harware. If the total number of subgroups used exceeds the number of
    subgroups of the hardware, then the subgroup used (say x) is
    x mod num_subgroups:

    num_subgroups = 4

    0 4               0 0
    1 5    x mod 4    1 1
    2 6    ------->   2 2
    3 7               3 3

    2. Threads per Subgroup:

    Threads in a subgroup are distributed in three levels.

    The first level, batches, are a way to represent instruction unrolling. For
    example, an intrinsic which can only take 4x4 shape at a time, uses batches
    to unroll a 16x16 shape to the native intrinsice shape.

    Batches can be thought of as loops around the original layout:

    for b_0 in range(batch_0):
      for b_1 in range(batch_1):
        ...

    `batch_tile` represents the range of each loop.

    The second level, outers, is a way to represent thread layout duplication
    required by a particular intrinsic. For example, some AMDGPU matrix
    multiplication variants require threads to be distributed
    like:

    0 1 2 3 4
    5 6 7 8 9
    --------- --> Thread Layout of shape 2x5 duplicated 2 times, to get a layout of shape 4x5
    0 1 2 3 4     outer_tile=[2, 1]
    5 6 7 8 9     thread_tile=[2, 5]

    `outer_tile` represents the number of outers in a batch.

    Finally, threads are distributed in a single outer. The thread
    distribution is represented by:

      - thread_tile: Sizes of this level of tiling
      - thread_order: Ordering of dimensions, from outermost to innermost

    Examples of thread distribution over a 8x4 shape:

    {
      batch_tile = [2, 1]
      outer_tile = [2, 2]
      thread_tile = [2, 2]

      thread_order = [1, 0]
    }

    Distributed tile:

    {
      [0 2]|[0 2]      0,1,2,3 --> thread ids
      [1 3]|[1 3]
      ------------     [x z]   --> a single outer tile
      [0 2]|[0 2]      [y w]
      [1 3]|[1 3]
    }{
      [0 2]|[0 2]      { ... } --> a single batch tile
      [1 3]|[1 3]
      ------------
      [0 2]|[0 2]
      [1 3]|[1 3]
    }

    So, the thread distribution looks like:

    [0 2 0 2]
    [1 3 1 3]
    [0 2 0 2]
    [1 3 1 3]
    [0 2 0 2]
    [1 3 1 3]
    [0 2 0 2]
    [1 3 1 3]

    The total number of threads used (computed by multiplying each dim in
    thread_tile) should be a multiple of subgroup size of the
    harware. If the total number of threads used exceeds the subgroup size of
    the hardware, then the threads used (say tid) is tid mod subgroup_size:

    subgroup_size = 4

    0 1                0 0
    2 3    tid mod 4   1 1
    4 5    -------->   2 2
    6 7                3 3

    3. Elements per Thread

    The final level of tiling, representing the minimum shape of vector that
    is treated as an atom.

    `element_tile` represents the native size of the vector.
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"int64_t", "subgroup_tile">:$subgroupTile,
    OptionalArrayRefParameter<"int64_t", "batch_tile">:$batchTile,
    OptionalArrayRefParameter<"int64_t", "outer_tile">:$outerTile,
    OptionalArrayRefParameter<"int64_t", "thread_tile">:$threadTile,
    OptionalArrayRefParameter<"int64_t", "element_tile">:$elementTile,

    OptionalArrayRefParameter<"int64_t", "subgroup_strides">:$subgroupStrides,
    OptionalArrayRefParameter<"int64_t", "thread_strides">:$threadStrides
  );

  let assemblyFormat = [{
    `<` `subgroup_tile`     `=` `[` (`]`) : ($subgroupTile^ `]`)? `,`
        `batch_tile`        `=` `[` (`]`) : ($batchTile^ `]`)? `,`
        `outer_tile`        `=` `[` (`]`) : ($outerTile^ `]`)? `,`
        `thread_tile`       `=` `[` (`]`) : ($threadTile^ `]`)? `,`
        `element_tile`      `=` `[` (`]`) : ($elementTile^ `]`)? `,`
        `subgroup_strides`  `=` `[` (`]`) : ($subgroupStrides^ `]`)? `,`
        `thread_strides`    `=` `[` (`]`) : ($threadStrides^ `]`)?
    `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$subgroupTile,
                     "ArrayRef<int64_t>":$batchTile,
                     "ArrayRef<int64_t>":$outerTile,
                     "ArrayRef<int64_t>":$threadTile,
                     "ArrayRef<int64_t>":$elementTile,
                     "ArrayRef<int64_t>":$subgroupStrides,
                     "ArrayRef<int64_t>":$threadStrides)>
  ];

  let extraClassDeclaration = [{
    // Returns the subgroup/lane ids delinearized from a single linearized
    // thread ID.
    SmallVector<Value> computeThreadIds(Value threadId, int64_t subgroupSize, RewriterBase &rewriter) const;

    // Get the undistributed shape that is subgroup x batch x outer x thread x element
    SmallVector<int64_t> getUndistributedPackedShape() const;
  }];

  let genVerifyDecl = 1;
}

#endif // IREE_DIALECT_VECTOREXT_ATTRS
