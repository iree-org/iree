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
  let summary = [{A layout representing a mapping from GPU thread hierarchy to a shape.}];
  let description = [{
    This layout explicitly defines how the shape of the associated vector
    is mapped to a compute hierarchy.
    We consider the following levels of hierarchy, inspired by GPUs:

    1. Subgroups per workgroup
    2. Threads per subgroup
    3. Elements per thread

    Note that elements in a thread is also conceptually viewed as
    a 3 dimensions. i.e. elements per thread = batch x outer x element
    However, the final order of sub-dimensions are not exactly in that
    hierarchy. For e.g. a single dimensional vector say `vector< n x f16>`
    is viewed as a
    `vector<subgroup x batch x outer x thread x element>` 5 dimensional
    vector. For a two dimensional vector, each above sub-dimension would
    be doubled. i.e. `vector< n1 x n2 x f16>` is viewed as a
    `vector<subgroup1 x subgroup2 x batch1 x batch2 x ... x element1 x element2>`

    Now, when the vector<subgroup x batch x outer x thread x element> is
    indexed, the indices of 'subgroup' and `thread` are not directly refferring
    to the subgroup_id and thread_id in the GPU context. lets define them
    as virtual_subgroup_id and virtual_thread_id and they hold the following
    definition:
    ```
    virtual_subgroup_id[i] = (subgroup_id / subgroup_stride[i]) % subgroup_tile_size[i]
    virtual_thread_id[i]   = (thread_id   / thread_stride[i]) % thread_tile_size[i]
    ```

    the inverse mapping would be:
    ```
    subgroup_id = sum_i(subgroup_stride[i] * virtual_subgroup_id[i]) % mul_i(subgroup_tile_size[i])
    thread_id = sum_i(thread_stride[i] * virtual_thread_id[i]) % mul_i(thread_tile_size[i])
        for i = [0 : rank(undistributed_vector)]
    ```

    NOTE: if stride is zero, it represents non-distribution of that
    dimension on that hierarchy.

    We now describe each level of tiling. Each level of tiling represents a
    count of tiles over the next level (rather than a list of tile sizes).

    #### Subgroups per Workgroup

    This level of tiling is also known as "subgroup/warp distribution". It
    represents how the vector is distributed into subgroups.

    For example, consider distributing `vector<4x2xf16>` to a
    `subgroup_tile=[4, 2], subgroup_stride=[1, 4]` will
    arrange the subgroups in the order:

    ```
    virtual_subgroups_ids:
    [0][0] , [0][1] , [1][0], [1][1], [2][0], [2][1], [3][0], [3][1]
    subgroups_ids:
    0, 4, 1, 5, 2, 6, 3, 7
    ```

    The subgroups are placed contiguously with their shape and ordering
    determined by:
      - `subgroup_tile`: Sizes of this level of tiling
      - `subgroup_strides`: Stride of this level of tiling. 0 if not distributed.
    Tiling levels must not overlap.

    The total number of subgroups used (computed by multiplying each dim in
    subgroup_tile) should be a multiple of number of subgroups in the
    harware. If the total number of subgroups used exceeds the number of
    subgroups of the hardware, then the subgroup used (say x) is
    x mod num_subgroups:

    ```
    num_subgroups = 4

    0, 4, 1, 5, 2, 6, 3, 7
    | mod 4
    V
    0, 0, 1, 1, 2, 2, 3, 3
    ```

    #### Threads per Subgroup:

    This level of tiling is also known as "thread distribution" within a subgroup.
    The logic is quite similiar to subgroup distribution using the tile sizes
    and the 'thread_strides'.

    #### Element distribution on a thread

    So after the vector is distributed per thread
    on a subgroup, it is viewed as [batch] x [outer] x [element]
    where each sub-dimensions group has dimensions equal
    to original rank of the undistributed vector.

    The first level, batches, are a way to represent instruction unrolling. For
    example, an intrinsic which can only take 4x4 shape at a time, uses batches
    to unroll a 16x16 shape to the native intrinsice shape.

    The second level, outers, is a way to represent thread layout duplication
    required by a particular intrinsic. For example, some AMDGPU matrix
    multiplication variants require threads to be distributed
    like:

    E.g.: `outer_tile=[2, 1], thread_tile=[2, 5]`
    the thread Layout of shape 2x5 duplicated 2 times, to get a layout of shape 4x5

    ```
    outer = 0,0 :
    [0 1 2 3 4]
    [5 6 7 8 9]

    outer = 1,0 :
    [0 1 2 3 4]
    [5 6 7 8 9]
    ```

    `outer_tile` represents the number of outers in a batch.

    The final level of tiling, representing the minimum shape of vector that
    is treated as an atom.

    `element_tile` represents the native size of the vector.

    #### A full example

    Vector to be distributed: `vector<64x64xf16>`
    ```
    NestedLayout : <
        subgroup_tile = [2, 1],
        batch_tile = [2, 4],
        outer_tile = [1, 1],
        thread_tile = [16, 4],
        element_tile = [1, 4],
        subgroup_strides = [1, 0],
        thread_strides = [1, 16]
    >
    ```

    This is conceptually viewed as a: `vector<[2x1]x[2x4]x[1x1]x[16x4]x[1x4]>`
    where the first groups of sub-dimensions
    represent the distribution into subgroups.
    The subgroup_strides being [1, 0] means
    each subgroup is going to get a vector
    as follows:

    ```
    subgroup0 : vector<[2x4]x[1x1]x[16x4]x[1x4]>
    from vector<[2x1]x[2x4]x[1x1]x[16x4]x[1x4]>[0,:,:,:,:,:,:,:,:,:]
    subgroup1 : vector<[2x4]x[1x1]x[16x4]x[1x4]>
    from vector<[2x1]x[2x4]x[1x1]x[16x4]x[1x4]>[1,:,:,:,:,:,:,:,:,:]
    subgroup2 : vector<[2x4]x[1x1]x[16x4]x[1x4]>
    from vector<[2x1]x[2x4]x[1x1]x[16x4]x[1x4]>[0,:,:,:,:,:,:,:,:,:]
    subgroup3 : vector<[2x4]x[1x1]x[16x4]x[1x4]>
    from vector<[2x1]x[2x4]x[1x1]x[16x4]x[1x4]>[1,:,:,:,:,:,:,:,:,:]
    ```

    Then each vector<[2x4]x[1x1]x[16x4]x[1x4]>
    is distributed threads in a subgroup using
    thread_strides = [1, 16]

    recall: `thread_id = sum_i(thread_stride[i] * virtual_thread_id[i]) % mul_i(thread_tile_size[i])`

    ```
    thread0 : vector<[2x4]x[1x1]x[1x4]>
    from vector<[2x4]x[1x1]x[16x4]x[1x4]>[:,:,:,:,0,0,:,:]
    thread1 : vector<[2x4]x[1x1]x[1x4]>
    from vector<[2x4]x[1x1]x[16x4]x[1x4]>[:,:,:,:,1,0,:,:]
    ...
    ...
    thread16 : vector<[2x4]x[1x1]x[1x4]>
    from vector<[2x4]x[1x1]x[16x4]x[1x4]>[:,:,:,:,0,1,:,:]
    ```

    Finally we are left with a distributed vector
    of conceptual view : `vector<[2x4]x[1x1]x[1x4]>`
    where the actual shape is : `vector<2x16>`.

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
                     "ArrayRef<int64_t>":$threadStrides)>,
    AttrBuilder<(ins "NestedLayoutAttr":$source,
                     "ArrayRef<int64_t>":$appendSubGroupLens,
                     "ArrayRef<int64_t>":$appendBatchLens,
                     "ArrayRef<int64_t>":$appendOuterLens,
                     "ArrayRef<int64_t>":$appendThreadLens,
                     "ArrayRef<int64_t>":$appendElementLens,
                     "ArrayRef<int64_t>":$appendSubgroupStrides,
                     "ArrayRef<int64_t>":$appendThreadStrides)>
  ];

  let extraClassDeclaration = [{
    // Returns the subgroup/lane ids delinearized from a single linearized
    // thread ID. Returns the empty vector on failure.
    SmallVector<Value> computeThreadIds(Value threadId, int64_t subgroupSize, RewriterBase &rewriter) const;

    // Get the undistributed shape that is subgroup x batch x outer x thread x element
    SmallVector<int64_t> getUndistributedPackedShape() const;

    // Get the subgroup x batch x outer x thread x element shape for original dim
    SmallVector<int64_t> getPackedShapeForUndistributedDim(int64_t dim) const;

    // Get the distributed shape but has the same rank as the undistributed shape.
    SmallVector<int64_t> getDistributedUnpackedShape() const;
  }];

  let genVerifyDecl = 1;
}

#endif // IREE_DIALECT_VECTOREXT_ATTRS
