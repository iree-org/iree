// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_ATTRS
#define IREE_DIALECT_VECTOREXT_ATTRS

include "iree-dialects/Dialect/VectorExt/IR/VectorExtBase.td"

//===---------------------------------------------------------------------===//
// Vector layout attributes
//===---------------------------------------------------------------------===//

// Defines the batch dimensions for the original SIMD tensor.
// By convention, X is along rows and Y along columns.
def BATCHX : I32EnumAttrCase<"BATCHX", 0>;
def BATCHY : I32EnumAttrCase<"BATCHY", 1>;
// Defines the vector dimension.
def VECTORX : I32EnumAttrCase<"VECTORX", 2>;
def VECTORY : I32EnumAttrCase<"VECTORY", 3>;
def VECTORZ : I32EnumAttrCase<"VECTORZ", 4>;
// Defines the lane dimensions.
def LANEX : I32EnumAttrCase<"LANEX", 5>;
def LANEY : I32EnumAttrCase<"LANEY", 6>;
def LANEZ : I32EnumAttrCase<"LANEZ", 7>;

def LayoutDimension : IREEVectorExt_I32EnumAttr<"LayoutDimension",
    "Describes the dimension of the high-dimensional layout", [
      BATCHX,
      BATCHY,
      VECTORX,
      VECTORY,
      VECTORZ,
      LANEX,
      LANEY,
      LANEZ,
    ]>;

def LayoutDimensionAttr : IREEVectorExt_EnumAttr<LayoutDimension, "dimension">;

def PerDimLayoutAttr : IREEVectorExt_Attr<"PerDimLayout"> {
   let mnemonic = "per_dim_layout";
   let summary = [{high-dimensional vector register layout for a given vector dimension}];
   let description = [{
    This attribute describes the per dimension register layout for a given vector
    that could be prescribed by an operator such as matrix multiplication.
    This is a way to explicitly represent the layout in the IR
    when it is in the SIMD form prior to converting to the SIMT form so that
    we can reason about layouts, propagating layouts and layout conflicts.
   }];
   let parameters = (ins
     ArrayRefParameter<"LayoutDimensionAttr", "labels for the high dimensional layout dims">:$labels,
     ArrayRefParameter<"int64_t", "shapes for the high dimensional layout dims">:$shapes
   );
   let assemblyFormat = "`<``[` $labels `]``,` `[` $shapes `]``>`";
   let genVerifyDecl = 0;
   let extraClassDeclaration = [{
      std::optional<int64_t> getShape(const LayoutDimension &dim);
      bool contains(const LayoutDimension &dim);
   }];
}

def LayoutAttr : IREEVectorExt_Attr<"Layout",
      [ DeclareAttrInterfaceMethods<VectorLayoutInterface> ]> {
  let mnemonic = "layout";
  let summary = [{high-dimensional vector register layout for a given vector}];
  let description = [{
    This contains a complete specification of the layout for a given vector,
    whereas the attribute above only specifies the per dimension layout.
  }];
  let parameters = (ins
    ArrayRefParameter<"PerDimLayoutAttr", "layout for each dimension of the vector">:$layouts
  );
  let assemblyFormat = "`<`$layouts`>`";
  let genVerifyDecl = 0;
  let extraClassDeclaration = [{
    // Get the shape for a given layout dimension.
    std::optional<int64_t> getShape(const LayoutDimension &dim);
    std::optional<int64_t> getBatchDim(int64_t dim);
    // Get the lane dimension shape for a provided simd tensor dim.
    std::optional<int64_t> getLaneDim(int64_t dim);
    // Get the lane dimension for a provided simd tensor dim.
    std::optional<LayoutDimension> getLane(int64_t dim);

    // Returns the grid of lane ids. Assumes a valid layout.
    ::std::tuple<int64_t, int64_t, int64_t> getLaneGrid();
    PerDimLayoutAttr getDimLayout(int64_t dim) const;

    // Given the reduction dim, computes the shuffle offset
    // based on the shapes of the lane dimensions. The shuffle
    // offset is used during the thread global reduction
    // when emitting a gpu::ShuffleOp and follows
    // the semantics of the offset operand defined there,
    // which is that for lane k, the shuffle op returns the
    // value from lane k ^ offset.
    uint64_t getShuffleOffset(int64_t reductionDim);
  }];
}

def LayoutV2Attr : IREEVectorExt_Attr<"LayoutV2",
      [ DeclareAttrInterfaceMethods<VectorLayoutInterface> ]> {
  let mnemonic = "layoutv2";
  let summary = [{A layout representing how a GPU thread hierarchy is distributed over a shape}];
  let description = [{
    This layout explicitly defines how a shape is distributed over a tile
    heirarchy. We consider the following levels of hierarchy:

    1. Subgroups per Workgroup
    2. Threads per Subgroup
    3. Elements per Thread

    Each level of heirarchy represents a tile over the next level of
    heirarchy, except the last level, which represents elements. The tiling is
    done on each dimension of the original vector shape. For example, for a
    shape <16x16x16>, the tiles are 3-D. For a shape <32x32>, the tiles are
    2-D.

    We now describe each level of tiling:

    1. Subgroups per Workgroup

    This level of tiling is also known as "warp distribution". It represents
    how subgroups are distributed over a workgroup. The total number of
    subgroups are expected to be equal to the workgroup size of the GPU.

    The subgroups are placed contigiously with their shape and ordering
    determined by:
      - `subgroups_per_workgroup`: Sizes of this level of tiling
      - `subgroup_order`: Ordering of dimensions, from outermost to innermost

    For example, subgroups_per_workgroup=[4, 2], subgroup_order=[0, 1] will
    arrange the subgroups in the order:

    0 4
    1 5
    2 6
    3 7

    2. Threads per Subgroup:

    Threads in a subgroup are distributed in three levels. The first two
    levels, batches and duplicates are simply ways to represent replication,
    but at different levels. 

    The first level of replication, batches, are a way to represent instruction
    duplication. For example, a GPU intrinsic which can only take 4x4 shape at
    a time, uses batches to represent thread distribution over a 16x16 shape.

    Batches can be thought of as loops around the original layout:

    for b_0 in range(batch_0):
      for b_1 in range(batch_1):
        ...

    Batches are represented using two attributes:
      - batches_per_subgroup: Ranges of each loop
      - batch_order: Ordering of each loop, from outermost to innermost

    The second level of replication, duplicates, are a way to represent
    thread duplication required by a particular intrinsic. For example,
    some AMDGPU matrix multiplication variants require threads to be distributed
    like:

    0 1 2 3 4
    5 6 7 8 9
    --------- --> Duplicates of shape 2x5 distributed over a batch of size 4x5.
    0 1 2 3 4
    5 6 7 8 9

    Duplicates are represented using two attributes:
      - duplicates_per_batch: Number of duplicates in a batch
      - duplicate_order: Ordering of duplicates, from outermost to innermost

    Finally, threads are distributed in a single duplicate. The thread
    distribution is represented by:

      - threads_per_duplicate: Sizes of this level of tiling
      - thread_order: Ordering of dimensions, from outermost to innermost

    Examples of thread distribution over a 8x4 shape:

    {
      batches_per_subgroup = [2, 1]
      duplicates_per_batch = [2, 2]
      threads_per_duplicate = [2, 2]

      batch_order = [0, 1]
      duplicate_order = [0, 1]
      thread_order = [1, 0]
    }

    Distributed tile:

    {
      [0 2]|[0 2]      0,1,2,3 --> thread ids
      [1 3]|[1 3]      
      ------------     [x z]   --> a single duplicate tile
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

    3. Elements per Thread

    The final level of tiling, representing the minimum shape of vector that
    is treated as an atom.

    The elements are placed contigiously with their shape and ordering
    determined by:
      - `elements_per_thread`: Sizes of this level of tiling
      - `element_order`: Ordering of dimensions, from outermost to innermost
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "subgroups_per_workgroup">:$subgroupsPerWorkgroup,
    ArrayRefParameter<"int64_t", "subgroup_order">:$subgroupOrder,

    ArrayRefParameter<"int64_t", "batches_per_subgroup">:$batchesPerSubgroup,
    ArrayRefParameter<"int64_t", "batch_order">:$batchOrder,

    ArrayRefParameter<"int64_t", "duplicates_per_batch">:$duplicatesPerBatch,
    ArrayRefParameter<"int64_t", "duplicate_order">:$duplicateOrder,

    ArrayRefParameter<"int64_t", "threads_per_duplicate">:$threadsPerDuplicate,
    ArrayRefParameter<"int64_t", "thread_order">:$threadOrder,

    ArrayRefParameter<"int64_t", "elements_per_thread">:$elementsPerThread,
    ArrayRefParameter<"int64_t", "element_order">:$elementOrder
  );

  // TODO: add custom parser/printer and builder to elide default value array
  // refs.
  let assemblyFormat = [{
    `<` `subgroups_per_workgroup` `=` `[` $subgroupsPerWorkgroup `]` `,`
        `batches_per_subgroup`    `=` `[` $batchesPerSubgroup `]` `,`
        `duplicates_per_batch`    `=` `[` $duplicatesPerBatch `]` `,`
        `threads_per_duplicate`   `=` `[` $threadsPerDuplicate `]` `,`
        `elements_per_thread`     `=` `[` $elementsPerThread `]` `,`

        `subgroup_order`          `=` `[` $subgroupOrder `]` `,`
        `batch_order`             `=` `[` $batchOrder `]` `,`
        `duplicate_order`         `=` `[` $duplicateOrder `]` `,`
        `thread_order`            `=` `[` $threadOrder `]` `,`
        `element_order`           `=` `[` $elementOrder `]` 
    `>`
  }];

  let genVerifyDecl = 1;
}


#endif // IREE_DIALECT_VECTOREXT_ATTRS
