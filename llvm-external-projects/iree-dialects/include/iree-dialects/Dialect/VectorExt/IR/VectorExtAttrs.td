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
    std::optional<int64_t> getShape(const LayoutDimension &dim) const;
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

    // Determines whether the other layout has a lane
    // dimension that the current layout does not have OR whether
    // the shape of the two layouts for a common lane dimension
    // is not the same.
    bool hasLaneConflictWith(const LayoutAttr &other);
  }];
}

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
    count of tiles over the next level (rather than a list of tile sizes) and
    an ordering over the tiles:

    1. Subgroups per Workgroup

    This level of tiling is also known as "subgroup/warp distribution". It
    represents how subgroups are distributed in a workgroup. 

    The size of this tile is represented by `subgroups_per_workgroup`. How
    the subgroups are arranged in this tile is represented by the layout
    of this level of hierarchy, which is explained later.

    2. Threads per Subgroup:

    This level of tiling represents how threads are distributed in a subgroup.

    The size of this tile is represented by `threads_per_subgroup`. How the
    threads in this tile are arranged in this tile is represented by the layout
    of this level of hierarchy, which is explained later.

    3. Elements per Thread

    The final level of tiling, representing the minimum shape of vector that
    is treated as an atom.

    The size of this tile is represented by `elements_per_thread`.

    Now, we describe the "layout" of a level of hierarchy that we talked about
    earlier.

    Given a tile of size X, the layout L describes how the tiles of next level
    are arranged in X. L is a function: (id) -> (index in X). Ideally, we would
    like to be in a state where this layout is just another attribute, but for
    now, we describe the layout using:
      - `layout_shape`
      - `layout_order`

    These two attributes define a contigious mapping of a `id` to a shape.
    For example:

    layout_shape=[2, 2]              layout_shape=[3, 2]
    layout_order=[1, 0]              layout_order=[0, 1]

    L: 0 2                          L: 0 1
       1 3                             2 3
                                       4 5

    Now let's see how the layout is distributed over the tile shape. There
    are three cases to consider (For each dimension):
      1. layout_shape[i] = tile_shape[i]:
        This case is obvious and we distribute the layout completely over the
        shape.
      2. layout_shape[i] > tile_shape[i]
        In this case, the layout is larger than the tile shape dimension. We
        distribute multiple users over the same tile. For example,

        tile_shape=[1, 2]
        layout_shape=[2, 2]
        layout_order=[1, 0]

        L:
        0 2
        1 3

        Distribution over shape:
        [0 1, 2 3]

        So the same tile has multiple users using it.

      3. layout_shape[i] < tile_shape[i]
        In this case the layout is smaller than the tile shape dimension. We
        make each user hold multiple tiles by duplicating the layout. For example,

        tile_shape=[3, 2]
        layout_shape=[1, 2]
        layout_order=[0, 1]

        L:
        0 1

        Distribution over shape:
        [0 1]
        [0 1]
        [0 1]

    If a user is holding more than one tile, then the tiles are iterated in
    the order of iteration of the original vector.

    For example:

    [0 1 2 3 0 1 2 3]
    [0 1 2 3 0 1 2 3]

    Here, id 0 is holding 4 tiles. Let's say the order of dim iteration in the
    original vector is: [1, 0]. Then, id 0 will iterate over the tiles in that order:

    [0 1 2 3 0 1 2 3]
     ^       ^
     0       2
    [0 1 2 3 0 1 2 3]
     ^       ^
     1       3

    This shape/order layout style is inspired from Triton:
    https://github.com/openai/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td

    Finally, the `dim_order` keeps track of the order of iteration over the
    dimensions of the vector. Note that since we keep track of this,
    transpose over a vector is a no-op and the layout keeps track of the
    iteration order.

    Example of a layout for MFMA A, B and C matrix:

    A: vector<128x32> B: vector<32x64> C: vector<128x64>

    mfma_layout_A = {
      subgroups_per_workgroup = [2, 1]
      threads_per_subgroup = [64, 8]
      elements_per_thread = [1, 4]

      subgroup_layout_shape = [2, 2]
      subgroup_layout_order = [1, 0]

      thread_layout_shape = [32, 2]
      thread_layout_order = [1, 0]

      dim_order = [0, 1]
    }

    mfma_layout_B = {
      subgroups_per_workgroup = [1, 2]
      threads_per_subgroup = [8, 32]
      elements_per_thread = [4, 1]

      subgroup_layout_shape = [2, 2]
      subgroup_layout_order = [0, 1]

      thread_layout_shape = [2, 32]
      thread_layout_order = [0, 1]

      dim_order = [1, 0]
    }

    mfma_layout_C = {
      subgroups_per_workgroup = [2, 2]
      threads_per_subgroup = [16, 32]
      elements_per_thread = [4, 1]

      subgroup_layout_shape = [2, 2]
      subgroup_layout_order = [1, 0]

      thread_layout_shape = [2, 32]
      thread_layout_order = [0, 1]

      dim_order = [1, 0]
    }
  }];
      
  let parameters = (ins
    ArrayRefParameter<"int64_t", "subgroups_per_workgroup">:$subgroupsPerWorkgroup,
    ArrayRefParameter<"int64_t", "threads_per_subgroup">:$threadsPerOuter,
    ArrayRefParameter<"int64_t", "elements_per_thread">:$elementsPerThread,

    ArrayRefParameter<"int64_t", "subgroup_layout_shape">:$subgroupLayoutShape,
    ArrayRefParameter<"int64_t", "subgroup_layout_order">:$subgroupLayoutOrder,

    ArrayRefParameter<"int64_t", "thread_layout_shape">:$threadLayoutShape,
    ArrayRefParameter<"int64_t", "thread_layout_order">:$threadLayoutOrder,

    ArrayRefParameter<"int64_t", "dim_order">:$dimOrder,
  );

  let assemblyFormat = [{
    `<` `subgroups_per_workgroup` `=` `[` $subgroupsPerWorkgroup `]` `,`
        `threads_per_subgroup`    `=` `[` $threadsPerSubgroup `]` `,`
        `elements_per_thread`     `=` `[` $elementsPerThread `]` `,`

        `subgroup_layout_shape`   `=` `[` $subgroupLayoutShape `]` `,`
        `subgroup_layout_order`   `=` `[` $subgroupLayoutOrder `]` `,`

        `thread_layout_shape`     `=` `[` $threadLayoutShape `]` `,`
        `thread_layout_order`     `=` `[` $threadLayoutOrder `]` `,`

        `dim_order`               `=` `[` $dimOrder `]`
    `>`
  }];

  let extraClassDeclaration = [{
    // Returns the subgroup/lane ids delinearized from a single linearized
    // thread ID.
    ValueRange computeThreadIds(Value threadId, RewriterBase &rewriter) const;
  }];

  let genVerifyDecl = 1;
}


#endif // IREE_DIALECT_VECTOREXT_ATTRS
