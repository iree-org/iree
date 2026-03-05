// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_MAP_IREEMAP_ATTRS
#define IREE_CODEGEN_DIALECT_MAP_IREEMAP_ATTRS

include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapBase.td"

//===----------------------------------------------------------------------===//
// PackMapAttr
//===----------------------------------------------------------------------===//

def IREEMap_PackMapAttr : IREEMap_Attr<"PackMap"> {
  let mnemonic = "pack_map";

  let summary = [{A layout map: a (Shape, Stride) pair mapping coordinates to indices.}];

  let description = [{
    A layout map is a function from coordinates to 1-D indices, defined by a
    `(Shape, Stride)` pair of hierarchical IntTuples.

    #### IntTuple

    An IntTuple is a recursive type: either a single integer or a tuple of
    IntTuples. For example: `4`, `(2, 3)`, `((2, 4), 8)`.

    All values are static i32 constants.

    #### Coordinate-to-index mapping

    A layout maps a coordinate to a 1-D index by pairing each shape leaf with
    its corresponding stride leaf and computing a weighted sum. The coordinate
    is decomposed into per-leaf natural coordinates (like mixed-radix digits),
    then each is multiplied by its stride and summed.

    For a flat layout `(S0, S1) : (D0, D1)`, coordinate `(c0, c1)` maps to:

    ```
    index = c0 * D0 + c1 * D1
    ```

    **Example: row-major `(4, 8) : (8, 1)`**

    ```
    coord (0, 0) → 0*8 + 0*1 =  0
    coord (0, 5) → 0*8 + 5*1 =  5
    coord (2, 3) → 2*8 + 3*1 = 19
    coord (3, 7) → 3*8 + 7*1 = 31

         col 0  col 1  col 2  ...  col 7
    row 0:  0      1      2   ...    7
    row 1:  8      9     10   ...   15
    row 2: 16     17     18   ...   23
    row 3: 24     25     26   ...   31
    ```

    **Example: column-major `(4, 8) : (1, 4)`**

    ```
    coord (0, 0) → 0*1 + 0*4 =  0
    coord (2, 0) → 2*1 + 0*4 =  2
    coord (0, 1) → 0*1 + 1*4 =  4
    coord (3, 7) → 3*1 + 7*4 = 31
    ```

    **Example: hierarchical `((2, 4), 8) : ((16, 1), 4)`**

    Hierarchical shapes nest sub-modes within a mode. Coordinate `(c0, c1)`
    first decomposes `c0` into sub-coordinates for the inner tuple `(2, 4)`,
    then computes the weighted sum across all leaves:

    ```
    coord (5, 3) → c0=5 decomposes as (1, 1) in shape (2, 4)
                  → 1*16 + 1*1 + 3*4 = 29
    ```

    #### Ordering convention

    This implementation uses **lexicographic (row-major)** ordering: the
    rightmost mode is the innermost (fastest-varying) dimension. The default
    stride for shape `(M, N, K)` is the suffix product `(N*K, K, 1)`.

    #### Congruence

    Shape and stride must be **congruent**: they have identical hierarchical
    tree structure.

    #### Assembly format

    ```mlir
    #iree_map.pack_map<(8) : (1)>                        // rank-1 contiguous
    #iree_map.pack_map<(4, 8) : (8, 1)>                  // rank-2 row-major
    #iree_map.pack_map<(4, 8) : (1, 4)>                  // rank-2 column-major
    #iree_map.pack_map<((2, 4), 8) : ((16, 1), 4)>       // hierarchical
    ```
  }];

  let parameters = (ins
    "::mlir::Attribute":$shape,
    "::mlir::Attribute":$stride
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // --- Properties ---
    int32_t getRank();
    int32_t getDepth();
    /// Product of all leaf shape values. May overflow for large layouts.
    int32_t getSize();
    int32_t getCosize();

    // --- Mode access ---
    ::mlir::Attribute getShapeMode(int32_t i);
    ::mlir::Attribute getStrideMode(int32_t i);

    // --- Evaluation ---
    int32_t evaluate(::llvm::ArrayRef<int32_t> coord);

    // --- Simplification ---

    /// Merge adjacent leaves that form a contiguous range. Does not change the
    /// layout function, only simplifies the representation.
    PackMapAttr coalesce();
    /// Flatten all hierarchy into a single-level tuple of leaves.
    PackMapAttr flatten();

    // --- Algebra ---

    /// Functional composition: result(c) = this(rhs(c)).
    PackMapAttr compose(PackMapAttr rhs);
    /// Find a layout B such that A and B together tile [0, cotarget).
    PackMapAttr complement(int32_t cotarget);
    /// Factor layout into (inner, outer) using a tiler: inner covers the tile,
    /// outer covers the rest.
    PackMapAttr logicalDivide(PackMapAttr tiler);
    /// Replicate this layout using a tiler pattern.
    PackMapAttr logicalProduct(PackMapAttr tiler);
    /// Remove trivial modes (size-1 or stride-0 leaves), then coalesce.
    PackMapAttr filter();
    /// Find R such that A(R(x)) = x for all x in the injective range.
    PackMapAttr rightInverse();
    /// Find L such that L(A(x)) = x for all x in [0, size).
    PackMapAttr leftInverse();
    /// Like logicalDivide, but flattens the outer modes to top level.
    PackMapAttr tiledDivide(PackMapAttr tiler);
    /// Like logicalProduct, but flattens the replicated modes to top level.
    PackMapAttr tiledProduct(PackMapAttr tiler);

    // --- Mode operations ---
    PackMapAttr permute(::llvm::ArrayRef<int64_t> perm);
    PackMapAttr project(::llvm::ArrayRef<bool> droppedDims);

    // --- Factory ---
    /// Create the row-major (lexicographic) identity layout for a flat shape.
    /// For shape (M, N, K) produces (M, N, K) : (N*K, K, 1).
    static PackMapAttr makeIdentity(::mlir::MLIRContext *ctx,
                                    ::llvm::ArrayRef<int64_t> shape);
  }];
}

#endif // IREE_CODEGEN_DIALECT_MAP_IREEMAP_ATTRS
