// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_OPS
#define IREE_DIALECT_VECTOREXT_OPS

include "mlir/Interfaces/VectorInterfaces.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtBase.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtAttrs.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREEVectorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEVectorExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Layout ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_ToLayoutOp : IREEVectorExt_PureOp<"to_layout", [
  Pure,
  AllTypesMatch<["input", "output"]>
  ]> {
  let summary = [{Layout conversion operator.}];
  let description = [{
    The layout conversion operator takes a shaped value and a layout and
    transforms the value to have that layout.

    If the "shared_memory_conversion" attribute is set, then this layout
    change has to be materialized through shared memory.
  }];
  let arguments = (ins
    AnyShaped:$input,
    VectorLayoutInterface:$layout,
    DefaultValuedAttr<UnitAttr, "false">:$shared_memory_conversion,
    // TODO: Solve cmake IREEGPU and VectorExt cyclic dependency to
    // change mma_Kind type to be of MMAInterfaceAttr.
    OptionalAttr<AnyAttr>:$mma_kind
  );
  let results = (outs
    AnyShaped:$output
  );
  let builders = [
    OpBuilder<(ins "Value":$input,
                   "VectorLayoutInterface":$layout,
                   "Attribute":$mma_kind_attr,
                   CArg<"bool", "false">:$shared_memory_conversion), [{
      UnitAttr defaultSharedMemoryConversion;
      if (shared_memory_conversion) {
        defaultSharedMemoryConversion = UnitAttr::get(input.getContext());
      }
      build($_builder, $_state, input.getType(), input, layout, defaultSharedMemoryConversion, mma_kind_attr);
    }]>,
  OpBuilder<(ins "Value":$input,
                "VectorLayoutInterface":$layout), [{
      UnitAttr defaultSharedMemoryConversion;
      Attribute emptyIntrinsic;
      build($_builder, $_state, input.getType(), input, layout, defaultSharedMemoryConversion, emptyIntrinsic);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool hasTensorSemantics() {
      return isa<RankedTensorType>(getOutput().getType());
    }
  }];
  let assemblyFormat = "$input `to` `layout` `(` $layout `)` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def IREEVectorExt_ToSIMDOp : IREEVectorExt_PureOp<"to_simd",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = [{SIMT to SIMD conversion operation.}];
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input
  );
  let results = (outs
    AnyVectorOfAnyRank:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_ToSIMTOp : IREEVectorExt_PureOp<"to_simt",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = [{SIMD to SIMT conversion operation.}];
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input
  );
  let results = (outs
    AnyVectorOfAnyRank:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_TransferGatherOp : IREEVectorExt_PureOp<"transfer_gather", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    AttrSizedOperandSegments,
    TypesMatchWith<"padding type must match source element type",
                   "base", "padding", "getElementTypeOrSelf($_self)">
  ]> {
  let arguments = (ins AnyShaped:$base,
                       Variadic<Index>:$offsets,
                       Variadic<AnyTypeOf<[Index, VectorOfAnyRankOf<[Index]>]>>:$index_vecs,
                       AffineMapArrayAttr:$indexing_maps,
                       AnyType:$padding,
                       Optional<VectorOfAnyRankOf<[I1]>>:$mask);
  let results = (outs AnyVectorOfAnyRank:$vector);

  let summary = [{Gathers a supervector from a shaped source into an SSA vector value.}];

  let description = [{
    The `transfer_gather` operation reads elements from a shaped source (memref
    or tensor) into a vector, where each source dimension can be independently
    contiguous, gathered, or broadcast.

    Semantically, for each position in the result vector:

    ```
    result[d0, d1, ...] = base[offsets[0] + f0(d, s), offsets[1] + f1(d, s), ...]
    ```

    where each `fi` is the i-th result of the source indexing map evaluated at
    the vector position `d = (d0, d1, ...)` and gathered index values
    `s = (s0, s1, ...)`.

    The `indexing_maps` attribute describes all indexing. Every map has
    `numDims = result vector rank` and `numSymbols = number of index vecs`:

    - Map 0 (source map): `(vector_dims)[symbols] -> (source_dims)`.
      A dim expr means the source dimension is contiguous (iterated in
      lockstep with the vector dimension). A symbol expr means the source
      dimension is gathered (looked up via the corresponding index vector).
      A constant 0 means the source dimension is broadcast (always reads
      at the base offset).
    - Maps 1..N (index vec maps): `(vector_dims)[symbols] -> (index_vec_dims)`.
      Describes how each index vector is indexed from the vector iteration
      space. Only dim exprs are allowed.
    - Optional last map (mask map): present only when a mask operand is
      provided. Only dim exprs are allowed.

    Example — embedding lookup: reading rows from a 3D source where the row
    index is gathered from an index vector, while the column is contiguous:

    ```
    // result[i, j] = base[0, indices[i], j]
    %result = iree_vector_ext.transfer_gather %base[%c0, %c0, %c0]
      [%indices : vector<16xindex>], %pad {
        indexing_maps = [
          affine_map<(d0, d1)[s0] -> (0, s0, d1)>,
          affine_map<(d0, d1)[s0] -> (d0)>
        ]
      } : memref<4096x512x8xf16>, vector<16x8xf16>
    ```
  }];

  let extraClassDeclaration = [{
    SmallVector<AffineMap> getIndexingMapsArray() {
      return llvm::to_vector(getIndexingMaps().getAsValueRange<AffineMapAttr>());
    }

    AffineMap getSourceIndexingMap() {
      return getIndexingMapsArray().front();
    }

    SmallVector<AffineMap> getIndexVecIndexingMaps() {
      auto maps = getIndexingMapsArray();
      return SmallVector<AffineMap>(
          maps.begin() + 1, maps.begin() + 1 + getIndexVecs().size());
    }

    std::optional<AffineMap> getMaskIndexingMap() {
      if (!getMask()) return std::nullopt;
      return getIndexingMapsArray().back();
    }

    /// Invert the source indexing map to get a permutation map suitable for
    /// vector.transfer_read. The source map is
    /// (vector_dims)[symbols] -> (source_dims). This returns
    /// (source_dims) -> (vector_dims), where non-dim results (gathered
    /// symbols, broadcast constants) become constant 0.
    AffineMap getPermutationMap() {
      AffineMap sourceMap = getSourceIndexingMap();
      MLIRContext *ctx = getContext();
      SmallVector<AffineExpr> exprs(sourceMap.getNumDims(),
                                    getAffineConstantExpr(0, ctx));
      for (auto [i, expr] : llvm::enumerate(sourceMap.getResults())) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          exprs[dimExpr.getPosition()] = getAffineDimExpr(i, ctx);
        }
      }
      return AffineMap::get(sourceMap.getNumResults(), /*symbolCount=*/0,
                            exprs, ctx);
    }
  }];

  let assemblyFormat = "$base `[` $offsets `]` (`[` $index_vecs^ `:` type($index_vecs) `]`)? `,` $padding (`,` $mask^)? attr-dict `:` type($base) `,` type($vector) (`,` type($mask)^)?";

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Terminator ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_YieldOp : IREEVectorExt_PureOp<"yield", [
    Pure,
    ReturnLike,
    Terminator,
    ParentOneOf<["ArgCompareOp"]>
  ]> {
  let summary = "Yield operation for VectorExt operations with regions";
  let description = [{
    Yields values from regions in VectorExt operations.
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LinalgExt-style ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_ArgCompareOp : IREEVectorExt_PureOp<"arg_compare", [
  Pure,
  AttrSizedOperandSegments,
  SingleBlockImplicitTerminator<"IREE::VectorExt::YieldOp">,
  AllElementTypesMatch<["input_value", "init_value"]>,
  AllTypesMatch<["init_value", "result_value"]>,
  AllTypesMatch<["init_index", "result_index"]>
]> {
  let summary = "Vectorized arg-reduction using a user-defined comparator.";
  let description = [{
    The `iree_vector_ext.arg_compare` op is the vectorized form of
    `iree_linalg_ext.arg_compare`. It performs a reduction over a given
    dimension of a vector, returning both the selected value and its
    corresponding index.

    **Comparator region semantics:**

    The comparator region defines the predication logic that determines the
    selection rule for the reduction (e.g., "greater than" for argmax or
    "less than" for argmin). The region takes two scalar values of the input
    element type as arguments and returns a single `i1` result via
    `iree_vector_ext.yield`.

    The region is invoked during the reduction to determine which element to
    select: when the comparison yields `true`, the first argument is selected;
    otherwise, the second argument is selected.

    The region must contain only pure operations (operations with the `Pure`
    trait). This ensures the comparator can be safely executed in any order
    during the reduction.

    Example (implicit-index mode - argmax over dim 1):
    ```mlir
    %input_vec = vector<4x128xf32>
    %out_val_vec = vector<4xf32>
    %out_idx_vec = vector<4xi32>
    %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%input_vec : vector<4x128xf32>)
      inits(%out_val_vec, %out_idx_vec : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
    } -> vector<4xf32>, vector<4xi32>
    ```

    Example (explicit-index mode):
    ```mlir
    %partial_vals = vector<4x32xf32>
    %partial_idxs = vector<4x32xi32>
    %out_val = vector<4xf32>
    %out_idx = vector<4xi32>
    %result:2 = iree_vector_ext.arg_compare
      dimension(1)
      ins(%partial_vals, %partial_idxs : vector<4x32xf32>, vector<4x32xi32>)
      inits(%out_val, %out_idx : vector<4xf32>, vector<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_vector_ext.yield %cmp : i1
    } -> vector<4xf32>, vector<4xi32>
    ```

    The `index_base` is an optional offset value that, when specified, is added
    to the computed indices in the result. This is useful when reducing over a
    sliced subregion where the indices need to be adjusted to reflect their
    position in the original vector. The `index_base` can only be used in
    implicit-index mode (single input).

    The `inits` operands provide the initial accumulator values for the
    reduction (initial max/min value and index).
  }];

  let arguments = (ins
    AnyVectorOfAnyRank:$input_value,
    Optional<AnyVectorOfAnyRank>:$input_index,
    AnyVectorOfAnyRank:$init_value,
    AnyVectorOfAnyRank:$init_index,
    Optional<Index>:$index_base,
    I64Attr:$dimension
  );

  let results = (outs
    AnyVectorOfAnyRank:$result_value,
    AnyVectorOfAnyRank:$result_index
  );

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $input_value (`,` $input_index^)? `:` type($input_value) (`,` type($input_index)^)? `)`
    `inits` `(` $init_value `,` $init_index `:` type($init_value) `,` type($init_index) `)`
    (`index_base` `(` $index_base^ `:` type($index_base) `)`)?
    $region `->` type($result_value) `,` type($result_index)
  }];

  let extraClassDeclaration = [{
    bool hasExplicitIndexInput() {
      return getInputIndex() != nullptr;
    }

    VectorType getInputValueType() {
      return getInputValue().getType();
    }

    VectorType getInitValueType() {
      return getInitValue().getType();
    }

    VectorType getInitIndexType() {
      return getInitIndex().getType();
    }

    VectorType getInputIndexType() {
      if (Value idx = getInputIndex()) {
        return cast<VectorType>(idx.getType());
      }
      return nullptr;
    }

    Type getInputIndexElementType() {
      if (VectorType idxType = getInputIndexType()) {
        return idxType.getElementType();
      }
      return nullptr;
    }

    Type getInputValueElementType() {
      return getInputValueType().getElementType();
    }

    Type getInitValueElementType() {
      return getInitValueType().getElementType();
    }

    Type getInitIndexElementType() {
      return getInitIndexType().getElementType();
    }

    int64_t getInputRank() {
      return getInputValueType().getRank();
    }
  }];

  let hasVerifier = 1;
}

#endif  // IREE_DIALECT_VECTOREXT_OPS
