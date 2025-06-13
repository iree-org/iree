// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREECODEGENOPS
#define IREE_CODEGEN_DIALECT_IREECODEGENOPS

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

def IREECodegen_QueryTileSizesOp :
    Op<IREECodegen_Dialect, "query_tile_sizes", [Pure]> {
  let summary = [{Yields tile sizes for the specified tensor type.}];

  let description = [{
    For targets where tile sizes can't be resolved at compile time, this
    operation allows querying the sizes at runtime. Today this only applies
    to VMVX.
  }];

  let arguments = (ins TensorTypeAttr:$tensor_type);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = [{
    attr-dict $tensor_type `->` type($results)
  }];
}

//===----------------------------------------------------------------------===//
// ExtractStridedMetadataOp
//===----------------------------------------------------------------------===//

def IREECodegen_ExtractStridedMetadataOp : Op<IREECodegen_Dialect, "extract_strided_metadata", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    SameVariadicResultSize,
    ViewLikeOpInterface,
    InferTypeOpAdaptor]> {
  let summary = [{Extracts a buffer base with offset and strides.}];
  let description = [{
    This op is implemented similarly to the upstream MemRef::ExtractStridedMetadataOp
    with the following differences.

    1. It does not fold away static offset/stride information.
    Hence unlike the upstream Op the link between the memref and consumers of the
    metadata is not broken when later passes change this information. A common
    example in IREE of this is buffer binding optimizations.

    2. Helper functions getConstifiedMixed{Offset|Strides|Sizes} are not implemented
    as the expectation is you should lower to the upstream op before using those
    functions if you need them.

    Copy of MemRef::ExtractStridedMetadataOp description for reference below.
    Extracts a base buffer, offset and strides. This op allows additional layers
    of transformations and foldings to be added as lowering progresses from
    higher-level dialect to lower-level dialects such as the LLVM dialect.

    The op requires a strided memref source operand. If the source operand is not
    a strided memref, then verification fails.

    This operation is also useful for completeness to the existing memref.dim op.
    While accessing strides, offsets and the base pointer independently is not
    available, this is useful for composing with its natural complement op:
    `memref.reinterpret_cast`.

    Intended Use Cases:

    The main use case is to expose the logic for manipulate memref metadata at a
    higher level than the LLVM dialect.
    This makes lowering more progressive and brings the following benefits:
      - not all users of MLIR want to lower to LLVM and the information to e.g.
        lower to library calls---like libxsmm---or to SPIR-V was not available.
      - foldings and canonicalizations can happen at a higher level in MLIR:
        before this op existed, lowering to LLVM would create large amounts of
        LLVMIR. Even when LLVM does a good job at folding the low-level IR from
        a performance perspective, it is unnecessarily opaque and inefficient to
        send unkempt IR to LLVM.
  }];

  let arguments = (ins
    AnyStridedMemRef:$source
  );
  let results = (outs
    AnyStridedMemRefOfRank<0>:$base_buffer,
    Index:$offset,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $source `:` type($source) `->` type(results) attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

//===----------------------------------------------------------------------===//
// SwizzleHintOp
//===----------------------------------------------------------------------===//

def IREECodegen_SwizzleHintOp : Op<IREECodegen_Dialect, "swizzle_hint", [
    SameOperandsAndResultType, Pure]> {
  let summary = [{Hint to swizzle accesses according to an access pattern.}];
  let description = [{
    Optimization hint to swizzle all accesses to the memref this takes a view
    of. This only affects reads/writes immediately consuming this operation and
    is best effort. If the desired swizzling is not apparently possible, this
    op will no-op. As a result, it should not be relied on for correctness.

    Any subviews on this operation will cause swizzling to fail. The expectation
    is for all view like operations to fold into the accessing ops
    (loads/stores) before this op takes effect.

    Note that this only rewrites direct users. If there are any aliased loads
    or stores of the data from/to the |src| memref of a hintOp, those accesses
    will not be swizzled. This allows reusing an allocation with different
    swizzled access patterns as long as there is no data dependency between
    memory with different layouts. For example:

    ```
    %0 = alloc()
    %1 = iree_codegen.swizzle_hint %0, #layout_0
    %2 = iree_codegen.swizzle_hint %0, #layout_1
    {
       vector.store %1
       vector.load %1
         ^
         |
        unrelated
         |
         v
       vector.store %2
       vector.load %2
    }
    ```

    If there is a data dependency between the accesses of %1 and %2, for example
    a value stored to %1 is loaded from %2, this is undefined behavior. Aliasing
    is otherwise perfectly legal.
  }];

  let arguments = (ins MemRefRankOf<[AnyType], [1]>:$operand,
                       IREECodegen_AnySwizzleAttr:$swizzle);
  let results = (outs MemRefRankOf<[AnyType], [1]>:$result);

  let assemblyFormat = [{
    $operand `[` $swizzle attr-dict `]` `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// NullPointerOp
//===----------------------------------------------------------------------===//

def IREECodegen_NullPointerOp :
     Op<IREECodegen_Dialect, "null_pointer", [Pure]> {
  let summary = [{Returns a null_pointer value.}];
  let description = [{
    This is meant to be used only as arguments to microkernels.
  }];
  let results = (outs NullPointer:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// LoadFrom/StoreToBuffer Ops
//===----------------------------------------------------------------------===//

def IREECodegen_LoadFromBufferOp : Op<IREECodegen_Dialect, "load_from_buffer",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]> {
  let summary = [{Loads a tensor from a memref.}];
  let description = [{
    Loads a tensor from a memref with a compatible shape and the same element
    type.
  }];

  let arguments = (ins
    AnyStridedMemRef:$buffer
  );
  let results = (outs
    AnyRankedTensor:$tensor
  );

  let assemblyFormat = [{
    $buffer attr-dict `:` type($buffer) `->` type($tensor)
  }];

  let hasVerifier = 1;
}

def IREECodegen_StoreToBufferOp : Op<IREECodegen_Dialect, "store_to_buffer",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{Stores a tensor into a memref.}];
  let description = [{
    Stores a tensor into a memref with a compatible shape and the same element
    type.
  }];

  let arguments = (ins
    AnyRankedTensor:$tensor,
    AnyStridedMemRef:$buffer
  );
  let results = (outs);

  let assemblyFormat = [{
    $tensor `,` $buffer
    attr-dict `:` type($tensor) `into` type($buffer)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InnerTiledOp
//===----------------------------------------------------------------------===//

def IREECodegen_InnerTiledOp : Op<IREECodegen_Dialect, "inner_tiled", [
    Pure,
    AttrSizedOperandSegments,
    InferTypeOpAdaptor,
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
    DeclareOpInterfaceMethods<TilingInterface,
       ["getIterationDomain",
        "getLoopIteratorTypes",
        "getTiledImplementation",
        "getResultTilePosition"]>
    ]> {
  let summary = [{Models an inner-tiled operation that may perform contractions.}];
  let description = [{
    Represents an operation that updates "inner tiles" (vector slices) of its accumulator
    outputs using inner tiles of its input operands. The way the inner tiles are
    used is determined by the semantics of the `kind` attribute. These inner tiles
    are the trailing dimensions of the tensor/vector operands that are not present
    in the indexing maps.

    In the case of a matrix-multiply-accumulate (MMA) inner tiled operation,
    the semantics logically match `vector.contraction`. However, instead of a
    combiner type, it has a intrinsic description that specifies how the inner tiles
    are combined.

    Similar to `vector.contract`, an iterator type attribute list must be
    specified, where each element of the list represents an iterator over one
    of the outer dimensions. Iteration of inner dimensions is defined solely by
    the intrinsic and may be opaque.

    An indexing map attribute list must be specified with an entry for input and
    output arguments. An indexing map attribute specifies a mapping from each
    outer loop iterator in the iterator type list, to each dimension of each
    operand.

    The combiner type is defined by the intrinsic.

    Example:

    ```mlir
    #contraction_accesses = [
     affine_map<(i, j, k) -> (i, k)>,
     affine_map<(i, j, k) -> (k, j)>,
     affine_map<(i, j, k) -> (i, j)>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    }
    %3 = iree_codegen.inner_tiled ins(%0, %1) outs(%2) #contraction_trait
      : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>

    // Takes tensors as well, however the inner dimensions must always be
    // static.
    %7 = iree_codegen.inner_tiled ins(%4, %5) outs(%6) #contraction_trait
      : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
    ```

    The example above can be logically lowered directly to loops like this
    (ignoring type conversions from tensor to vector needed for the mfma).
    ```
    %outer_m = tensor.dim %6, %c0 : index
    %outer_n = tensor.dim %6, %c1 : index
    %outer_k = tensor.dim %4, %c1 : index
    %7 = scf.for %i = %c0 to %outer_m iter_args(%arg0 = %6) {
      %8 = scf.for %j = %c0 to %outer_n iter_args(%arg1 = %arg0) {
        %9 = scf.for %k = %c0 to %outer_k iter_args(%arg2 = %arg1) {
          %lhs = tensor.extract_slice %4 [%i, %k, 0] [1, 1, 4] [1, 1, 1] : tensor<4xf16>
          %rhs = tensor.extract_slice %5 [%k, %j, 0] [1, 1, 4] [1, 1, 1] : tensor<4xf16>
          %acc = tensor.extract_slice %arg2 [%i, %j, 0] [1, 1, 4] [1, 1, 1] : tensor<4xf32>
          %res = amdgpu.mfma %lhs, %rhs, %acc : tensor<4xf32>
          %ret = tensor.insert_slice %acc into %arg2 [%i, %j, 0] [1, 1, 4] [1, 1, 1] : tensor<?x?x4xf32>
          scf.yield %ret : tensor<?x?x4xf32>
        }
        scf.yield %9 : tensor<?x?x4xf32>
      }
      scf.yield %8 : tensor<?x?x4xf32>
    }
    ```

    Or alternatively unrolled to a single intrinsic when operation on vectors.
    ```mlir
    #contraction_accesses = [
     affine_map<() -> ()>,
     affine_map<() -> ()>,
     affine_map<() -> ()>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = [],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    }
    %3 = iree_codegen.inner_tiled ins(%0, %1) outs(%2) #contraction_trait
      : vector<4xf16>, vector<4xf16> into vector<4xf32>
    ```

    This operation can represent an intrinsic both in undistributed
    (workgroup/subgroup/warp) and distributed (thread) level. The descriptor
    attribute specifies the inner tile sizes for both the undistributed form
    (where the operands represent the data to be processed by an entire group of
    parallel workers) and the distributed form (where the operands represent the
    data processed by a single workitem/thread).

    In some cases, variations on the inner tiled operations can be expressed
    with the permutations attribute. This attribute represents the permutation
    from that intrinsic's "canonical" layout (in the case of matrix multiplication,
    this is row-major storage of the inner tile) to the format of the inner tile
    in the arguments, with a permutation specified for each argument.

    Since the canonical dimensionality of the inner dimensions are somewhat
    intrinsic specific, verification of this op requires only that element
    counts of the inner dimensions match the intrinsic.

    For example, an MMT product of inner dimensions with warp semantics can be
    represented with the following. Permutations are only allowed for ops with
    undistributed semantics and must be resolved before distribution.

    ```mlir
    #contraction_accesses = [
     affine_map<(i, j, k) -> (i, k)>,
     affine_map<(i, j, k) -> (k, j)>,
     affine_map<(i, j, k) -> (i, j)>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      permutations = [[0, 1], [1, 0], [0, 1]]
    }
    %7 = iree_codegen.inner_tiled ins(%4, %5) outs(%6) #contraction_trait
      : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>
    ```

    #### Motivation, Design Choices, and Pitfalls

    This operation grew out of a general representation for matrix multiplication
    intrinsics on GPUs, where the inner tiles would be the tiles of the A, B,
    and C matrices that were computed by an entire GPU workgroup or subgroup. It
    is now used for generalizations of such multiplications. Currently,
    the only usage is for scaled matrix-multiply-accumulate, where block scales
    must be passed in as additional inputs, but it's possible more uses
    will be possible in the future.

    The idea behind this operation is to decouple the layout setting/tiling
    required to target certain intrinsics from the lowering to them. Because
    typically tiling of this sort happens on tensor operands, however the target
    intrinsics operate on vectors, we use this operation to bridge the gap. The
    choice for a shared operation is intended to ease the lowering process and
    allow for different transformations at different stages of the pipeline
    without needing to essentially clone this op.

    The choice to let the inner dimensions required to compute the intrinsic be
    implicit based on the indexing maps was made to make this operation easier
    to generate and to skip the need for type conversion ops. However this comes
    at the expense of ease of verification for the operation. It is also
    implicitly linked to a lane-level parent `scf.forall` operation.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensorOrVector>:$inputs,
    Variadic<AnyRankedTensorOrVector>:$outputs,
    TypedArrayAttrBase<AffineMapAttr, "indxing affine maps">:$indexing_maps,
    IteratorTypeArrayAttr:$iterator_types,
    IREECodegen_AnyInnerTileDescAttr:$kind,
    OptionalAttr<TypedArrayAttrBase<DenseI64ArrayAttr, "permutations">>:$permutations
  );
  let results = (outs
    Variadic<AnyRankedTensorOrVector>:$results
  );

  let assemblyFormat = [{
    `ins` `(` $inputs `)` `outs` `(` $outputs `)` attr-dict
    `:` type($inputs) `into` type($outputs)
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
      "ArrayAttr":$indexingMaps, "ArrayAttr":$iteratorTypes,
      "InnerTileDescAttrInterface":$intrinsic,
      CArg<"std::optional<ArrayAttr>", "std::nullopt">:$permutations)>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
      "ArrayRef<ArrayRef<AffineExpr>>":$indexingExprs,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "InnerTileDescAttrInterface":$intrinsic,
      CArg<"std::optional<SmallVector<SmallVector<int64_t>>>", "std::nullopt">:$permutationss)>,
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
      "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "InnerTileDescAttrInterface":$intrinsic,
      CArg<"std::optional<SmallVector<SmallVector<int64_t>>>", "std::nullopt">:$permutationss)>
  ];
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::ShapedType> getOperandShapedTypes() {
      return ::llvm::map_to_vector(getOperandTypes(), [](auto t) {
        return ::llvm::cast<::mlir::ShapedType>(t);
      });
    }

    int64_t getNumInputs() {
      return getInputs().size();
    }

    int64_t getNumOutputs() {
      return getOutputs().size();
    }

    bool hasTensorSemantics() {
      return isa<RankedTensorType>(getOutputs().front().getType());
    }

    bool hasThreadSemantics();

    llvm::SmallVector<::mlir::AffineMap, 4> getIndexingMapsArray() {
      return llvm::to_vector<4>(getIndexingMaps().getAsValueRange<::mlir::AffineMapAttr>());
    }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    void getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds);

    SmallVector<utils::IteratorType> getIteratorTypesArray() {
      auto range =
          getIteratorTypes()
              .template getAsValueRange<::mlir::linalg::IteratorTypeAttr, utils::IteratorType>();
      return {range.begin(), range.end()};
    }

    ArrayRef<int64_t> getOperandInnerShape(uint32_t index) {
      auto opType = ::llvm::cast<::mlir::ShapedType>(getOperandTypes()[index]);
      int64_t innerDimRank = opType.getRank() - getIndexingMapsArray()[index].getNumResults();
      return opType.getShape().take_back(innerDimRank);
    }

    int64_t getOperandOuterRank(int64_t index) {
      return getIndexingMapsArray()[index].getNumResults();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      if (hasTensorSemantics()) {
        return getOutputsMutable();
      }
      // There are no destinations with vector semantics.
      return MutableOperandRange(*this, 0, 0);
    }
  }];

  let hasVerifier = 1;
}

#endif // IREE_CODEGEN_DIALECT_IREECODEGENOPS
