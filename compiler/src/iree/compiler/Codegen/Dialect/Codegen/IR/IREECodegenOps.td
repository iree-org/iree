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
include "iree/compiler/Utils/CommonTypeConstraints.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Properties.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ValueBoundsOpInterface.td"
include "mlir/Dialect/SMT/IR/SMTTypes.td"

def TensorTypeAttr : TypeAttrBase<"::mlir::TensorType", "Tensor type attribute">;

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
// FusionBarrierOp
//===----------------------------------------------------------------------===//

def IREECodegen_FusionBarrierOp :
    Op<IREECodegen_Dialect, "fusion_barrier", [
      SameOperandsAndResultType, Pure]> {
  let summary = [{Prevents fusion through a tensor value}];

  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    attr-dict $source `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// SwizzleHintOp
//===----------------------------------------------------------------------===//

def IREECodegen_SwizzleHintOp : Op<IREECodegen_Dialect, "swizzle_hint", [
    SameOperandsAndResultType, Pure]> {
  let summary = [{Hint to swizzle accesses according to an access pattern.}];
  let description = [{
    Optimization hint to swizzle all accesses to the memref or tensor that this takes a
    view of. This only affects reads/writes that immediately consume this operation
    and is best effort. If the desired swizzling is not apparently possible, this
    op will no-op. As a result, it should not be relied on for correctness.

    Any subviews on this operation will cause the swizzle application to fail. The
    expectation is for all view like operations to fold into the accessing ops
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

  let arguments = (ins AnyRankedTensorOrMemRef:$operand,
                       IREECodegen_AnySwizzleAttr:$swizzle);
  let results = (outs AnyRankedTensorOrMemRef:$result);

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
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
       ["reifyResultShapes"]>]> {
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
    DeclareOpInterfaceMethods<ValueBoundsOpInterface,
       ["populateBoundsForShapedValueDim"]>,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
    DeclareOpInterfaceMethods<TilingInterface,
       ["getIterationDomain",
        "getLoopIteratorTypes",
        "getTiledImplementation",
        "getResultTilePosition"]>
    ]> {
  let summary = [{Generic operation on tiled operands based on an intrinsic.}];
  let description = [{
    Each operand's shape is viewed as `outer_dims x inner_tile`:
    - The outer dims are those described by the operand's indexing map; they
      are iterated by an outer loop nest specified by `iterator_types` and
      `indexing_maps`, with the same conventions as `vector.contract`.
    - The remaining trailing dims form the operand's inner tile. At each
      outer iteration point, the output inner tiles are updated from the
      input inner tiles by the intrinsic named by `kind`.

    Operands may be tensors, memrefs or vectors. Inner-tile dims are generally
    static, the only exception being scalable vectors dimensions which
    correspond to dynamic tensor dimensions.

    ### Attributes

    - `kind` (`InnerTileDescAttrInterface`): names the intrinsic. It fixes
      the number of inputs/outputs and, together with `semantics`, the
      canonical inner-tile vector type of each operand.
    - `semantics` (`InnerTiledSemanticsAttrInterface`): controls how strictly
      operand inner tiles are checked against `kind`. Different `semantics`
      attributes from different dialects may adopt different conventions, but
      there are generally two main modes:
      * `opaque = true`: only the element type and total element count of
        each inner tile must match the canonical vector type. Shape and
        rank are otherwise free. Scalable vectors and dynamic tensor dims
        are not allowed in this mode.
      * Default / `opaque = false`: after dropping non-scalable unit dims, the
        operand inner-tile shape must match the (permuted) canonical shape
        one-to-one. Scalable vector dimensions are supported and must correspond
        to dynamic tensor dimensions. Every other tensor dimension must be static.
    - `indexing_maps`: one projected-permutation `AffineMap` per operand
      (inputs first, then outputs), all sharing the same number of dims
      (equal to `iterator_types.size()`) and no symbols.
    - `iterator_types`: one of `parallel` / `reduction` for each outer
      iterator dim.
    - `permutations` (optional): If present, there is one entry per operand,
      each a permutation vector
      `p` of length equal to that operand's inner rank, giving the
      relationship between the canonical inner-tile shape `C` (from
      `kind`/`semantics`) and the operand's actual inner-tile shape `S`:
      `S[i] == C[p[i]]`.

    ### Examples

    Subgroup-level (undistributed) MMA on tensors using the AMD MFMA
    intrinsic `MFMA_F32_32x32x8_F16`, whose canonical per-operand inner
    tiles are LHS = `32x8` (M, K), RHS = `8x32` (K, N) and ACC = `32x32`
    (M, N). In practice the RHS is typically stored with its inner dims
    swapped (the matmul is really an MMT at the tile level); this is
    expressed by the `[1, 0]` entry in `permutations` on the RHS operand,
    which makes its inner-tile shape `canonical[1] x canonical[0]` =
    `N x K` = `32 x 8`.

    ```mlir
    %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>
        ],
        kind         = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
        semantics    = #iree_gpu.mma_semantics<distributed = false, opaque = false>,
        permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
      } : tensor<?x?x32x8xf16>, tensor<?x?x32x8xf16>
          into tensor<?x?x32x32xf32>
    ```

    After distribution across the 64 lanes of a wave, the per-operand
    inner tiles become the canonical *per-lane* shares of the intrinsic:
    `256 / 64 = 4` f16 elements per lane for LHS and RHS, `1024 / 64 = 16`
    f32 elements per lane for ACC.

    ```mlir
    %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>
        ],
        kind      = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
        semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
      } : tensor<?x?x4xf16>, tensor<?x?x4xf16>
          into tensor<?x?x16xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensorOrVector>:$inputs,
      Variadic<AnyRankedTensorOrVector>:$outputs,
      TypedArrayAttrBase<AffineMapAttr, "indexing affine maps">:$indexing_maps,
      IteratorTypeArrayAttr:$iterator_types,
      IREECodegen_AnyInnerTileDescAttr:$kind,
      IREECodegen_AnyInnerTiledSemanticscAttr:$semantics,
      OptionalAttr<TypedArrayAttrBase<DenseI64ArrayAttr,
                                      "permutations">>:$permutations);
  let results = (outs
    Variadic<AnyRankedTensorOrVector>:$results
  );

  let assemblyFormat = [{
    `ins` `(` $inputs `)` `outs` `(` $outputs `)` attr-dict
    `:` type($inputs) `into` type($outputs)
  }];

  let builders =
      [OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
           "ArrayAttr":$indexingMaps, "ArrayAttr":$iteratorTypes,
           "InnerTileDescAttrInterface":$intrinsic,
           "InnerTiledSemanticsAttrInterface":$semantics,
           CArg<"std::optional<ArrayAttr>", "std::nullopt">:$permutations)>,
       OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
           "ArrayRef<ArrayRef<AffineExpr>>":$indexingExprs,
           "ArrayRef<utils::IteratorType>":$iteratorTypes,
           "InnerTileDescAttrInterface":$intrinsic,
           "InnerTiledSemanticsAttrInterface":$semantics,
           CArg<"std::optional<SmallVector<SmallVector<int64_t>>>",
                "std::nullopt">:$permutationss)>,
       OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
           "ArrayRef<AffineMap>":$indexingMaps,
           "ArrayRef<utils::IteratorType>":$iteratorTypes,
           "InnerTileDescAttrInterface":$intrinsic,
           "InnerTiledSemanticsAttrInterface":$semantics,
           CArg<"std::optional<SmallVector<SmallVector<int64_t>>>",
                "std::nullopt">:$permutationss)>];
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

//===----------------------------------------------------------------------===//
// HintWorkgroupCountOp
//===----------------------------------------------------------------------===//

def IREECodegen_WorkgroupCountHintOp :
    Op<IREECodegen_Dialect, "workgroup_count_hint", []> {
  let summary = [{Hints at the workgroup count to set}];

  let description = [{
    Captures a set of values to use as the workgroup count. The backward slice
    starting from this op's operands is cloned into the workgroup count region
    of all transitive callers.

    The sizes are specified in logical order (innermost to outermost), matching
    the workgroup count region's (x, y, z) convention. If fewer than 3 sizes
    are provided, the remaining dimensions default to 1.

    If multiple hints inform the same entry point, the **elementwise maximum**
    across all hints is used as the count along each dimension. For example:

    ```mlir
    hal.executable.export @entry_point
    module {
      func.func @entry_point() {
        iree_codegen.workgroup_count_hint sizes(%a, %b, %c)
        iree_codegen.workgroup_count_hint sizes(%x, %y, %z)
      }
    }
    ```

    resolves to:

    ```mlir
    hal.executable.export @entry_point {
      %wx = arith.maxsi %a, %x
      %wy = arith.maxsi %b, %y
      %wz = arith.maxsi %c, %z
      hal.return %wx, %wy, %wz
    }
    ```

    #### Common Usage: Linearized Workgroup Counts

    The most common use case for this operation involves computing a linearized
    (1-D) workgroup count and specifying it as a single size.

    ```mlir
    %num_workgroups = arith.ceildivui %total_iterations, %tile_size : index
    iree_codegen.workgroup_count_hint sizes(%num_workgroups)
    ```

    This results in `(%num_workgroups, 1, 1)` as the final workgroup count.
  }];

  let arguments = (ins
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_sizes
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes)>
  ];

  let extraClassDeclaration = [{
    /// Returns a vector with all the static and dynamic sizes.
    SmallVector<OpFoldResult> getMixedSizes() {
      OpBuilder builder(getContext());
      return ::mlir::getMixedValues(getStaticSizes(), getSizes(), builder);
    }
  }];
}

//===----------------------------------------------------------------------===//
// IndexHintOp
//===----------------------------------------------------------------------===//

def IREECodegen_IndexHintOp : Op<IREECodegen_Dialect, "index_hint", [Pure]> {
  let summary = "Compiler hint providing semantic information about an index";
  let description = [{
    Pure pass-through operation that annotates an index value with semantic
    information about how it varies across parallel workers (e.g., GPU lanes).

    The hint attribute describes the index behavior. Common hints include:
    - `#iree_gpu.lane_constant<N>`: Index is uniform within groups of N lanes
    - `#iree_gpu.lane_increment<N>`: Index increments by 1 within groups of N lanes

    This operation is always safe to remove (replace with input). It exists
    purely to guide optimization passes.

    Example:
    ```mlir
    %row = iree_codegen.index_hint %idx(#iree_gpu.lane_constant<16>) : index
    %col = iree_codegen.index_hint %idx(#iree_gpu.lane_increment<16>) : index
    ```
  }];

  let arguments = (ins Index:$input, AnyAttr:$hint);
  let results = (outs Index:$result);

  let assemblyFormat = "$input `(` $hint `)` attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ConstraintsOp
//===----------------------------------------------------------------------===//

def IREECodegen_ConstraintsOp :
    Op<IREECodegen_Dialect, "smt.constraints",
       [IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "SMT constraints for a codegen configuration of root ops.";
  let description = [{
    Declares SMT constraints over problem dimensions and configuration
    knobs for a codegen pipeline, targeting a set of root ops.

    `target`: A `#iree_codegen.root_op<set = N>` attribute identifying
    which root op set these constraints apply to. All ops marked with
    the same `#iree_codegen.root_op<set = N>` attribute share the same
    lowering config. This decouples constraints from SSA values, so ops
    with zero or multiple results are supported.

    `pipeline`: The codegen pipeline to use. This is a fixed choice,
    not decided by the solver.

    `knobs`: DictionaryAttr mirroring GPULoweringConfigAttr. Leaves
    that are `#iree_codegen.smt.int_knob<"name">` attrs name tunable SMT
    constants (materialized by `iree_codegen.smt.knob` ops in the body).
    Integer/attr leaves are fixed.

    `problem_dims`: index-typed problem dimensions; corresponding block
    arguments are !smt.int.

    Does not have any execution semantics and is meant to be used by the
    tuner or verification passes, and erased before lowering.

    Example:
    ```mlir
    // The matmul is marked: {root_op = #iree_codegen.root_op<set = 0>}
    iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
     knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">, #iree_codegen.smt.int_knob<"wg_n">]}
     dims(%M, %N, %K) {
    ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
      %wg_m = iree_codegen.smt.knob "wg_m" : !smt.int
      %wg_n = iree_codegen.smt.knob "wg_n" : !smt.int
    }
    ```
  }];
  let arguments = (ins
    IREECodegen_RootOpAttr:$target,
    IREECodegen_PipelineAttrInterface:$pipeline,
    DictionaryAttr:$knobs,
    Variadic<Index>:$problem_dims
  );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);
  let assemblyFormat = [{
    `target` `=` $target `,` `pipeline` `=` $pipeline `,`
    custom<KnobsDictionary>($knobs)
    `dims` `(` $problem_dims `)` attr-dict-with-keyword
    $body
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def IREECodegen_AssertOp :
    Op<IREECodegen_Dialect, "smt.assert", [HasParent<"ConstraintsOp">]> {
  let summary = "Named SMT assertion for constraint verification.";
  let description = [{
    Asserts that a boolean SMT expression holds, with a human-readable
    format string describing what is being checked. The format string may
    contain `{}` placeholders that correspond positionally to the `args`
    operands. An evaluator or verifier can substitute concrete values into
    the placeholders for diagnostics.

    Args are restricted to `!smt.int` because all constraint knobs and
    dimension values are integers.

    Used inside `iree_codegen.smt.constraints` regions.

    Examples:
    ```mlir
    // Static message (no args):
    iree_codegen.smt.assert %cond, "dim_0 == 128" : !smt.bool

    // Format string with args:
    iree_codegen.smt.assert %cond, "wg_x ({}) < wg_y ({})", %x, %y : !smt.bool, !smt.int, !smt.int
    ```
  }];
  let arguments = (ins
    BoolType:$condition,
    StrAttr:$msg,
    Variadic<IntType>:$printArgs
  );
  let assemblyFormat = [{
    $condition `,` $msg (`,` $printArgs^)?
        `:` type($condition) (`,` type($printArgs)^)? attr-dict
  }];
  let builders = [
    // Convenience builder for assertions with no format args.
    OpBuilder<(ins "::mlir::Value":$condition, "::llvm::StringRef":$msg), [{
      build($_builder, $_state, condition, msg, ::mlir::ValueRange{});
    }]>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DispatchConfigOp
//===----------------------------------------------------------------------===//

def IREECodegen_DispatchConfigOp :
    Op<IREECodegen_Dialect, "dispatch_config", [
      IsolatedFromAbove,
      SingleBlockImplicitTerminator<"YieldOp">,
    ]> {
  let summary = [{Holds the workgroup count computation for a dispatch.}];
  let description = [{
    A module-level op that captures the workgroup count computation and
    dispatch metadata for a function. The region computes the workgroup
    count (x, y, z) from workload values passed as block arguments.
    The terminator must yield exactly 3 index values, matching the
    contract of `hal.executable.export`'s workgroup count region.

    The `function_ref` is a symbol reference to the corresponding
    `func.func`. The op itself is not a Symbol because it lives
    alongside the `func.func` of the same name in a `builtin.module`
    symbol table, and two ops cannot define the same symbol.

    Example:
    ```mlir
    iree_codegen.dispatch_config @matmul
        workgroup_size = [64, 16, 1] subgroup_size = 64 {
      ^bb0(%w0: index, %w1: index, %w2: index, %w3: index):
        %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 256)>()[%w2]
        %c1 = arith.constant 1 : index
        iree_codegen.yield %0, %c1, %c1 : index, index, index
    }
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$function_ref,
    OptionalAttr<DenseI64ArrayAttr>:$workgroup_size,
    OptionalAttr<I64Attr>:$subgroup_size
  );
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "::mlir::FlatSymbolRefAttr":$functionRef)>,
  ];

  let assemblyFormat = [{
    $function_ref
    (`workgroup_size` `=` $workgroup_size^)?
    (`subgroup_size` `=` $subgroup_size^)?
    $body attr-dict
  }];
  let extraClassDeclaration = [{
    /// Returns the static workgroup counts from the yield terminator.
    /// Dynamic values are represented as ShapedType::kDynamic.
    SmallVector<int64_t> getStaticNumWorkgroups();
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// KnobOp
//===----------------------------------------------------------------------===//

def IREECodegen_KnobOp :
    Op<IREECodegen_Dialect, "smt.knob", [Pure, HasParent<"ConstraintsOp">]> {
  let summary = "Declare an SMT constant for a tunable configuration knob.";
  let description = [{
    Materializes a named SMT constant (!smt.int) for use in constraint
    expressions. The name must match an `#iree_codegen.smt.int_knob<"name">`
    or `#iree_codegen.smt.one_of_knob<"name", [...]>` leaf in the
    enclosing `iree_codegen.smt.constraints` op's `knobs` dictionary.

    In SMT terminology this is a constant (0-ary function), not a variable.
    The tuner assigns concrete integer values to these constants.

    Example:
    ```mlir
    %wg_m = iree_codegen.smt.knob "wg_m" : !smt.int
    ```
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs IntType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// LookupOp
//===----------------------------------------------------------------------===//

def IREECodegen_LookupOp :
    Op<IREECodegen_Dialect, "smt.lookup", [Pure, HasParent<"ConstraintsOp">]> {
  let summary = "Integer table lookup for SMT constraints.";
  let description = [{
    Maps an SMT integer index to an integer value via a sparse key-value
    table. Used to derive values from enumerated knobs (e.g., MMA shape
    dimensions from an mma_idx knob).

    TODO: During constraint verification, the lookup will be evaluated
    directly. During SMT-LIB export, it will be lowered to a chain of
    `smt.ite` ops. Neither pass is implemented yet.

    Example:
    ```mlir
    %mma_m = iree_codegen.smt.lookup %idx [3, 7, 12] -> [16, 32, 64] : !smt.int
    ```
  }];
  let arguments = (ins
    IntType:$index,
    DenseI64ArrayAttr:$keys,
    DenseI64ArrayAttr:$values
  );
  let results = (outs IntType:$result);
  let assemblyFormat = "$index $keys `->` $values attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def IREECodegen_YieldOp :
    Op<IREECodegen_Dialect, "yield", [
      Pure,
      ReturnLike,
      Terminator,
    ]> {
  let summary = [{IREECodegen yield op.}];
  let description = [{
    `iree_codegen.yield` is a special terminator operation for blocks inside
    regions in `iree_codegen` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // IREE_CODEGEN_DIALECT_IREECODEGENOPS
