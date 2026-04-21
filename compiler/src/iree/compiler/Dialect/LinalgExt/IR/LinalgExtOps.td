// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_OPS
#define IREE_DIALECT_LINALGEXT_OPS

include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
include "iree/compiler/Utils/CommonTypeConstraints.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREELinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    IREELinalgExt_PureOp<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DestinationStyleOpInterface, LinalgExtInterface,
         SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::YieldOp">
  ])> {
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = "";
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def OpGroupNonStructuredOps : OpDocGroup {
  let summary = "Non-structured ops";
  let description = "";
}

let opDocGroup = OpGroupNonStructuredOps in {

def IREELinalgExt_ScatterOp : IREELinalgExt_Op<"scatter",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<LinalgFusionInterface,
      ["getIndexingMapsForResults", "getIndexingMapsForOperands",
       "getStaticLoopRanges"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateScalarImplementation",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation",
         "getIterationDomainTileFromOperandTiles",
         "getTiledImplementationFromOperandTiles"]>]> {
  let summary = [{Scatters an input in slices based on a tensor of indices.}];
  let description = [{
    Takes two or three `inputs` (`update`, `indices`, and optional `mask`) and
    one `outputs` value (`original`).
    The operation updates the value at the slices specified by `indices` by
    combining the current value with the value in `updates` using the computation
    specified in `region`. The `region` specifies a binary operation
    of signature `(T, T) -> T`, where `T` is the element-type of
    `updates` (and `original`). The first argument is from `updates`,
    and the second is from `original`.

    The size of the `dimension_map` attribute is used to determine how many
    indices are used to index into `original`, i.e. `index_depth`. The
    `dimension_map` attribute describes which index value maps to which dimension
    in the destination.

    The operand `indices` is a N-D tensor/memref type that is composed
    of two logical parts. The first `N-1` dimensions represent the batch of
    updates. The last dim (at index `N-1`) is the `index_depth`, which can be
    omitted if `index_depth` is 1.

    The operand `update` is a M-D tensor/memref type and similarly
    consists of two parts. The first `N-1` dimensions represent the batch of
    updates. This must exactly match to the first `N-1` dimensions in `indices`.
    Dimensions `N..M-1` represent the slice scattered into `original`,
    `update_slice`, and must match the last dimensions in original. This
    represents a contiguous slice to be inserted into `original`.

    The operand `original` is a DPS init representing the destination that
    `update` gets scattered to.
    Where `rank(original) = rank(update_slice) + index_depth`

    If the optional operand `mask` is present, it must have batch shape and a
    boolean element type (`i1`, or storage-legalized `i8` in backend
    pipelines). A `false` mask value suppresses the corresponding scatter
    update, leaving `original` unchanged for that batch element.

    The unique_indices attribute carries the information whether all the
    indices are unique. If `unique_indices` is `true` and two or more updates
    scatter to the same location in `original` the final value in `original` is
    not guaranteed. If `unique_indices` is set to false, the first
    `batch_rank` iteration loops will be marked as reduction.

    The shapes definition follows tensorflow operations. See more information in
      https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update
  }];
  let arguments = (ins
      AnyRankedTensorOrMemRef:$updates,
      AnyRankedTensorOrMemRef:$indices,
      Optional<AnyRankedTensorOrMemRef>:$mask,
      AnyRankedTensorOrMemRef:$original,
      DenseI64ArrayAttr:$dimension_map,
      DefaultValuedAttr<BoolAttr, "true">:$unique_indices
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict `dimension_map` `=` $dimension_map
              `unique_indices` `(` $unique_indices `)`
    `ins` `(` $updates `,` $indices (`,` $mask^)? `:` type($updates) `,` type($indices) (`,` type($mask)^)? `)`
    `outs` `(` $original `:` type($original) `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    int64_t getIndexDepth() {
      return getDimensionMap().size();
    }

    ShapedType getUpdateType() {
      return cast<ShapedType>(getUpdates().getType());
    }

    ShapedType getIndicesType() {
      return cast<ShapedType>(getIndices().getType());
    }

    std::optional<ShapedType> getMaskType() {
      if (Value mask = getMask()) {
        return cast<ShapedType>(mask.getType());
      }
      return std::nullopt;
    }

    ShapedType getOriginalType() {
      return cast<ShapedType>(getOriginal().getType());
    }

    /// Utility to get the rank of the portion of `indices` that
    /// represents the batch dimensions.
    int64_t getBatchRank() {
      return getUpdateType().getRank() - getUpdateSliceRank();
    }

    /// Utility to get the shape of the portion of `indices` that
    /// represents the batch dimensions.
    ArrayRef<int64_t> getBatchShape() {
      return getIndicesType().getShape().slice(0, getBatchRank());
    }

    /// Utility to get the rank of the portion of `updates` that
    /// is scattered into `original`.
    int64_t getUpdateSliceRank() {
      return getOriginalType().getRank() - getIndexDepth();
    }

    bool isScalarUpdate() {
      return getUpdateSliceRank() == 0;
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOriginalMutable();
    }
  }];
}

def IREELinalgExt_GatherOp : IREELinalgExt_Op<"gather",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<LinalgFusionInterface,
      ["getIndexingMapsForResults", "getIndexingMapsForOperands",
       "getStaticLoopRanges"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateScalarImplementation",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation",
         "generateResultTileValue"]>]> {
  let summary = [{Gathers slices from a source based on a tensor of indices.}];
  let description = [{
    Takes two or three inputs (`source`, `indices`, and optional `mask`) and
    one output value (`output`). The operation returns the value at the slices
    specified by `indices`.

    The size of the `dimension_map` attribute is used to determine how many
    indices are used to index into `source`, i.e. `index_depth`. The
    `dimension_map` attribute describes which index value maps to which dimension
    in the destination.

    This operation preforms the opposite operation of `iree_linalg_ext.scatter`.
    Instead of scattering `updates` into `original`, it gathers the values from
    `source` into `output` using the indices in `indices`. See the documentation
    on `iree_linalg_ext.scatter` for more details regarding the indexing/shape
    semantics.

    If the optional operand `mask` is present, it must have batch shape and a
    boolean element type (`i1`, or storage-legalized `i8` in backend
    pipelines). A `false` mask value suppresses the corresponding gather
    update, leaving `output` unchanged for that batch element.
  }];
  let arguments = (ins
      AnyRankedTensorOrMemRef:$source,
      AnyRankedTensorOrMemRef:$indices,
      Optional<AnyRankedTensorOrMemRef>:$mask,
      AnyRankedTensorOrMemRef:$output,
      DenseI64ArrayAttr:$dimension_map
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict `dimension_map` `=` $dimension_map
    `ins` `(` $source `,` $indices (`,` $mask^)? `:` type($source) `,` type($indices) (`,` type($mask)^)? `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    /// Utility to get the number of indices used to index into `source`.
    int64_t getIndexDepth() {
      return getDimensionMap().size();
    }

    /// Utility to get rank of the portion of `output` that is contiguous.
    int64_t getOutputSliceRank() {
      return getSourceType().getRank() - getIndexDepth();
    }

    /// Utility to get the rank of the portion of `indices` that represents the
    /// batch dimensions.
    int64_t getBatchRank() {
      return getOutputType().getRank() - getOutputSliceRank();
    }

    ShapedType getSourceType(){
      return cast<ShapedType>(getSource().getType());
    }

    ShapedType getIndicesType(){
      return cast<ShapedType>(getIndices().getType());
    }

    std::optional<ShapedType> getMaskType() {
      if (Value mask = getMask()) {
        return cast<ShapedType>(mask.getType());
      }
      return std::nullopt;
    }

    ShapedType getOutputType(){
      return cast<ShapedType>(getOutput().getType());
    }

    /// For DPS interface.
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];

  let hasCanonicalizer = 1;
}

def IREELinalgExt_MapLoadOp : IREELinalgExt_Op<"map_load",
    [DeclareOpInterfaceMethods<TilingInterface,
        ["getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation",
         "generateResultTileValue",
         "getIterationDomainTileFromOperandTiles",
         "getTiledImplementationFromOperandTiles",
         "generateScalarImplementation"]>]> {
  let summary = [{Load with a mapping from result indices to source indices.}];
  let description = [{
    Takes two operands, `source` and `output`, and reads every element from
    the `source` based on a computed index mapping to store into the `output`.
    If the operands are tensors, the op will also return a Value for the result.
    For an element of the `output`, the index to read from the `source` is
    determined by index computation mapping the index in the `output` value to
    an index in the `source` value. This computation is contained in the
    `transformation_region`, which contains a single block, with one argument
    for each dimension in the `output` value. The block arguments represent the
    index of a given element along the corresponding dimension (i.e., block arg
    `i` represents the index along dimension `i` of the `output` value). The
    block is terminated by an iree_linalg_ext.yield op, which yields one index
    for each dimension in the `source`, and an additional padding value which
    represents the value to use if the computed source indices are out of bounds.
  }];
  let arguments = (ins
      AnyShaped:$source,
      AnyRankedTensorOrMemRef:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$transformation_region);
  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict $source `into` $output
    $transformation_region
    `:` type($source) `into` type($output) (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    ShapedType getSourceType() {
      return cast<ShapedType>(getSource().getType());
    }
    int64_t getSourceRank() {
      return getSourceType().getRank();
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }

    // Helper to apply transformations to the output index block arguments of
    // the transformation body, and replace the uses of the previous output
    // indices with the values returned by `transformationBuilder`. The argument
    // to `transformationBuilder` function will be an ArrayRef of the new block
    // arguments, the number of which is determined by `numOutputIndices`. The
    // `transformationBuilder` should return the replacements for the old block
    // arguments.
    void insertTransformationAtStart(
        OpBuilder &builder,
        function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
            transformationBuilder,
        int64_t numOutputIndices);

    // Inline the transformation region of the map_load op without its
    // terminator, replacing the block arguments with the passed
    // `transformBodyIndices`. The `bodyBuilder` function is called with the
    // cloned `Value`s that would have been yielded by the terminator of
    // the inlined transformation body (source indices and padding value).
    void inlineMapLoadBody(
      OpBuilder &b, Location loc, ValueRange transformBodyIndices,
      function_ref<void(OpBuilder &, Location, ArrayRef<Value>)> bodyBuilder);

    // Return true if the mapLoadOp is an identity transformation, meaning
    // it is just doing a copy from source to output. This is true if the
    // padding is unused (source and output shapes match) and the mapping
    // from output to source index is identity.
    bool isIdentity();

    // Return the padding value yielded by the transformation region.
    // The padding value is the last operand of the yield op.
    Value getPaddingValue();

    // Return true if the map_load op has vector semantics.
    bool isVectorized() {
      return isa<VectorType>(getSource().getType());
    }

    // Creates a MapLoadOp with an identity mapping in the transformation
    // body. Expects the types of `source` and `output` to be the same.
    static MapLoadOp createIdentityMapLoad(OpBuilder &builder,
                                               Location loc, Value source,
                                               Value output);

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];

  let hasCanonicalizer = 1;
}

def IREELinalgExt_MapStoreOp : IREELinalgExt_Op<"map_store",
    [DeclareOpInterfaceMethods<TilingInterface,
        ["getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation",
         "getIterationDomainTileFromOperandTiles",
         "getTiledImplementationFromOperandTiles",
         "generateScalarImplementation"]>]> {
  let summary = [{Store with a mapping from source indices to result indices.}];
  let description = [{
    Takes two operands, `input` and `output`, and stores every element of
    `input` to a unique location in `output`. If the operands are tensors, the
    op will also return a Value for the result. For an element of the `input`,
    the index to store into the `output` is determined by index computation
    mapping the index in the `input` value to an index in the `output` value.
    This computation is contained in the `transformation_region`, which contains
    a single block, with one argument for each dimension in the `input` value.
    The block arguments represent the index of a given element along the
    corresponding dimension (i.e., block arg `i` represents the index along
    dimension `i` of the `input` value). The block is terminated by an
    iree_linalg_ext.yield op, which yields one index for each dimension in
    the `output`, and an additional `i1` Value, which represents a mask on
    whether or not to write to the `output` at the yielded set of indices. If
    the mask value is true, the input value will be written.
  }];
  let arguments = (ins
      AnyShaped:$input,
      AnyRankedTensorOrMemRef:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$transformation_region);
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    attr-dict $input `into` $output
    $transformation_region
    `:` type($input) `into` type($output) (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }

    // Return the input index at the provided position.
    Value getInputIndex(int64_t position);

    // Return the output index at the provided position.
    Value getOutputIndex(int64_t position);

    // Returns the mask value.
    Value getMask();

    // Return true if the map_store op has vector semantics.
    bool isVectorized() {
      return isa<VectorType>(getInputType());
    }

    // Helper to apply transformations to the source index block arguments of
    // the transformation body, and replace the uses of the previous source
    // indices with the values returned by `transformationBuilder`. The argument
    // to `transformationBuilder` function will be an ArrayRef of the new block
    // arguments, the number of which is determined by `numSourceIndices`. The
    // `transformationBuilder` should return the replacements for the old block
    // arguments.
    void insertTransformationAtStart(
        OpBuilder &builder,
        function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
            transformationBuilder,
        int64_t numSourceIndices);

    // Inline the transformation region of the map_store op without its
    // terminator, replacing the block arguments with the passed
    // `transformBodyIndices`. The `bodyBuilder` function is called with the
    // cloned index `Value`s that would have been yielded by the terminator of
    // the inlined transformation body.
    void inlineMapStoreBody(
      OpBuilder &b, Location loc, ValueRange transformBodyIndices,
      function_ref<void(OpBuilder &, Location, ArrayRef<Value>)> bodyBuilder);

    // Return true if the mapStoreOp is an identity transformation, meaning
    // it is just doing a copy from input to output. This is true if the mask
    // always evaluates to true, and the mapping from input to output index is
    // identity.
    bool isIdentity();

    // Creates a MapStoreOp with an identity mapping in the transformation
    // body. Expects the types of `input` and `output` to be the same.
    static MapStoreOp createIdentityMapStore(OpBuilder &builder,
                                                 Location loc, Value input,
                                                 Value output);

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];
}

def IREELinalgExt_SortOp : IREELinalgExt_Op<"sort",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateScalarImplementation",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation"]>]> {
  let summary = [{Sorts a tensor a specified dimension.}];
  let description = [{
    Based on XLA operation semantics, sorts the given `operands` at the given
    `dimension` with the given `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getOperand(int index) {
      return getOutputs()[index];
    }
    ShapedType getOperandType(int index) {
      return cast<ShapedType>(getOperand(index).getType());
    }
    int64_t getOperandRank() {
      return getOperandType(0).getRank();
    }
    ArrayRef<int64_t> getOperandShape() {
      return getOperandType(0).getShape();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];

  let hasCanonicalizer = 1;
}

def IREELinalgExt_FftOp : IREELinalgExt_Op<"fft", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
  DeclareOpInterfaceMethods<TilingInterface,
      ["generateScalarImplementation",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = [{Fft operator.}];
  let description = [{
    Apply 1D FFT to innermost dim. This is an iterative FFT, not recurrsive.
    Thus, the bit reversal is assumed applied on the input. The op carries an
    input -- stage, which indicates the level of reduction loop in the
    algorithm. It represents the computation body. For more details, see
    "Data reordering, bit reversal, and in-place algorithms" section in
    https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

    The size of innermost dim is expected to be a power of 2.

    It is optional to carry coefficient tensors/buffers as inputs. In this
    context, they will be the second and third inputs.
  }];

  let arguments = (ins AnyType:$stage,
                       Optional<AnyShaped>:$real_coeff,
                       Optional<AnyShaped>:$imag_coeff,
                       AnyShaped:$real,
                       AnyShaped:$imag
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $stage (`,` $real_coeff^ `,` $imag_coeff)? `:` type($stage) (`,` type($real_coeff)^ `,` type($imag_coeff))? `)`
    `outs` `(` $real `,` $imag `:` type($real) `,` type($imag) `)`
    (`:` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    bool hasCoeff() { return static_cast<bool>(getRealCoeff()); }
    void generateScalarImplWithoutCoeffBuf(
        OpBuilder & b, Location loc, ArrayRef<Value> operands, Value wholeSize);
    void generateScalarImplWithCoeffBuf(OpBuilder & b, Location loc,
                                        ArrayRef<Value> operands);
    ShapedType getOperandType() {
      return cast<ShapedType>(getReal().getType());
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
    ArrayRef<int64_t> getOperandShape() {
      return getOperandType().getShape();
    }
    int64_t getFftLength() {
      return getOperandShape().back();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable();
  }];
}

def IREELinalgExt_ScanOp : IREELinalgExt_Op<"scan",
    [IndexingMapOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["generateScalarImplementation",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = [{Scan operator.}];
  let description = [{
    Computes the inclusive/exclusive scan along a given dimension.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       AnyShaped:$accumulator,
                       I64Attr:$dimension,
                       BoolAttr:$inclusive
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `,` $accumulator `:` type($output) `,` type($accumulator) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    ShapedType getOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }

    ArrayAttr getIndexingMaps();
    SmallVector<AffineMap> getIndexingMapsArray();

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable();
  }];
}

def IREELinalgExt_TopkOp : IREELinalgExt_Op<"topk",[
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["generateScalarImplementation",
     "getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation"]>
]>{
  let summary = [{Top-K operator.}];
  let description = [{
   A Top-K operation for N-D tensors. Reduces the target dimension from the input
   size N down to K elements based on the supplied binary region.

   Accepts an N-D tensor input consisting of values and an optional N-D tensor
   for indices of those values (i32 type). If input indices aren't provided, the
   index mapping is inferred based on the k dim.  Both input values/indices
   tensors and output values/indices tensors must have the same shape. Top-K is
   computed along the target dimension (from dimension()). Returns two output
   tensors of values and the indices of Top-K results. The output dimensions
   must match the input save for the dimension that is reduced to K results.

   Region accepts lhs=[next N input] and rhs=[exiting K output] and yields an
   i1. If true, the two values are swapped:
     - For Top-K compoarision: >
     - For Min-K comparison: <
   Note: when the two values are equal, the first occurrence is always selected.
  }];

  let arguments = (ins AnyShaped:$values,
                       Optional<AnyShaped>:$indices,
                       AnyShaped:$output_values,
                       AnyShaped:$output_indices,
                       I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $values (`,` $indices^)? `:` type($values) (`,` type($indices)^)? `)`
    `outs` `(` $output_values `,` $output_indices `:` type($output_values) `,` type($output_indices) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    ShapedType getInputType() {
      return cast<ShapedType>(getValues().getType());
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable();
  }];
}

def IREELinalgExt_TopkV2Op : IREELinalgExt_Op<"topk_v2",[
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["generateScalarImplementation",
     "getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation"]>
]>{
  let summary = [{Top-K v2 operator.}];
  let description = [{
    Selects the top-K elements along the given `dimension` using the given
    `comparator`. K is determined by the output shape. The output sort
    dimension must be less than or equal to the input sort dimension.

    This op replaces the legacy `topk` op with several improvements:
    output indices are optional instead of mandatory, input indices of any
    integer type are supported for pre-permuted inputs, and the comparator
    region uses a pure ordering predicate decoupled from the swap semantics
    of the legacy op.

    Accepts a single N-D input tensor of values and an optional N-D tensor of
    input indices (any integer type). If input indices aren't provided, the
    index mapping defaults to [0, N) along the sort dimension, where N is
    the input size. Input indices may only be provided when output indices
    are also present, since the comparator operates solely on values and
    input indices only carry positional provenance to the output. Produces
    output values and optionally output indices (integer type) tracking
    original positions.

    When `is_sorted` is present, the output top-K elements are guaranteed to
    be in sorted order along the sort dimension. When absent (default), the
    top-K elements are returned but their relative order is unspecified.

    Comparator region accepts two scalar arguments of the input value element
    type and yields an i1 indicating whether the first argument should be
    ordered before the second.
  }];

  let arguments = (ins I64Attr:$dimension,
                       UnitAttr:$is_sorted,
                       AnyShaped:$values,
                       Optional<AnyShaped>:$input_indices,
                       AnyShaped:$output_values,
                       Optional<AnyShaped>:$output_indices
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    (`is_sorted` $is_sorted^)?
    `ins` `(` $values (`,` $input_indices^)? `:` type($values) (`,` type($input_indices)^)? `)`
    `outs` `(` $output_values (`,` $output_indices^)? `:`
               type($output_values) (`,` type($output_indices)^)? `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    ShapedType getInputType() {
      return cast<ShapedType>(getValues().getType());
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
    bool hasOutputIndices() {
      return static_cast<bool>(getOutputIndices());
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable();
  }];
}

def IREELinalgExt_ArgCompareOp : IREELinalgExt_Op<"arg_compare", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
  DeclareOpInterfaceMethods<LinalgFusionInterface,
   ["getIndexingMapsForResults", "getIndexingMapsForOperands",
    "getStaticLoopRanges"]>,
  DeclareOpInterfaceMethods<IndexingMapOpInterface, ["getMatchingIndexingMap"]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
  ["generateScalarImplementation",
   "getLoopIteratorTypes",
   "getResultTilePosition",
   "getTiledImplementation",
   "generateResultTileValue",
   "getIterationDomain"]
  >,
  DeclareOpInterfaceMethods<PartialReductionOpInterface,
   ["generateInitialTensorForPartialReduction",
    "tileToPartialReduction",
    "mergeReductions",
    "getPartialResultTilePosition"]>]> {
  let summary = "Performs an arg-reduction using a user-defined comparator.";
  let description = [{
    The `arg_compare` op performs a reduction over a given dimension of a tensor,
    returning both the selected value and its corresponding index. The selection
    logic is defined by a user-specified comparator region.

    The operation supports three index modes for flexibility in tiling:

    1. Implicit-index without index_base (single input, no index_base):
       Indices are computed from local position (linalg.index on reduction dimension).
       Used for untiled operations or when global offset is not needed.

    2. Implicit-index with index_base (single input, with index_base):
       Indices are computed as: index_base + local position.
       Used in OuterParallel tiling where each parallel chunk has a fixed offset.

    3. Explicit-index (two inputs: values + indices, no index_base):
       Indices are provided as input tensor alongside values.
       Used in OuterReduction tiling and merge operations where indices are
       accumulated across iterations.

    The comparator region defines the sorting rule, e.g., "greater than" for argmax or
    "less than" for argmin, allowing for generalization beyond simple argmax-style
    behavior.

    Example (implicit-index mode - argmax over dim 1):
    ```mlir
    %input = memref<2x10xf32>
    %out_val = memref<2xf32>
    %out_idx = memref<2xi32>
    iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : memref<2x10xf32>)
      outs(%out_val, %out_idx : memref<2xf32>, memref<2xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
    }
    ```

    Example (implicit-index mode with index_base = 5):
    ```mlir
    %input = memref<2x10xf32>
    %out_val = memref<2xf32>
    %out_idx = memref<2xi32>
    %base = arith.constant 5 : index
    iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : memref<2x10xf32>)
      outs(%out_val, %out_idx : memref<2xf32>, memref<2xi32>)
      index_base(%base : index) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
    }
    ```

    Example (explicit-index mode - combining partial results):
    ```mlir
    %partial_vals = memref<2x4xf32>
    %partial_idxs = memref<2x4xi32>
    %out_val = memref<2xf32>
    %out_idx = memref<2xi32>
    iree_linalg_ext.arg_compare
      dimension(1)
      ins(%partial_vals, %partial_idxs : memref<2x4xf32>, memref<2x4xi32>)
      outs(%out_val, %out_idx : memref<2xf32>, memref<2xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
    }
    ```

    The `index_base` is optional. When specified, it is added to the selected index
    in the result, which is useful when reducing over a tiled or sliced subregion.
  }];

  let arguments = (ins
    AnyShaped:$input_value,
    Optional<AnyShaped>:$input_index,
    AnyShaped:$output_value,
    AnyShaped:$output_index,
    Optional<Index>:$index_base,
    I64Attr:$dimension
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );
  let regions = (region SizedRegion<1>:$region);
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $input_value (`,` $input_index^)? `:` type($input_value) (`,` type($input_index)^)? `)`
    `outs` `(` $output_value `,` $output_index `:` type($output_value) `,` type($output_index) `)`
    (`index_base` `(` $index_base^ `:` type($index_base) `)`)?
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    // Returns true when the operation is in explicit-index mode, which occurs
    // when a second input operand provides pre-computed indices.
    bool hasExplicitIndexInput() {
      return static_cast<bool>(getInputIndex());
    }

    ShapedType getInputType() {
      return cast<ShapedType>(getInputValue().getType());
    }

    ShapedType getOutputValueType() {
      return cast<ShapedType>(getOutputValue().getType());
    }

    ShapedType getOutputIndexType() {
      return cast<ShapedType>(getOutputIndex().getType());
    }

    // Returns the ShapedType of the index input, or nullptr if not in
    // explicit-index mode.
    ShapedType getInputIndexType() {
      if (Value idx = getInputIndex()) {
        return cast<ShapedType>(idx.getType());
      }
      return nullptr;
    }

    // Returns the element type of the index input, or nullptr if not in
    // explicit-index mode.
    Type getInputIndexElementType() {
      if (ShapedType idxType = getInputIndexType()) {
        return idxType.getElementType();
      }
      return nullptr;
    }

    Type getOutputIndexElementType() {
      return getOutputIndexType().getElementType();
    }

    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // For DPS interface.
    MutableOperandRange getDpsInitsMutable();

    SmallVector<AffineMap> getIndexingMapsArray();
  }];
}

//===----------------------------------------------------------------------===//
// Attention
//===----------------------------------------------------------------------===//

def IREELinalgExt_AttentionOp : IREELinalgExt_Op<"attention",
    [DeclareOpInterfaceMethods<LinalgFusionInterface,
      ["getIndexingMapsForResults", "getIndexingMapsForOperands",
       "getStaticLoopRanges"]>,
     DeclareOpInterfaceMethods<IndexingMapOpInterface, ["getMatchingIndexingMap"]>,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = [{Attention operator.}];
  let description = [{
    Computes the scaled dot product attention function:

    attention(Q, K, V, scale) = softmax(Q @ K.T * scale) @ V

    Here Q, K, V are given tensors and scale is a scalar value specifying
    the scale to use.

    If an additional mask argument M is included, the result of the first matmul is modified according to:

    Q @ K.T += M
  }];

  let arguments = (ins AnyShaped:$query,
                       AnyShaped:$key,
                       AnyShaped:$value,
                       AnyFloat:$scale,
                       Optional<AnyShaped>:$mask,
                       AnyShaped:$output,
                       AffineMapArrayAttr:$indexing_maps,
                       OptionalAttr<DictionaryAttr>:$decomposition_config
  );
  let regions = (region SizedRegion<1>:$region);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $query `,` $key `,` $value `,` $scale (`,` $mask^)?  `:` type($query) `,` type($key) `,` type($value) `,` type($scale) (`,` type($mask)^ )?`)`
    `outs` `(` $output `:` type($output) `)`
    $region
    (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results,
      "Value":$query,
      "Value":$key,
      "Value":$value,
      "Value":$scale,
      "Value":$output,
      "ArrayAttr":$indexing_maps,
      CArg<"std::optional<Value>", "std::nullopt">:$mask)>,
    OpBuilder<(ins "TypeRange":$results,
      "ValueRange":$inputOperands,
      "ValueRange":$initOperands,
      "ArrayAttr":$indexing_maps)>
  ];

  let extraClassDeclaration = [{
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable();

    SmallVector<AffineMap> getIndexingMapsArray();

    AffineMap getQueryMap() {
      return cast<AffineMap>(getIndexingMapsArray()[0]);
    }
    AffineMap getKeyMap() {
      return cast<AffineMap>(getIndexingMapsArray()[1]);
    }
    AffineMap getValueMap() {
      return cast<AffineMap>(getIndexingMapsArray()[2]);
    }
    AffineMap getScaleMap() {
      return cast<AffineMap>(getIndexingMapsArray()[3]);
    }
    std::optional<AffineMap> getMaskMap() {
      if (getMask()) {
        return cast<AffineMap>(getIndexingMapsArray()[4]);
      }
      return std::nullopt;
    }
    AffineMap getOutputMap() {
      return cast<AffineMap>(getIndexingMapsArray()[getNumDpsInputs()]);
    }
    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }

    /* Decomposition control attributes */

    // Attributes to set on QK and PV matmul after decomposition.
    static StringRef getQKAttrStr() { return "qk_attrs"; }
    static StringRef getPVAttrStr() { return "pv_attrs"; }
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// OnlineAttention
//===----------------------------------------------------------------------===//

def IREELinalgExt_OnlineAttentionOp : IREELinalgExt_Op<"online_attention",
    [DeclareOpInterfaceMethods<LinalgFusionInterface,
      ["getIndexingMapsForResults", "getIndexingMapsForOperands",
       "getStaticLoopRanges"]>,
     IndexingMapOpInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>,
     DeclareOpInterfaceMethods<PartialReductionOpInterface,
      ["generateInitialTensorForPartialReduction",
       "tileToPartialReduction",
       "mergeReductions",
       "getPartialResultTilePosition"]>]> {
  let summary = [{Online Attention operator.}];
  let description = [{
    Traditional scaled dot product attention computes:

    attention(Q, K, V, scale) = softmax(Q @ K.T * scale) @ V

    Online Attention on the other hand, uses an online normalizer instead of
    softmax:

    online_attention(Q, K, V, scale, running_max, running_sum)
      = online_normalizer(Q @ K.T * scale, running_max, running_sum) @ V

    If an additional mask argument M is included, the result of the first matmul is modified according to:

    Q @ K.T += M

    The advantage of this online_normalizer is that it can be tiled along
    its reduction dimension, making the online_attention operator:
      - Tilable along softmax reduction dimension
      - Associative along softmax reduction dimension
      - Commutative along softmax associative dimension

    Note: The results of online_attention need to be combined after computing
    it over the entire softmax reduction dimension by:
      x, _, sum : results
      x = (1 / sum) * x

    Decomposition Configuration:
    The `decomposition_config` attribute is a DictionaryAttr that controls how
    this operation is decomposed into lower-level operations. It supports:
      - "qk_attrs": DictionaryAttr - Attributes to attach to the Q@K matmul
        operation after decomposition (e.g., lowering_config, attention markers)
      - "pv_attrs": DictionaryAttr - Attributes to attach to the P@V matmul
        operation after decomposition
      - "use_exp2": BoolAttr - If true, uses exp2 with log2(e) scaling instead
        of exp. (Gives better perf on some hardware, but trades off accuracy)
  }];

  let arguments = (ins AnyShaped:$query,
                       AnyShaped:$key,
                       AnyShaped:$value,
                       AnyFloat:$scale,
                       Optional<AnyShaped>:$mask,
                       AnyShaped:$output,
                       AnyShaped:$max,
                       AnyShaped:$sum,
                       AffineMapArrayAttr:$indexing_maps,
                       OptionalAttr<DictionaryAttr>:$decomposition_config
  );
  let regions = (region SizedRegion<1>:$region);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $query `,` $key `,` $value `,` $scale (`,` $mask^)?  `:` type($query) `,` type($key) `,` type($value) `,` type($scale) (`,` type($mask)^ )?`)`
    `outs` `(` $output `,` $max `,` $sum `:` type($output) `,` type($max) `,` type($sum) `)`
    $region
    (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results,
    "Value":$query,
    "Value":$key,
    "Value":$value,
    "Value":$scale,
    "Value":$output,
    "Value":$max,
    "Value":$sum,
    "ArrayAttr":$indexing_maps,
    CArg<"std::optional<Value>", "std::nullopt">:$mask)>
  ];


  let extraClassDeclaration = [{
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable();

    SmallVector<AffineMap> getIndexingMapsArray();

    AffineMap getQueryMap() {
      return getIndexingMapsArray()[0];
    }
    AffineMap getKeyMap() {
      return getIndexingMapsArray()[1];
    }
    AffineMap getValueMap() {
      return getIndexingMapsArray()[2];
    }
    AffineMap getScaleMap() {
      return getIndexingMapsArray()[3];
    }
    std::optional<AffineMap> getMaskMap() {
      if (getMask()) {
        return cast<AffineMap>(getIndexingMapsArray()[4]);
      }
      return std::nullopt;
    }
    AffineMap getOutputMap() {
      return cast<AffineMap>(getIndexingMapsArray()[getNumDpsInputs()]);
    }
    AffineMap getMaxMap() {
      return cast<AffineMap>(getIndexingMapsArray()[getNumDpsInputs() + 1]);
    }
    AffineMap getSumMap() {
      return cast<AffineMap>(getIndexingMapsArray()[getNumDpsInputs() + 2]);
    }

    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }
    int64_t getNumLoops() { return getIterationDomainRank(); }

    /* Decomposition control attributes */

    // Attributes to set on QK and PV matmul after decomposition.
    static StringRef getQKAttrStr() { return "qk_attrs"; }
    static StringRef getPVAttrStr() { return "pv_attrs"; }
    // Flag to control whether to use exp2 (with log2(e) scaling) or exp.
    static StringRef getUseExp2AttrStr() { return "use_exp2"; }
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ExpReduction
//===----------------------------------------------------------------------===//

def IREELinalgExt_ExpReductionOp : IREELinalgExt_Op<"exp_reduction", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
  DeclareOpInterfaceMethods<IndexingMapOpInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
  [
    "getLoopIteratorTypes",
    "getIterationDomain",
    "getTiledImplementation",
    "getResultTilePosition",
    "generateResultTileValue"
  ]>
]> {
  let summary = [{
    A linalg.generic extension with support for exponential reduction.
  }];

  let description = [{
    This operation is a restricted form of a linalg.generic operation,
    representing a reduction over `e^{x}` where `x` is an input, and it is known
    that the function's result is independent of translation over x,
      i.e. f(x) = f(x + c).

    Since `e{x}` grows exponentially for `x > 0`, if the values of `x` are
    unbounded, the intermediate computation for `e^{x}` in `f(x)`, may not fit
    in the floating point range, even if the result does. To prevent such
    numerical inaccuracies, we can instead compute `f(max(x) - x)`, which will
    ensure that `e{max(x) - x}` always fits in the range (0, 1], because
    0 < e{negative} <= 1.

    These properties allow us to compute an online normalization of this
    function, in which we don't have to compute the `max(x)` as a separate
    reduction, but as a running max.

    Generally, these properties are hard to infer from a linalg.generic, because
    two back-to-back reduction combiners in a linalg.generic can have hard to
    infer properties. This restricted form ensures we don't have to do
    complicated analysis to infer these properties.

    The first input and output operands are intrinsically linked.
    - The first output operand is always the maximum of the first input operand,
      henceforth 'nmax'.
    - The first input operand in the basic block is always `2^{nmax-x}`, where
      `x` is the value

    `norm` is calculated by `2^{nmax - pmax}`, where
    - `pmax` is the value from the previous iteration.
    - `nmax` is the new maximum, read from the current element from the input
      tensor

    The other input operands are unaffected.

    The operands listed in `exp_reduced_operands` will be multiplied by `norm`
    before executing the basic block. Other output operands are unaffected.

    For example, we write the online normalized softmax algorithm as follows:

    ```mlir
    // compute the exponentials and the sum
    %M, %O = iree_linalg_ext.exp_reduction {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> ()>,
        affine_map<(i) -> ()>
      ],
      iterator_types = [
        #iree_linalg_ext.iterator_type<reduction>
      ],
      exp_reduced_operands = [ 1 ]  // 1 corresponds to %s being normalized by %m
    } ins(%V: tensor<256xf32>)
      outs(%M_init, %S: tensor<f32>, tensor<f32>) {
      ^bb0(%exp_v: f32, %m: f32, %s: f32):
        // take note of the operand names when comparing with the subsequent
        // code snippets
        %add = arith.addf %exp_v, %s : f32
        linalg.yield %m, %add        : f32, f32
    } -> tensor<f32>, tensor<f32>
  ```

  This encodes the following computation, sans handling of the norms

  ```
  %M, %O = linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> ()>,
        affine_map<(i) -> ()>
      ],
  } ins(%V: tensor<256xf32>)
    outs(%M_init, %S: tensor<f32>, tensor<f32>) {
    ^bb0(%v: f32, %m: f32, %s: f32):
      %exp_v = math.exp2 %v       : f32
      %add = arith.addf %exp_v, %s : f32
      linalg.yield %m, %add       : f32, f32
  } -> tensor<f32>, tensor<f32>
  ```

  And decomposes into operations equivalent to the following:

  ```
  %M, %O = linalg.generic {
      indexing_maps = [
        affine_map<(i) -> (i)>,
        affine_map<(i) -> ()>,
        affine_map<(i) -> ()>
      ],
  } ins(%V: tensor<256xf32>)
    outs(%M_init, %S: tensor<f32>, tensor<f32>) {
    ^bb0(%v: f32, %prev_m: f32, %prev_s: f32):
      %m      = arith.maximumf %v, %prev_m : f32
      %m_diff = arith.subf %m, %prev_m     : f32
      %norm   = math.exp2 %m_diff          : f32

      %exp_v  = math.exp2 %v               : f32  // raise v to the exponent
      %s      = arith.mulf %prev_s, %norm  : f32  // multiply s by the norm
      %add    = arith.addf %exp_v, %s      : f32
      linalg.yield %m, %add                : f32, f32
  } -> tensor<f32>, tensor<f32>
  ```
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyRankedTensor>:$outputs,
    AffineMapArrayAttr:$indexing_maps,
    IREELinalgExt_IteratorTypeArrayAttr:$iterator_types,
    DenseI64ArrayAttr:$exp_reduced_operands);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `{`
      `indexing_maps` `=` $indexing_maps `,`
      `iterator_types` `=` $iterator_types `,`
      `exp_reduced_operands` `=` $exp_reduced_operands
    `}`
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];


  let extraClassDeclaration = [{
    static std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>, llvm::function_ref<mlir::InFlightDiagnostic()>)>
    getRegionBuilder() {
      return nullptr;
    }

    // DestinationStyleOpInterface methods.
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }

    int64_t getReducingOpIndex() {
      return 0;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Im2col
//===----------------------------------------------------------------------===//

def IREELinalgExt_Im2colOp : IREELinalgExt_Op<"im2col",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = [{Im2col operation for convolutions.}];
  let description = [{
    Transforms a convolution input tensor into a GEMM-compatible layout. Each
    output position `(batch, m, k)` maps to a specific input element determined
    by the convolution metadata (strides, dilations, kernel_size).

    #### Attributes

    **Convolution metadata** — derived from the original convolution:
    - `strides`: spatial stride per m_pos dimension.
    - `dilations`: dilation factor per m_pos dimension.
    - `kernel_size`: filter kernel extent per m_pos dimension.

    **Dimension mapping** — describes which input dimensions are batch, spatial
    (M), or reduction (K):
    - `batch_pos`: input dimensions that are batch (pass-through to output).
    - `m_pos`: input dimensions that are spatial (indexed by stride/dilation).
    - `k_pos`: input dimensions that are channels (direct index).

    **Indexing**:
    - `offsets`: one offset per output dimension in canonical
      `[Batch..., M..., K...]` order. Updated by tiling to reflect the tile
      position; `output_sizes` remain unchanged.
    - `output_sizes`: nested list of delinearization sizes per output dimension.
      Each entry describes how to delinearize that output dimension back to
      spatial/channel coordinates. For example:
      `output_sizes = [[2], [32, 32], [3, 3, 640]]` means batch=[2],
      M delinearizes into [OH=32, OW=32], K delinearizes into [KH=3, KW=3, C=640].

    **Layout permutations**:
    - `input_k_perm`: permutation that maps the delinearized K output dimension
      coordinates (window offsets from spatial dims and channel indices) to their
      corresponding input dimension positions. Used when
      the filter layout differs from the input layout (e.g., input=HWC,
      filter=CHW requires `input_k_perm = [2, 0, 1]`). The identity permutation
      means the layouts are already aligned.
    - `output_perm`: permutation of the canonical `[Batch, M, K]` order to the
      actual output tensor layout. For example, `output_perm = [2, 0, 1]` means
      the result is in `KxBxM` layout.

    **Padding** (optional):
    - `input_pad_low` / `input_pad_high`: per-input-dimension padding amounts,
      representing absorbed input padding (e.g., from a folded tensor.pad).
      Source coordinates are computed in the padded coordinate space; reads
      outside the unpadded input bounds produce `pad_value`.
    - `output_pad_low` / `output_pad_high`: per-output-dimension padding
      amounts in actual output tensor dim order, representing how many
      positions at the start/end of each output dimension are outside the
      valid region. Valid region for output dim d is
      `[output_pad_low[d], tensor_dim[d] - output_pad_high[d])`.
      Updated during tiling to reflect tile-local padding amounts.
    - `pad_value`: the value for out-of-bounds positions. Can be present
      without `input_pad_low`/`input_pad_high` when only output padding exists
      (e.g., from GEMM alignment). When absent, there is no padding.

    #### Semantics

    For each output position `(b, m, k)`, the op computes input coordinates,
    where `i` indexes over `m_pos` dimensions (spatial) and `j` indexes over
    `k_pos` dimensions (channel):
    ```
      spatial_coord[i] = m_coord[i] * stride[i] + window_offset[i] * dilation[i]
      channel_coord[j] = k_coord[j]
    ```
    where `m_coord` and `window_offset`/`k_coord` come from delinearizing the
    M and K output indices using `output_sizes`.

    #### Examples

    Basic im2col (NHWC conv, 3x3 kernel):
    ```mlir
      %im2col = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0]
          output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%in : tensor<2x34x34x640xf32>)
          outs(%out : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
    ```

    With input padding (same padding absorbed from tensor.pad):
    ```mlir
      %im2col = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0]
          output_sizes = [[2], [34, 34], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          input_pad_low = [0, 1, 1, 0] input_pad_high = [0, 1, 1, 0]
          pad_value(%cst : f32)
          ins(%in : tensor<2x34x34x640xf32>)
          outs(%out : tensor<2x1156x5760xf32>) -> tensor<2x1156x5760xf32>
    ```

    With output-only padding (GEMM alignment, no input padding):
    ```mlir
      // M valid = 32*32 = 1024; output has 1040 (16 extra M positions filled
      // with pad_value for GEMM alignment). K valid = 3*3*640 = 5760.
      %im2col = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, 0, 0]
          output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          output_pad_low = [0, 0, 0] output_pad_high = [0, 16, 0]
          pad_value(%cst : f32)
          ins(%in : tensor<2x34x34x640xf32>)
          outs(%out : tensor<2x1040x5760xf32>) -> tensor<2x1040x5760xf32>
    ```
  }];

  let arguments = (ins AnyShaped:$input, AnyShaped:$output,
      DenseI64ArrayAttr:$strides, DenseI64ArrayAttr:$dilations,
      Variadic<Index>:$kernel_size, DenseI64ArrayAttr:$static_kernel_size,
      Variadic<Index>:$offsets, DenseI64ArrayAttr:$static_offsets,
      Variadic<Index>:$output_sizes, ArrayAttr:$static_output_sizes,
      DenseI64ArrayAttr:$batch_pos, DenseI64ArrayAttr:$m_pos,
      DenseI64ArrayAttr:$k_pos, DenseI64ArrayAttr:$input_k_perm,
      DenseI64ArrayAttr:$output_perm, Variadic<Index>:$input_pad_low,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_input_pad_low,
      Variadic<Index>:$input_pad_high,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_input_pad_high,
      Variadic<Index>:$output_pad_low,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_output_pad_low,
      Variadic<Index>:$output_pad_high,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$static_output_pad_high,
      Optional<AnyType>:$pad_value);

  let results = (outs Variadic<AnyShaped>:$results);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    attr-dict
    `strides` `=` $strides
    `dilations` `=` $dilations
    `kernel_size` `=`
    custom<DynamicIndexList>($kernel_size, $static_kernel_size)
    `offsets` `=`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `output_sizes` `=`
    custom<NestedDynamicIndexList>($output_sizes, $static_output_sizes)
    `batch_pos` `=` $batch_pos
    `m_pos` `=` $m_pos
    `k_pos` `=` $k_pos
    `input_k_perm` `=` $input_k_perm
    `output_perm` `=` $output_perm
    (`input_pad_low` `=`
     custom<DynamicIndexList>($input_pad_low, $static_input_pad_low)
     `input_pad_high` `=`
     custom<DynamicIndexList>($input_pad_high, $static_input_pad_high)^)?
    (`output_pad_low` `=`
     custom<DynamicIndexList>($output_pad_low, $static_output_pad_low)
     `output_pad_high` `=`
     custom<DynamicIndexList>($output_pad_high, $static_output_pad_high)^)?
    (`pad_value` `(` $pad_value^ `:` type($pad_value) `)`)?
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let builders = [OpBuilder<(ins "Value":$input, "Value":$output,
      "ArrayRef<int64_t>":$strides, "ArrayRef<int64_t>":$dilations,
      "ArrayRef<OpFoldResult>":$kernel_size, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<SmallVector<OpFoldResult>>":$output_sizes,
      "ArrayRef<int64_t>":$batch_dimensions, "ArrayRef<int64_t>":$m_dimensions,
      "ArrayRef<int64_t>":$k_dimensions, "ArrayRef<int64_t>":$input_k_perm,
      "ArrayRef<int64_t>":$output_perm,
      CArg<"ArrayRef<OpFoldResult>", "{}">:$input_pad_low,
      CArg<"ArrayRef<OpFoldResult>", "{}">:$input_pad_high,
      CArg<"ArrayRef<OpFoldResult>", "{}">:$output_pad_low,
      CArg<"ArrayRef<OpFoldResult>", "{}">:$output_pad_high,
      CArg<"Value", "Value()">:$pad_value)>];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration#[{
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }

    // Helpers to get output dimensions corresponding to batch, m, and k.
    SmallVector<int64_t> getBatchOutputDims();
    SmallVector<int64_t> getMOutputDims();
    SmallVector<int64_t> getKOutputDims();

    // Returns the number of M output dimensions.
    int64_t getNumMOutputDims();

    // Returns the mapping from input dimensions to the corresponding set of
    // output dimensions that can vectorize along with them. A pair of input and
    // output dims are vectorizable together if incrementing the output
    // dimension by 1 will also increment the input dimension by exactly 1.
    // Each input dim can have zero or more corresponding output dims. If an
    // input dim maps to multiple output dims, then the input dim can vectorize
    // with any one of the corresponding output dims.
    //
    // This method does not find vectorizable dims if output dims are partially
    // collapsed. For example, it computes K dim mapping for standard conv_2d if
    // the K dim is fully collapsed or fully expanded (including all input
    // channel and filter loop dims), but otherwise omits mappings for the K
    // dims.
    SmallVector<SmallVector<int64_t>> getInputToOutputDimVectorizationMap();

    // Return op metadata.
    SmallVector<OpFoldResult> getMixedKernelSize();
    SmallVector<OpFoldResult> getMixedOffsets();
    SmallVector<SmallVector<OpFoldResult>> getMixedOutputSizes();
    SmallVector<OpFoldResult> getMixedInputPadLow();
    SmallVector<OpFoldResult> getMixedInputPadHigh();
    SmallVector<OpFoldResult> getMixedOutputPadLow();
    SmallVector<OpFoldResult> getMixedOutputPadHigh();

    // Set op metadata.
    void setMixedOffsets(SmallVector<OpFoldResult> offsets);
    void setMixedOutputSizes(ArrayRef<SmallVector<OpFoldResult>> outputSizes);
    void setMixedInputPadLow(SmallVector<OpFoldResult> padLow);
    void setMixedInputPadHigh(SmallVector<OpFoldResult> padHigh);
    void setMixedOutputPadLow(SmallVector<OpFoldResult> padLow);
    void setMixedOutputPadHigh(SmallVector<OpFoldResult> padHigh);

    // Returns true if the op has any padding (input or output).
    bool hasPadding() { return static_cast<bool>(getPadValue()); }

    // Returns true if the op has non-zero output padding.
    bool hasOutputPadding();

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];
}

} // OpGroupNonStructuredOps


//===----------------------------------------------------------------------===//
// Winograd ops
//===----------------------------------------------------------------------===//

def OpGroupWinogradOps : OpDocGroup {
  let summary = "Winograd ops";
  let description = "";
}

let opDocGroup = OpGroupWinogradOps in {

def IREELinalgExt_WinogradInputTransformOp : IREELinalgExt_Op<"winograd.input_transform",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = [{Winograd Input Transform operator.}];
  let description = [{
    This operator is part of the first step in converting a convolution to
    its Winograd equivalent. Given a tile of an input image (I),
    this operator computes matmul(transpose(B), matmul(I, B)).
    The input tile is assumed to be square with each side of size m + r - 1,
    where the convolutional kernel is m x m and the output tile size is r x r.
    B is a constant 2-d square matrix of the same shape as the input tile I.
    The input to the operator is an image of shape (N, H, W, C) or (N, C, H, W)
    and the output is an operator of shape (m + r - 1, m + r - 1, N, H', W', C)
    where H' = ceil((H - m + 1)/r) and W' = ceil((W - m + 1)/r). The result
    of this operator is first collapsed and then fed to a batch matmul op.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       I64Attr:$output_tile_size,
                       I64Attr:$kernel_size,
                       DenseI64ArrayAttr:$image_dimensions
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `output_tile_size` `(` $output_tile_size `)`
    `kernel_size` `(` $kernel_size `)`
    `image_dimensions` `(` $image_dimensions `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getOriginalOperand() {
      return getInput();
    }
    Value getTransformedOperand() {
      return getOutput();
    }
    ShapedType getOriginalOperandType() {
      return getInputType();
    }
    ShapedType getTransformedOperandType() {
      return getOutputType();
    }
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }
    int64_t getInputTileSize() {
      return getOutputTileSize() + getKernelSize() - 1;
    }
    ArrayRef<int64_t> getHwDimensions() {
      return getImageDimensions();
    }
    std::array<int64_t, 2> nhwcImageDimensions() {
      return {1, 2};
    }
    std::array<int64_t, 2> nchwImageDimensions() {
      return {2, 3};
    }
    bool isNhwc() {
      std::array<int64_t, 2> nhwcImageDims = nhwcImageDimensions();
      return getImageDimensions() == ArrayRef<int64_t>(nhwcImageDims);
    }
    bool isNchw() {
      std::array<int64_t, 2> nchwImageDims = nchwImageDimensions();
      return getImageDimensions() == ArrayRef<int64_t>(nchwImageDims);
    }
    int getChannelDim() {
      return isNhwc() ? 3 : 1;
    }
    int64_t getIterationDomainRank() {
      return getOutputRank() - getImageDimensions().size();
    }
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];
}

def IREELinalgExt_WinogradFilterTransformOp : IREELinalgExt_Op<"winograd.filter_transform",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = [{Winograd Filter Transform operator.}];
  let description = [{
    This operator is part of the first step in converting a convolution to
    its Winograd equivalent. Given a tile of a convolution filter (F),
    this operator computes matmul(G, matmul(F, transpose(B))).
    The filter tile is assumed to be the full m x m convolutional kernel,
    and the result of the transformation on this tile is a square with each
    side of size m + r - 1, where the output tile size is r x r. G is a constant
    2-d matrix of shape (m + r - 1) x m. The input to the operator is a filter
    of shape (H, W, C, F) or (F, C, H, W) and the output is an operator of shape
    (m + r - 1, m + r - 1, C, F). The result of this operator is first collapsed
    and then fed to a batch matmul op.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       I64Attr:$output_tile_size,
                       I64Attr:$kernel_size,
                       DenseI64ArrayAttr:$kernel_dimensions
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `output_tile_size` `(` $output_tile_size `)`
    `kernel_size` `(` $kernel_size `)`
    `kernel_dimensions` `(` $kernel_dimensions `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    Value getOriginalOperand() {
      return getInput();
    }
    Value getTransformedOperand() {
      return getOutput();
    }
    ShapedType getOriginalOperandType() {
      return getInputType();
    }
    ShapedType getTransformedOperandType() {
      return getOutputType();
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }
    int64_t getInputTileSize() {
      return getOutputTileSize() + getKernelSize() - 1;
    }
    ArrayRef<int64_t> getHwDimensions() {
      return getKernelDimensions();
    }
    std::array<int64_t, 2> hwcfKernelDimensions() {
      return {0, 1};
    }
    std::array<int64_t, 2> fchwKernelDimensions() {
      return {2, 3};
    }
    bool isHwcf() {
      std::array<int64_t, 2> hwcfKernelDims = hwcfKernelDimensions();
      ArrayRef<int64_t> kernelDims = getKernelDimensions();
      return kernelDims == ArrayRef<int64_t>(hwcfKernelDims);
    }
    bool isFchw() {
      std::array<int64_t, 2> fchwKernelDims = fchwKernelDimensions();
      ArrayRef<int64_t> kernelDims = getKernelDimensions();
      return kernelDims == ArrayRef<int64_t>(fchwKernelDims);
    }
    int getChannelDim() {
      return isHwcf() ? 2 : 1;
    }
    int getFilterDim() {
      return isHwcf() ? 3 : 0;
    }
    int64_t getIterationDomainRank() {
      return getInputRank() - getKernelDimensions().size();
    }
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];
}

def IREELinalgExt_WinogradOutputTransformOp : IREELinalgExt_Op<"winograd.output_transform",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = [{Winograd Output Transform operator.}];
  let description = [{
    This operator is the last transform in converting a convolution to
    its Winograd equivalent. After convolution in the Winograd domain
    (which turns into an elementwise product for a single channel and
    batch matrix multiplication for many channels), this operator converts
    the output back into the original domain. Given a tile of the
    output (O) in the Winograd domain, this operator computes
    matmul(transpose(A), matmul(O, A)). The output tile is square with
    each side of size m + r - 1, where the convolutional kernel is m x m
    and the output tile size is r x r. A is a constant 2-d matrix of
    shape (m + r - 1) x r. The input to the operator is a tensor of
    shape (m + r - 1, m + r - 1, N, H', W', C) and the output is a
    tensor of shape (N, H, W, C) or (N, C, H, W) where H = r H' and W = r W'.
    This operator is followed by a tensor.extract_slice which extracts
    only the non-padded part of the output.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       I64Attr:$output_tile_size,
                       I64Attr:$kernel_size,
                       DenseI64ArrayAttr:$image_dimensions
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `output_tile_size` `(` $output_tile_size `)`
    `kernel_size` `(` $kernel_size `)`
    `image_dimensions` `(` $image_dimensions `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    Value getOriginalOperand() {
      return getOutput();
    }
    Value getTransformedOperand() {
      return getInput();
    }
    ShapedType getOriginalOperandType() {
      return getOutputType();
    }
    ShapedType getTransformedOperandType() {
      return getInputType();
    }
    ArrayRef<int64_t> getHwDimensions() {
      return getImageDimensions();
    }
    std::array<int64_t, 2> nhwcImageDimensions() {
      return {1, 2};
    }
    std::array<int64_t, 2> nchwImageDimensions() {
      return {2, 3};
    }
    bool isNhwc() {
      std::array<int64_t, 2> nhwcImageDims = nhwcImageDimensions();
      return getImageDimensions() == ArrayRef<int64_t>(nhwcImageDims);
    }
    bool isNchw() {
      std::array<int64_t, 2> nchwImageDims = nchwImageDimensions();
      return getImageDimensions() == ArrayRef<int64_t>(nchwImageDims);
    }
    int getChannelDim() {
      return isNhwc() ? 3 : 1;
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }
    int64_t getIterationDomainRank() {
      return getInputRank() - getImageDimensions().size();
    }
    int64_t getInputTileSize() {
      return getOutputTileSize() + getKernelSize() - 1;
    }
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];
}

} // OpGroupWinogradOps

//===---------------------------------------------------------------------===//
// Custom tilable op
//===---------------------------------------------------------------------===//

def IREELinalgExt_CustomOp : IREELinalgExt_Op<"custom_op", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<AggregatedOpInterface, [
    "decomposeOperation"]>,
  DeclareOpInterfaceMethods<LinalgFusionInterface>,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation"]>
  ]> {
  let summary = [{Custom operation for compiling with IREE.}];
  let description = [{
    This operation is meant to allow computation sequences that are fused at
    tile level prescriptively. This is to account for cases where such fusion
    cannot/is not yet discovered appropriately.

    The operation implements all the interfaces needed to be able to
    1. Compile e2e using IREE
    2. Still be able to fuse with other operations that the compiler can
       figure out automatically.

    Similar to how `LinalgOp`s represent a perfectly nested loop computation
    with
    - `indexing_maps` representing how the `ins`/`outs` are accessed
    - `region` representing the scalar computation performed
    - `iterator_types` representing the dependence along each iteration space
      dimension
    this operation represent a tiled computation with perfectly nested
    inter-tile loop nest.
    - `indexing_maps` represent what slices slices of the `ins`/`outs` are
      needed for each iteration of the tiled computation.
    - `region` represents the tiled computation performed using these slices
    - `iterator_types` represents the dependence between tiles along each
      iteration space.

    Some modifications required to handle the tile-level semantics are
    - Some dimensions of operands might not be accessed by dimensions of the
      inter-tile iteration space. This means that along these dimensions the
      slice size matches the dimension size. This access pattern of operands
      is captured in the respective indexing map using a `symbol` to represent
      that the entire dimension needs to be sliced.
    - The basic block arguments of the region represent the slice of the
      operand. These are either scalar types (if the corresponding operand is a
      scalar), or a `tensor` type with dynamic shapes (if the corresponding
      operand is a `tensor` type).

    For example, one could represent a prescriptively fused matmul computation
    as follows

    ```
    %0:2 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (d0, s0)>,
                         affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                         affine_map<(d0, d1)[s0, s1] -> (s1, d1)>,
                         affine_map<(d0, d1)[s0, s1] -> (d0, s1)>,
                         affine_map<(d0, d1)[s0, s1] -> (d0, d1)],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs1, %rhs1, %rhs2
            : tensor<1000000x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%outs1, %outs2 : tensor<1000000x?xf32>, tensor<1000000x?xf32>) {
      ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?x?xf32>,
           %t3 : tensor<?x?xf32>, %t4 : tensor<?x?xf32>) :
        %0 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %1 = linalg.matmul ins(%0, %t2 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%t4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        iree_linalg_ext.yield %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
    } -> tensor<1000000x?xf32>, tensor<x?xf32>
    ```
  }];
  let arguments = (ins
    Variadic<AnyRankedTensorOrScalarType>:$inputs,
    Variadic<AnyRankedTensor>:$outputs,
    AffineMapArrayAttr:$indexing_maps,
    IREELinalgExt_IteratorTypeArrayAttr:$iterator_types);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `indexing_maps` `=` $indexing_maps `,`
    `iterator_types` `=` $iterator_types `}`
    attr-dict-with-keyword
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration =[{
    // Helper accessor methods.
    unsigned getNumLoops();
    int64_t getRank(Value);

    // Return the number of non-loop dimensions of the op.
    unsigned getNumNonLoopDimensions();

    // Return the ranges for the loop dimensions and symbol
    // dimensions of the operation.
    SmallVector<Range> getIterationDomainForDimensions(OpBuilder &builder,
        ArrayRef<unsigned> dims, ArrayRef<unsigned> symbols);

    // DestinationStyleOpInterface methods
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

#endif  // IREE_DIALECT_LINALGEXT_OPS
