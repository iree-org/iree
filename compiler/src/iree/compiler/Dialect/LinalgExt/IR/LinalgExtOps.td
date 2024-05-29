// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_OPS
#define IREE_DIALECT_LINALGEXT_OPS

include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
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

class IREELinalgExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREELinalgExt_Dialect, mnemonic, traits> {
}

class IREELinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    IREELinalgExt_PureOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DestinationStyleOpInterface, LinalgExtInterface,
         SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::YieldOp">
  ])> {
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = "";
}

//===----------------------------------------------------------------------===//
// Utility ops
//===----------------------------------------------------------------------===//

def OpGroupUtilityOps : OpDocGroup {
  let summary = "Utility ops";
  let description = "";
}

let opDocGroup = OpGroupUtilityOps in {

def IREELinalgExt_YieldOp : IREELinalgExt_PureOp<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `iree_linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `iree_linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

} // OpGroupUtilityOps

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def OpGroupNonStructuredOps : OpDocGroup {
  let summary = "Non-structured ops";
  let description = "";
}

let opDocGroup = OpGroupNonStructuredOps in {

def IREELinalgExt_ScatterOp : IREELinalgExt_Op<"scatter",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<LinalgFusionInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateScalarImplementation",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation"]>]> {
  let summary = "Scatter operator";
  let description = [{
    Based on XLA operation semantics, takes two `inputs` (`update` and
    `indices`) and `outputs` value (`original`). The operation updates
    the value at the slices specified by `indices` by combining the
    current value with the value in `updates` using the computation
    specified in `region`. The `region` specifies a binary operation
    of signature (T, T) -> T, where `T` is the element-type of
    `updates` (and `original`). The first argument correspond the
    value to be updated (i.e. from `updates`), and the second the
    current value (i.e. value from `original`).

    The `indices` is a 2D tensor/memref type. The first dim is the number of
    updates, and the second dim is index depth. The index depth should always be
    static.

    The first dim of `updates` and `indices` is identical, since they represent
    the number of updates.

    The rank of the `original`/`result` is at least
    `index_depth + rank(%updates) - 1`. The first `index_depth` indices are
    derived from `indices` and the shape of update value has the last
    rank(%original) - index_depth values match %(originals) last dimensions,
    with the previous dims extending from the index offsets.

    The dimension_map attributes describes which index value maps to which
    dimension in the destionation. It cannot contain duplicate values, must
    have as many entries as index depth, and values must be within the rank of
    the destination.

    The unique_indices attribute carries the information whether all the indices
    are unique. If there are repeated indices, the first iteration loop will be
    marked as reduction.

    The shapes definition follows tensorflow operations execept that it force
    batch dims to be 1D. See more information in
      https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update
  }];
  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs,
      DenseI64ArrayAttr:$dimension_map,
      DefaultValuedAttr<BoolAttr, "true">:$unique_indices
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict `dimension_map` `=` $dimension_map
              `unique_indices` `(` $unique_indices `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    int64_t getIndexDepth() {
      return cast<ShapedType>(getDpsInputOperand(1)->get().getType())
          .getShape()
          .back();
    }

    Value getUpdates() {
      return getDpsInputOperand(0)->get();
    }

    ShapedType getUpdateType() {
      return cast<ShapedType>(getUpdates().getType());
    }

    Value getIndices() {
      return getDpsInputOperand(1)->get();
    }

    ShapedType getIndicesType() {
      return cast<ShapedType>(getIndices().getType());
    }

    Value getOriginal() {
      return getDpsInitOperand(0)->get();
    }

    ShapedType getOriginalType() {
      return cast<ShapedType>(getOriginal().getType());
    }

    int64_t getUpdateSliceRank() {
      return cast<ShapedType>(getUpdates().getType()).getRank() - 1;
    }

    bool isScalarUpdate() {
      return getUpdateSliceRank() == 0;
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_SortOp : IREELinalgExt_Op<"sort",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateScalarImplementation",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition",
         "getTiledImplementation"]>]> {
  let summary = "Sort operator";
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
}

def IREELinalgExt_FftOp : IREELinalgExt_Op<"fft", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
      ["generateScalarImplementation",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Fft operator";
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

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    (`:` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getStage() { return getInputs()[0]; }
    Value getReal() { return getOutputs()[0]; }
    Value getImag() { return getOutputs()[1]; }
    bool hasCoeff() { return getNumDpsInputs() > 1; }
    void generateScalarImplWithoutCoeffBuf(
        OpBuilder & b, Location loc, ArrayRef<Value> operands, Value wholeSize);
    void generateScalarImplWithCoeffBuf(OpBuilder & b, Location loc,
                                        ArrayRef<Value> operands);
    Value getRealCoeff() {
      if (!hasCoeff()) return Value();
      return getInputs()[1];
    }
    Value getImagCoeff() {
      if (!hasCoeff()) return Value();
      return getInputs()[2];
    }
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
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_ScanOp : IREELinalgExt_Op<"scan",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["generateScalarImplementation",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Scan operator";
  let description = [{
    Computes the inclusive/exclusive scan along a given dimension.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension,
                       BoolAttr:$inclusive
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "0">:$dimension, CArg<"bool", "true">:$inclusive)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }
    Value getAccumulator() {
      return getDpsInitOperand(1)->get();
    }
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }
    ShapedType getOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_ReverseOp : IREELinalgExt_Op<"reverse", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<LinalgFusionInterface>,
  DeclareOpInterfaceMethods<
      TilingInterface,
      ["generateScalarImplementation",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>]> {
  let summary = "Reverse operator";
  let description = [{
    A temporary solution for lowering reverse ops into IREE, allowing IREE to
    tile and distribute them.
    }
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64ElementsAttr:$dimensions
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict `dimensions` `(` $dimensions `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }
    ShapedType getOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
    ArrayRef<int64_t> getOprerandShape() {
      return getOperandType().getShape();
    }
    SmallVector<int64_t> getDimensionsArray() {
      SmallVector<int64_t> ret;
      for (const APInt& elem : getDimensions()) {
        ret.push_back(elem.getLimitedValue());
      }
      return ret;
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_TopkOp : IREELinalgExt_Op<"topk",[
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["generateScalarImplementation",
     "getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation"]>
]>{
  let summary = "Top-K operator";
  let description = [{
   A Top-K operation for N-D tensors. Reduces the target dimension from the input
   size N down to K elements based on the supplied binary region.

   Accepts an N-D tensor input consisting of values and an optioanl N-D tensor
   for indices of those values (i32 type). If input indices aren't provided, the
   index mapping is inferred based on the k dim.  Both input values/indices
   tensors and output values/indicies tensors must have the same shape. Top-K is
   computed along the target dimension (from dimension()). Returns two output
   tensors of values and the indicies of Top-K results. The output dimensions
   must match the input save for the dimension that is reduced to K results.

   Region accepts lhs=[next N input] and rhs=[exiting K output] and yeilds an
   i1. If true, the two values are swapped:
     - For Top-K compoarision: >
     - For Min-K comparision: <
   Note: when the two values are equal, the first occurence is always selected.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getValues() {
      return getDpsInputOperand(0)->get();
    }
    std::optional<Value> getIndices() {
      if (getNumDpsInputs() < 2) {
        return {};
      } else {
        return getDpsInputOperand(1)->get();
      }
    }
    Value outputValues() {
      return getDpsInitOperand(0)->get();
    }
    Value outputIndices() {
      return getDpsInitOperand(1)->get();
    }
    ShapedType getInputType() {
      return cast<ShapedType>(getValues().getType());
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Attention
//===----------------------------------------------------------------------===//

def IREELinalgExt_AttentionOp : IREELinalgExt_Op<"attention",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Attention operator";
  let description = [{
    Computes the scaled dot product attention function:

    attention(Q, K, V, scale) = softmax(Q @ K.T * scale) @ V

    Here Q, K, V are given tensors and scale is a scalar value specifying
    the scale to use.

    For self-attention, all inputs and the result have the same shape BxNxd
    where B is the batch dimension, N is the sequence length and d is head
    dimension. Typically N >>> d. Usually, this operator also performs
    masking and dropout, but we leave that out of the current implementation.
    For cross-attention, the query and output have the same shape (BxNxd),
    while the key and value differ in sequence length (they have shape BxLxd,
    where L != N).

    This operator after tiling results in a tiled result as per
    FlashAttention 2 and optionally results in the current `max` and `sum`
    statistics while processing the current tile.

    If transpose_v is speciifed, the V tensor passed as input is assumed to
    be transposed:

    attention(Q, K, V, scale) = softmax(Q @ K.T * scale) @ V.T

    TODO: We should be moving to using a indexing map like approach so we
    can generalize which tensor is transposed and which is not.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$transpose_v
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs)>,
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getQuery() {
      return getDpsInputOperand(0)->get();
    }
    Value getKey() {
      return getDpsInputOperand(1)->get();
    }
    Value getValue() {
      return getDpsInputOperand(2)->get();
    }
    Value getScale() {
      return getDpsInputOperand(3)->get();
    }
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }
    std::optional<Value> getMax() {
      if (getNumResults() < 2)
        return std::nullopt;
      return getDpsInitOperand(1)->get();
    }
    std::optional<Value> getSum() {
      if (getNumResults() < 3)
        return std::nullopt;
      return getDpsInitOperand(2)->get();
    }

    ShapedType getQueryType() {
      return cast<ShapedType>(getQuery().getType());
    }
    ShapedType getKeyType() {
      return cast<ShapedType>(getKey().getType());
    }
    ShapedType getValueType() {
      return cast<ShapedType>(getValue().getType());
    }
    FloatType getScaleType() {
      return cast<FloatType>(cast<ShapedType>(getScale().getType()));
    }
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }
    std::optional<ShapedType> getMaxType() {
      std::optional<Value> maxVal = getMax();
      if (!maxVal) return std::nullopt;
      return cast<ShapedType>(maxVal->getType());
    }
    std::optional<ShapedType> getSumType() {
      std::optional<Value> sumVal = getSum();
      if (!sumVal) return std::nullopt;
      return cast<ShapedType>(sumVal->getType());
    }

    int64_t getQueryRank() {
      return getQueryType().getRank();
    }
    int64_t getKeyRank() {
      return getKeyType().getRank();
    }
    int64_t getValueRank() {
      return getValueType().getRank();
    }
    int64_t getOutputRank() {
      return getOutputType().getRank();
    }
    std::optional<int64_t> getMaxRank() {
      std::optional<ShapedType> maxType = getMaxType();
      if (!maxType) return std::nullopt;
      return maxType->getRank();
    }
    std::optional<int64_t> getSumRank() {
      std::optional<ShapedType> sumType = getSumType();
      if (!sumType) return std::nullopt;
      return sumType->getRank();
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }

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
    AffineMap getOutputMap() {
      return cast<AffineMap>(getIndexingMapsArray()[3]);
    }
    std::optional<AffineMap> getMaxMap() {
      if (getNumResults() < 2)
        return std::nullopt;
      return cast<AffineMap>(getIndexingMapsArray()[4]);
    }
    std::optional<AffineMap> getSumMap() {
      if (getNumResults() < 3)
        return std::nullopt;
      return cast<AffineMap>(getIndexingMapsArray()[5]);
    }

    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }
  }];
}

//===----------------------------------------------------------------------===//
// OnlineAttention
//===----------------------------------------------------------------------===//

def IREELinalgExt_OnlineAttentionOp : IREELinalgExt_PureOp<"online_attention",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DestinationStyleOpInterface, LinalgExtInterface,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Online Attention operator";
  let description = [{}];

  let arguments = (ins AnyShaped:$query,
                       AnyShaped:$key,
                       AnyShaped:$value,
                       AnyFloat:$scale,
                       AnyShaped:$output,
                       AnyShaped:$max,
                       AnyShaped:$sum,
                       AffineMapArrayAttr:$indexing_maps
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $query `,` $key `,` $value `,` $scale `:` type($query) `,` type($key) `,` type($value) `,` type($scale) `)`
    `outs` `(` $output `,` $max `,` $sum `:` type($output) `,` type($max) `,` type($sum) `)`
    (`->` type($results)^)?
  }];

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
    AffineMap getOutputMap() {
      return getIndexingMapsArray()[3];
    }
    AffineMap getMaxMap() {
      return getIndexingMapsArray()[4];
    }
    AffineMap getSumMap() {
      return getIndexingMapsArray()[5];
    }

    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }
  }];
}

} // OpGroupNonStructuredOps

//===----------------------------------------------------------------------===//
// Data tiling ops
//===----------------------------------------------------------------------===//

def OpGroupDataTilingOps : OpDocGroup {
  let summary = "Data tiling ops";
  let description = [{
    Operations for working with data layouts, padding, encodings, and other
    properties useful for tiling computations across iteration space dimensions.
  }];
}

let opDocGroup = OpGroupDataTilingOps in {

def IREELinalgExt_PackOp : IREELinalgExt_Op<"pack", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "generateScalarImplementation"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]>{
  let summary = "pack operation";
  let description = [{
    The pack operation converts an `input` into a tiled and packed layout. The
    dimensions to be tiled are obtained from `inner_dims_pos` and the size of the
    tile is obtained from `inner_tiles`. The dimensions listed in `inner_dims_pos`
    do not need to be contiguous in which case the tile will get transposed.  We
    handle only full tiles if `padding_value` is not set; it is UB if the tile does
    not perfectly divide the dimension. If `padding_value` is set, it will pad
    along high dimensions, i.e., it pads at the bottom and on the right if the
    input has rank 2, and the result type shape, will be dynamic in any dimension
    if and only if the input shape is. As optional input, the operation takes
    `outer_dims_perm` that allows to permute the tiled loops.

    Example KC_to_KCck:

    ```mlir
    iree_linalg_ext.pack %arg0 inner_dims_pos = [1, 0]
      inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<16x8x32x8xf32>)
    ```

    Example NC_to_NCnc:

    ```mlir
    iree_linalg_ext.pack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [8, 32] into %arg1 : (memref<128x256xf32> memref<16x8x8x32xf32>)
    ```
    Example KC_to_CKkc

    ```mlir
    iree_linalg_ext.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<32x4x32x8xf32>)
    ```

    In all cases, dimension at position 0 in the input memref (128) is tiled
    with a factor of 8, while dimension at position 1 (256) is tiled with a factor
    of 32. In the KC_to_KCck example, the point loops are interchanged, while in the
    KC_to_CKkc example the tiled loops.

    Example NC_to_NCnc with padding:

    ```mlir
    iree_linalg_ext.pack %arg padding_value(%pad : f32) inner_dims_pos = [0, 1]
      inner_tiles = [8, 2] into %arg1 : (memref<13x15xf32> memref<2x8x8x2xf32>)
    ```

  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
    Variadic<AnyShaped>:$outputs,
    DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$outer_dims_perm,
    DenseI64ArrayAttr:$inner_dims_pos,
    Variadic<Index>:$inner_tiles,
    DenseI64ArrayAttr:$static_inner_tiles,
    Optional<AnyType>:$padding_value);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    $inputs
    (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
    (`outer_dims_perm` `=` $outer_dims_perm^)?
    `inner_dims_pos` `=` $inner_dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)
    `into` $outputs `:` `(` type($inputs) type($outputs) `)`
     (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$output,
      "ArrayRef<int64_t>":$innerDimsPos,
      "ArrayRef<OpFoldResult>":$innerTiles,
      CArg<"std::optional<Value>", "std::nullopt">:$paddingValue,
      CArg<"ArrayRef<int64_t>", "{}">:$outerDimsPerm)>
  ];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    // Return the output operand.
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }

    // Return the input operand.
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }

    // Return the output rank.
    int64_t getOutputRank() {
      return  getOutputType().getRank();
    }

    // Return the output type.
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }

    // Return the input type.
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }

    // Return the output shape.
    ArrayRef<int64_t> getOutputShape() {
      return getOutputType().getShape();
    }

    // Return the input shape.
    ArrayRef<int64_t> getInputShape() {
      return getInputType().getShape();
    }

    // Return the element type.
    Type getElementType() {
      return getInputType().getElementType();
    }

    // Return the rank of the input operand.
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Return the tile sizes.
    SmallVector<OpFoldResult> getMixedTiles();
    SmallVector<int64_t> getStaticTiles();

    // Return a mapping from positions `dims_pos` to their tile factors.
    DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();

    // Method to get the shape of the result as `SmallVector<OpFoldResult>`.
    // This is a static method to allow getting the shape of the destination
    // expected while creating a `pack` op.
    static SmallVector<OpFoldResult> getResultShape(OpBuilder &builder,
        Location loc, ArrayRef<OpFoldResult> sourceDims,
        ArrayRef<OpFoldResult> innerTileDims, ArrayRef<int64_t> innerDimsPos,
        ArrayRef<int64_t> outerDimsPerm = {});
    // Method to return the shape of the result as `SmallVector<OpFoldResult>`.
    SmallVector<OpFoldResult> getResultShape(OpBuilder &builder);

    // Method to get the `ShapedType` of the result. This is a static method
    // to allow getting the type of the destination while creating the `pack`
    // op.
    static ShapedType getPackedType(ShapedType sourceType,
        ArrayRef<int64_t> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
        ArrayRef<int64_t> outerDimsPerm = {});

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_UnPackOp : IREELinalgExt_Op<"unpack", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "generateScalarImplementation"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]>{
  let summary = "unpack operation";

  let description = [{
    The unpack operation converts a tiled and packed input to an unpacked
    output. See `pack` for more details on `inner_tiles` and `dims_pos`; it is UB
    if the tile does not perfectly divide the dimension. Optionally, the operation
    also supports permuting the tiled loops.

    Example KCck_to_KC:

    ```mlir
    iree_linalg_ext.unpack %arg0 dims_pos = [1, 0]
      inner_tiles = [32, 8] into %arg1 : (memref<16x8x32x8xf32> memref<128x256xf32>)
    ```

    Example NCnc_to_NC:

    ```mlir
    iree_linalg_ext.unpack %arg0 dims_pos = [0, 1]
      inner_tiles = [8, 32] into %arg1 : (memref<16x8x8x32xf32> memref<128x256xf32>)
    ```

    Example CKkc_to_KC:

    ```mlir
    iree_linalg_ext.unpack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [32, 8] into %arg0 : (memref<32x4x32x8xf32> memref<128x256xf32>)
    ```
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
    Variadic<AnyShaped>:$outputs,
    DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$outer_dims_perm,
    DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$inner_dims_pos,
    Variadic<Index>:$inner_tiles,
    DenseI64ArrayAttr:$static_inner_tiles);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    $inputs
    (`outer_dims_perm` `=` $outer_dims_perm^)?
    `inner_dims_pos` `=` $inner_dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles)
    `into` $outputs `:` `(` type($inputs) type($outputs) `)`
     (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$output,
    "ArrayRef<int64_t>":$innerDimsPos,
    "ArrayRef<OpFoldResult>":$innerTiles,
    CArg<"ArrayRef<int64_t>", "{}">:$outerDimsPerm)>
  ];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    // Return the output operand.
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }

    // Return the input operand.
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }

    // Return the output rank.
    int64_t getOutputRank() {
      return  getOutputType().getRank();
    }

    // Return the output type.
    ShapedType getOutputType() {
      return cast<ShapedType>(getOutput().getType());
    }

    // Return the input type.
    ShapedType getInputType() {
      return cast<ShapedType>(getInput().getType());
    }

    // Return the output shape.
    ArrayRef<int64_t> getOutputShape() {
      return getOutputType().getShape();
    }

    // Return the input shape.
    ArrayRef<int64_t> getInputShape() {
      return getInputType().getShape();
    }

    // Return the rank of the input operand.
    int64_t getInputRank() {
      return getInputType().getRank();
    }

    // Return the tile sizes.
    SmallVector<OpFoldResult> getMixedTiles();
    SmallVector<int64_t> getStaticTiles();

    // Return a mapping from positions `dims_pos` to their tile factors.
    DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

} // OpGroupDataTilingOps

//===----------------------------------------------------------------------===//
// Winograd ops
//===----------------------------------------------------------------------===//

def OpGroupWinogradOps : OpDocGroup {
  let summary = "Winograd ops";
  let description = "";
}

let opDocGroup = OpGroupWinogradOps in {

def IREELinalgExt_WinogradInputTransformOp : IREELinalgExt_Op<"winograd.input_transform",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd Input Transform operator";
  let description = [{
    This operator is part of the first step in converting a convolution to
    its Winograd equivalent. Given a tile of an input image (I),
    this operator computes matmul(tranpose(B), matmul(I, B)).
    The input tile is assumed to be square with each side of size m + r - 1,
    where the convolutional kernel is m x m and the output tile size is r x r.
    B is a constant 2-d square matrix of the same shape as the input tile I.
    The input to the operator is an image of shape (N, H, W, C) or (N, C, H, W)
    and the output is an operator of shape (m + r - 1, m + r - 1, N, H', W', C)
    where H' = ceil((H - m + 1)/r) and W' = ceil((W - m + 1)/r). The result
    of this operator is first collapsed and then fed to a batch matmul op.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$output_tile_size,
                       I64Attr:$kernel_size,
                       DenseI64ArrayAttr:$image_dimensions
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
      CArg<"ArrayRef<int64_t>", "{1, 2}">:$image_dimensions)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `output_tile_size` `(` $output_tile_size `)`
    `kernel_size` `(` $kernel_size `)`
    `image_dimensions` `(` $image_dimensions `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }
    Value getOutput() {
      return getDpsInitOperand(0)->get();
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
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_WinogradFilterTransformOp : IREELinalgExt_Op<"winograd.filter_transform",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd Filter Transform operator";
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

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$output_tile_size,
                       I64Attr:$kernel_size,
                       DenseI64ArrayAttr:$kernel_dimensions
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
      CArg<"ArrayRef<int64_t>", "{0, 1}">:$kernel_dimensions)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `output_tile_size` `(` $output_tile_size `)`
    `kernel_size` `(` $kernel_size `)`
    `kernel_dimensions` `(` $kernel_dimensions `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }
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
      return getOutputsMutable();
    }
  }];
}

def IREELinalgExt_WinogradOutputTransformOp : IREELinalgExt_Op<"winograd.output_transform",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd Output Transform operator";
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

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$output_tile_size,
                       I64Attr:$kernel_size,
                       DenseI64ArrayAttr:$image_dimensions
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
      CArg<"ArrayRef<int64_t>", "{1, 2}">:$image_dimensions)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `output_tile_size` `(` $output_tile_size `)`
    `kernel_size` `(` $kernel_size `)`
    `image_dimensions` `(` $image_dimensions `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getInput() {
      return getDpsInputOperand(0)->get();
    }
    Value getOutput() {
      return getDpsInitOperand(0)->get();
    }
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
      return getOutputsMutable();
    }
  }];
}

} // OpGroupWinogradOps

#endif  // IREE_DIALECT_LINALGEXT_OPS
