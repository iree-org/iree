// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_OPS
#define IREE_DIALECT_LINALGEXT_OPS

include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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
         LinalgExtInterface,
         SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::YieldOp">
  ])> {
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = [{
    SmallVector<Value> getDestinationOperands(OpBuilder &b) {
      SmallVector<Value> dest(getOutputs().begin(), getOutputs().end());
      return dest;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def IREELinalgExt_ScatterOp : IREELinalgExt_Op<"scatter",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
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
      DefaultValuedAttr<BoolAttr, "true">:$unique_indices
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict `unique_indices` `(` $unique_indices `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    int64_t getIndexDepth() {
      return getInputOperand(1)
          ->get()
          .getType()
          .cast<ShapedType>()
          .getShape()
          .back();
    }

    Value updates() {
      return getInputOperand(0)->get();
    }

    ShapedType getUpdateType() {
      return updates().getType().cast<ShapedType>();
    }

    Value indices() {
      return getInputOperand(1)->get();
    }

    ShapedType getIndicesType() {
      return indices().getType().cast<ShapedType>();
    }

    Value original() {
      return getOutputOperand(0)->get();
    }

    ShapedType getOriginalType() {
      return original().getType().cast<ShapedType>();
    }

    int64_t getUpdateSliceRank() {
      return updates().getType().cast<ShapedType>().getRank() - 1;
    }

    bool isScalarUpdate() {
      return getUpdateSliceRank() == 0;
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
    Value operand(int index) {
      return getOutputs()[index];
    }
    ShapedType getOperandType(int index) {
      return operand(index).getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType(0).getRank();
    }
    ArrayRef<int64_t> getOperandShape() {
      return getOperandType(0).getShape();
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
    bool hasCoeff() { return getNumInputs() > 1; }
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
      return getReal().getType().cast<ShapedType>();
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
    Value input() {
      return getInputOperand(0)->get();
    }
    Value accumulator() {
      return getOutputOperand(1)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];
}

def IREELinalgExt_ReverseOp : IREELinalgExt_Op<"reverse", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
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
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
    ArrayRef<int64_t> getOprerandShape() {
      return getOperandType().getShape();
    }
    SmallVector<int64_t> dims() {
      SmallVector<int64_t> ret;
      for (const APInt& elem : getDimensions()) {
        ret.push_back(elem.getLimitedValue());
      }
      return ret;
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
    Value values() {
      return getInputOperand(0)->get();
    }
    Optional<Value> indices() {
      if (getNumInputs() < 2) {
        return {};
      } else {
        return getInputOperand(1)->get();
      }
    }
    Value outputValues() {
      return getOutputOperand(0)->get();
    }
    Value outputIndices() {
      return getOutputOperand(1)->get();
    }
    ShapedType getInputType() {
      return values().getType().cast<ShapedType>();
    }
    int64_t getInputRank() {
      return getInputType().getRank();
    }
  }];
}

def IREELinalgExt_PackOp : IREELinalgExt_Op<"pack", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<LinalgExtInterface>,
  DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "generateScalarImplementation"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]>{
  let summary = "pack operation";
  let description = [{
    The pack operation converts an `input` into a tiled and packed layout. The
    dimensions to be tiled are obtained from `dims_pos` and the size of the tile is
    obtained from `inner_tiles`. The dimensions listed in `dims_pos` do not need to
    be contiguous in which case the tile will get transposed.  We handle only full
    tiles if `padding_value` is not set; it is UB if the tile does not perfectly
    divide the dimension. If `padding_value` is set, it will pad along high
    dimensions, i.e., it pads at the bottom and on the right if the input has
    rank 2, and the result type shape, will be dynamic in any dimension if and
    only if the input shape is.

    Example KC_to_KCck:

    ```mlir
    iree_linalg_ext.pack %arg0 dims_pos = [1, 0]
      inner_tiles = [32, 8] into %arg1 : (memref<128x256xf32> memref<16x8x32x8xf32>)
    ```

    Example NC_to_NCnc:

    ```mlir
    iree_linalg_ext.pack %arg0 dims_pos = [0, 1]
      inner_tiles = [8, 32] into %arg1 : (memref<128x256xf32> memref<16x8x8x32xf32>)
    ```

    In both cases, dimension at position 0 in the input memref (128) is tiled
    with a factor of 8, while dimension at position 1 (256) is tiled with a factor
    of 32. In the KC_to_KCck example, the point loops are interchanged.

    Example NC_to_NCnc with padding:

    ```mlir
    iree_linalg_ext.pack %arg padding_value(%pad : f32) dims_pos = [0, 1]
      inner_tiles = [8, 2] into %arg1 : (memref<13x15xf32> memref<2x8x8x2xf32>)
    ```

  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
    Variadic<AnyShaped>:$outputs,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$dims_pos,
    Variadic<Index>:$inner_tiles,
    I64ArrayAttr:$static_inner_tiles,
    Optional<AnyType>:$padding_value);

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict
    $inputs
    (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
    `dims_pos` `=` $dims_pos
    `inner_tiles` `=`
    custom<DynamicIndexList>($inner_tiles, $static_inner_tiles,
                             "ShapedType::kDynamicSize")
    `into` $outputs `:` `(` type($inputs) type($outputs) `)`
     (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    // Return the output operand.
    Value getOutput() {
      return getOutputOperand(0)->get();
    }

    // Return the input operand.
    Value getInput() {
      return getInputOperand(0)->get();
    }

    // Return the output rank.
    int64_t getOutputRank() {
      return  getOutputType().getRank();
    }

    // Return the output type.
    ShapedType getOutputType() {
      return getOutput().getType();
    }

    // Return the input type.
    ShapedType getInputType() {
      return getInput().getType();
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

    // Infer the output type.
    ShapedType inferResultType();

    // Return a mapping from positions `dims_pos` to their tile factors.
    DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();
  }];
}

//===----------------------------------------------------------------------===//
// Pure ops
//===----------------------------------------------------------------------===//

def IREELinalgExt_YieldOp : IREELinalgExt_PureOp<"yield", [NoSideEffect, ReturnLike, Terminator]> {
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

#endif  // IREE_DIALECT_LINALGEXT_OPS
