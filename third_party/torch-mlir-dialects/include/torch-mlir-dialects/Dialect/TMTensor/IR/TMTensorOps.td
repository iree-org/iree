//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_OPS
#define TORCH_MLIR_DIALECT_TMTENSOR_OPS

include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorBase.td"
include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorInterfaces.td"
include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class TMTensor_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<TMTensor_Dialect, mnemonic, traits> {
}

class TMTensor_Op<string mnemonic, list<Trait> traits = []> :
    TMTensor_PureOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         TMTensorInterface,
         SingleBlockImplicitTerminator<"::mlir::torch::TMTensor::YieldOp">
  ])> {
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraTMTensorOpClassDeclaration = [{
    SmallVector<Value> getDestinationOperands(OpBuilder &b) {
      SmallVector<Value> dest(getOutputs().begin(), getOutputs().end());
      return dest;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def TMTensor_ScanOp : TMTensor_Op<"scan",
    [DeclareOpInterfaceMethods<ScalarLoopOpInterface,
        ["generateScalarImplementation"]>]> {
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
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraTMTensorOpClassDeclaration # [{
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

def TMTensor_ScatterOp : TMTensor_Op<"scatter",
    [DeclareOpInterfaceMethods<ScalarLoopOpInterface,
        ["generateScalarImplementation"]>]> {
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
  let extraClassDeclaration = extraTMTensorOpClassDeclaration # [{

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

def TMTensor_SortOp : TMTensor_Op<"sort",
    [DeclareOpInterfaceMethods<ScalarLoopOpInterface,
        ["generateScalarImplementation"]>]> {
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
  let extraClassDeclaration = extraTMTensorOpClassDeclaration # [{
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

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      std::pair<unsigned, unsigned> outputsIndexAndLength =
        getODSOperandIndexAndLength(1);
      return std::make_pair<int64_t, int64_t>(
          outputsIndexAndLength.first,
          outputsIndexAndLength.first + outputsIndexAndLength.second);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Pure ops
//===----------------------------------------------------------------------===//

def TMTensor_YieldOp : TMTensor_PureOp<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "TMTensor yield op";
  let description = [{
    `tm_tensor.yield` is a special terminator operation for blocks inside
    regions in `tm_tensor` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif  // TORCH_MLIR_DIALECT_TMTENSOR_OPS
