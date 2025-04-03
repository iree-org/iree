// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_INJECTDISPATCHTRACINGPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

struct TensorValue {
  Value value;
  SmallVector<Value> dynamicDims;
};

/// Filters out a list of all `Value`s in range that are tensor types, and
/// groups them with their corresponding dynamic dimensions from `dynamicDims`.
/// The `dynamicDims` range is expected to have all dynamic dims of the values
/// in `range`, in the order that they appear in the tensor shapes.
static SmallVector<TensorValue> filterTensorValues(ValueRange &&range,
                                                   ValueRange &&dynamicDims) {
  SmallVector<TensorValue> result;
  for (auto [idx, value] : llvm::enumerate(range)) {
    if (llvm::isa<TensorType>(value.getType())) {
      SmallVector<Value> dims =
          IREE::Util::findDynamicDimsInList(idx, range, dynamicDims);
      result.push_back({value, dims});
    }
  }
  return result;
}

/// Sets all `Value`s of the `TensorValue`s in `tensorValues` to the row major
/// layout by inserting flow.tensor.encode ops before any Value that has an
/// encoding. Populates `decodedIndices` with the indices of `tensorValues` that
/// were decoded.
static SmallVector<Value>
getInRowMajorLayout(OpBuilder &builder, SmallVector<TensorValue> tensorValues,
                    SmallVector<int64_t> &decodedIndices) {
  SmallVector<Value> rowMajorTensors;
  for (auto [idx, v] : llvm::enumerate(tensorValues)) {
    auto rankedTensorType = dyn_cast<RankedTensorType>(v.value.getType());
    if (!rankedTensorType || !rankedTensorType.getEncoding()) {
      rowMajorTensors.push_back(v.value);
      continue;
    }
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfterValue(v.value);
    Value rowMajorTensor = builder.create<IREE::Flow::TensorEncodeOp>(
        v.value.getLoc(), rankedTensorType.dropEncoding(), v.value,
        /*operand_dims=*/v.dynamicDims, /*result_dims=*/v.dynamicDims);
    rowMajorTensors.push_back(rowMajorTensor);
    decodedIndices.push_back(idx);
  }
  return rowMajorTensors;
}

namespace {

struct InjectDispatchTracingPass
    : public IREE::Flow::impl::InjectDispatchTracingPassBase<
          InjectDispatchTracingPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto appendDecodedValuesToLabel = [](std::string str,
                                         SmallVector<int64_t> decodedIndices) {
      llvm::raw_string_ostream os(str);
      if (!decodedIndices.empty()) {
        os << " with {";
        llvm::interleaveComma(decodedIndices, os);
        os << "} decoded to row major layout";
      }
      return os.str();
    };
    for (auto dispatchOp : funcOp.getFunctionBody().getOps<DispatchOp>()) {
      std::string entryPointName = dispatchOp.getEntryPointName();

      // Input tensors:
      OpBuilder builder(dispatchOp);
      SmallVector<TensorValue> inputTensorValues = filterTensorValues(
          dispatchOp.getArguments(), dispatchOp.getArgumentDims());
      SmallVector<int64_t> decodedInputIndices;
      SmallVector<Value> decodedInputValues =
          getInRowMajorLayout(builder, inputTensorValues, decodedInputIndices);
      std::string inputsLabelStr = appendDecodedValuesToLabel(
          entryPointName + " inputs", decodedInputIndices);
      StringAttr inputsLabel = builder.getStringAttr(inputsLabelStr);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(), inputsLabel, decodedInputValues);

      // Output tensors:
      SmallVector<TensorValue> resultTensorValues = filterTensorValues(
          dispatchOp.getResults(), dispatchOp.getResultDims());
      SmallVector<int64_t> decodedOutputIndices;
      SmallVector<Value> decodedResultValues = getInRowMajorLayout(
          builder, resultTensorValues, decodedOutputIndices);
      std::string outputsLabelStr = appendDecodedValuesToLabel(
          entryPointName + " outputs", decodedOutputIndices);
      StringAttr outputsLabel = builder.getStringAttr(outputsLabelStr);

      // Set insertion point to the last decoded value before creating the
      // trace op.
      Operation *lastResult = decodedResultValues.front().getDefiningOp();
      DominanceInfo domInfo(funcOp);
      for (Value v : decodedResultValues) {
        if (domInfo.dominates(lastResult, v.getDefiningOp())) {
          lastResult = v.getDefiningOp();
        }
      }
      builder.setInsertionPointAfter(lastResult);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(), outputsLabel, decodedResultValues);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
