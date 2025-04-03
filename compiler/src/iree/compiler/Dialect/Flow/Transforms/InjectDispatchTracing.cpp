// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_INJECTDISPATCHTRACINGPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

static SmallVector<Value> filterTensorValues(ValueRange &&range) {
  SmallVector<Value> result;
  for (auto value : range) {
    if (llvm::isa<TensorType>(value.getType()))
      result.push_back(value);
  }
  return result;
}

/// Sets all Values in `values` to the row major layout by inserting
/// flow.tensor.encode ops before any Value that has an encoding. Returns a
/// list of booleans that are true for each Value that was decoded, and false
/// otherwise.
static SmallVector<int64_t> setToRowMajorLayout(OpBuilder &builder,
                                                SmallVector<Value> &values) {
  SmallVector<Value> rowMajorTensors;
  SmallVector<int64_t> decodedIndices;
  for (auto [idx, v] : llvm::enumerate(values)) {
    auto rankedTensorType = dyn_cast<RankedTensorType>(v.getType());
    if (!rankedTensorType || !rankedTensorType.getEncoding()) {
      rowMajorTensors.push_back(v);
      continue;
    }
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfterValue(v);
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(builder, v.getLoc(), v);
    SmallVector<Value> dynamicDimSizes;
    std::tie(std::ignore, dynamicDimSizes) = decomposeMixedValues(mixedSizes);
    Value rowMajorTensor = builder.create<IREE::Flow::TensorEncodeOp>(
        v.getLoc(), rankedTensorType.dropEncoding(), v,
        /*operand_dims=*/dynamicDimSizes, /*result_dims=*/dynamicDimSizes);
    rowMajorTensors.push_back(rowMajorTensor);
    decodedIndices.push_back(idx);
  }
  values = rowMajorTensors;
  return decodedIndices;
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
      SmallVector<Value> inputValues =
          filterTensorValues(dispatchOp.getArguments());
      SmallVector<int64_t> decodedInputIndices =
          setToRowMajorLayout(builder, inputValues);
      std::string inputsLabelStr = appendDecodedValuesToLabel(
          entryPointName + " inputs", decodedInputIndices);
      StringAttr inputsLabel = builder.getStringAttr(inputsLabelStr);
      builder.create<IREE::Flow::TensorTraceOp>(dispatchOp.getLoc(),
                                                inputsLabel, inputValues);

      // Output tensors:
      SmallVector<Value> resultTensorValues =
          filterTensorValues(dispatchOp.getResults());
      SmallVector<int64_t> decodedOutputIndices =
          setToRowMajorLayout(builder, resultTensorValues);
      std::string outputsLabelStr = appendDecodedValuesToLabel(
          entryPointName + " outputs", decodedOutputIndices);

      // Set insertion point to the last decoded value before creating the
      // trace op.
      Operation *lastResult = resultTensorValues.front().getDefiningOp();
      DominanceInfo domInfo(funcOp);
      for (Value v : resultTensorValues) {
        if (domInfo.dominates(lastResult, v.getDefiningOp())) {
          lastResult = v.getDefiningOp();
        }
      }
      builder.setInsertionPointAfter(lastResult);
      StringAttr outputsLabel = builder.getStringAttr(outputsLabelStr);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(), outputsLabel, resultTensorValues);
    }

    RewritePatternSet patterns(&getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
