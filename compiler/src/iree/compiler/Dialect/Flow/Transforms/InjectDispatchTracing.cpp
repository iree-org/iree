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

static SmallVector<Value> getWithRowMajorLayout(RewriterBase &rewriter,
                                                ArrayRef<Value> values) {
  SmallVector<Value> rowMajorTensors;
  for (auto v : values) {
    auto rankedTensorType = dyn_cast<RankedTensorType>(v.getType());
    if (!rankedTensorType || !rankedTensorType.getEncoding()) {
      rowMajorTensors.push_back(v);
      continue;
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfterValue(v);
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, v.getLoc(), v);
    SmallVector<Value> dynamicDimSizes;
    std::tie(std::ignore, dynamicDimSizes) = decomposeMixedValues(mixedSizes);
    Value rowMajorTensor = rewriter.create<IREE::Flow::TensorEncodeOp>(
        v.getLoc(), rankedTensorType.dropEncoding(), v,
        /*operand_dims=*/dynamicDimSizes, /*result_dims=*/dynamicDimSizes);
    rowMajorTensors.push_back(rowMajorTensor);
  }
  return rowMajorTensors;
}

namespace {

struct InjectDispatchTracingPass
    : public IREE::Flow::impl::InjectDispatchTracingPassBase<
          InjectDispatchTracingPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    for (auto dispatchOp : funcOp.getFunctionBody().getOps<DispatchOp>()) {
      std::string entryPointName = dispatchOp.getEntryPointName();

      // Input tensors:
      OpBuilder builder(dispatchOp);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(),
          builder.getStringAttr(entryPointName + " inputs"),
          getWithRowMajorLayout(rewriter,
                                filterTensorValues(dispatchOp.getArguments())));

      // Output tensors:
      builder.setInsertionPointAfter(dispatchOp);
      builder.create<IREE::Flow::TensorTraceOp>(
          dispatchOp.getLoc(),
          builder.getStringAttr(entryPointName + " outputs"),
          getWithRowMajorLayout(rewriter,
                                filterTensorValues(dispatchOp.getResults())));
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
