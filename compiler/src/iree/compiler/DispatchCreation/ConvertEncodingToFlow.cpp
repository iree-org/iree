// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-dispatch-creation-convert-encoding-to-flow"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CONVERTENCODINGTOFLOWPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct ConvertEncodingToFlowPass
    : public impl::ConvertEncodingToFlowPassBase<ConvertEncodingToFlowPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void ConvertEncodingToFlowPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  funcOp.walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
    if (encodingOp->getParentOfType<IREE::Flow::DispatchRegionOp>()) {
      return;
    }
    rewriter.setInsertionPointAfter(encodingOp);
    Value source = encodingOp.getSource();
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, encodingOp.getLoc(), source);
    SmallVector<Value> dynamicDimSizes;
    std::tie(std::ignore, dynamicDimSizes) = decomposeMixedValues(mixedSizes);
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEncodeOp>(
        encodingOp, encodingOp.getResultType(), source,
        /*operand_dims=*/dynamicDimSizes, /*result_dims=*/dynamicDimSizes);
  });
}

} // namespace mlir::iree_compiler::DispatchCreation
