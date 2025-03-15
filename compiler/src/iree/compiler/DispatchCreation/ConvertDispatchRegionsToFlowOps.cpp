// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#define DEBUG_TYPE "iree-dispatch-creation-convert-dispatch-regions-to-flow-ops"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CONVERTDISPATCHREGIONSTOFLOWOPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct ConvertDispatchRegionsToFlowOpsPass
    : public impl::ConvertDispatchRegionsToFlowOpsPassBase<
          ConvertDispatchRegionsToFlowOpsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static std::optional<IREE::Encoding::SetEncodingOp>
getEncodingFromSetEncodingDispatchRegion(
    IREE::Flow::DispatchRegionOp regionOp) {
  Region &region = regionOp.getBody();
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }
  Block &block = region.front();
  if (!llvm::hasSingleElement(block.without_terminator())) {
    return std::nullopt;
  }
  auto encoding = dyn_cast<IREE::Encoding::SetEncodingOp>(*block.begin());
  if (!encoding) {
    return std::nullopt;
  }
  return encoding;
}

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void ConvertDispatchRegionsToFlowOpsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  IRRewriter rewriter(&getContext());
  funcOp.walk([&](IREE::Flow::DispatchRegionOp op) {
    std::optional<IREE::Encoding::SetEncodingOp> encodingOp =
        getEncodingFromSetEncodingDispatchRegion(op);
    if (!encodingOp) {
      return;
    }
    rewriter.setInsertionPointAfter(op);
    Value source = encodingOp->getSource();
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, op.getLoc(), source);
    SmallVector<Value> dynamicDimSizes;
    std::tie(std::ignore, dynamicDimSizes) = decomposeMixedValues(mixedSizes);
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEncodeOp>(
        op, encodingOp->getResultType(), source, dynamicDimSizes);
  });
}
} // namespace mlir::iree_compiler::DispatchCreation
