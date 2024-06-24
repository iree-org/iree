// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-flow-convert-dispatch-regions-to-workgroups"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CONVERTDISPATCHREGIONSTOWORKGROUPSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
struct ConvertDispatchRegionsToWorkgroupsPass
    : public IREE::Flow::impl::ConvertDispatchRegionsToWorkgroupsPassBase<
          ConvertDispatchRegionsToWorkgroupsPass> {
  using IREE::Flow::impl::ConvertDispatchRegionsToWorkgroupsPassBase<
      ConvertDispatchRegionsToWorkgroupsPass>::
      ConvertDispatchRegionsToWorkgroupsPassBase;
  void runOnOperation() override;
};
} // namespace

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void ConvertDispatchRegionsToWorkgroupsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  TensorDimTrackingRewriter rewriter(funcOp);

  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  funcOp.walk([&](Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  for (auto regionOp : regionOps) {
    auto maybeWorkgroupOp =
        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) {
      regionOp.emitError(
          "failed to convert dispatch.region op to dispatch.workgroup op");
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler::IREE::Flow
