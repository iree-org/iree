// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-flow-dispatch-regions-to-workgroups"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_DISPATCHREGIONSTOWORKGROUPSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
struct DispatchRegionsToWorkgroupsPass
    : public IREE::Flow::impl::DispatchRegionsToWorkgroupsPassBase<
          DispatchRegionsToWorkgroupsPass> {
  using IREE::Flow::impl::DispatchRegionsToWorkgroupsPassBase<
      DispatchRegionsToWorkgroupsPass>::DispatchRegionsToWorkgroupsPassBase;
  void runOnOperation() override;
};
} // namespace

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void DispatchRegionsToWorkgroupsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  mlir::TensorDimTrackingRewriter rewriter(funcOp);

  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  funcOp.walk([&](Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  for (auto regionOp : regionOps) {
    auto maybeWorkgroupOp =
        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) {
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler::IREE::Flow
