// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "iree-flow-materialize-default-workgroup-count-region"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_MATERIALIZEDEFAULTWORKGROUPCOUNTREGIONPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

/// Creates the workgroup count region where the materialized computation
/// is derived as a program slice of the body of the dispatch. This method
/// - Computes the `workload` to use for the `workgroupsOp`, which are
///   derived from the values captured by the `workgroupsOp`.
/// - Populates the workgroup count region for this with the placeholder
///   op `flow.dispatch.workgroups_count_from_body_slice`. This op is
///   resolved in the backends into the actual workgroup count computation.
/// - To correlate back to the captured workload,
/// `flow.dispatch.workload.ordinal`
///   to map the captured operand to the position in the workload list.
static void createDefaultWorkgroupCountRegion(
    RewriterBase &rewriter, IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
  Region &workgroupCountBody = workgroupsOp.getWorkgroupCount();
  if (!workgroupCountBody.empty()) {
    // Preserve pre-existing workgroup count region.
    return;
  }

  // Compute the `workload`. For now all `IndexType` are treated as workload.
  SmallVector<Value> workload;
  SmallVector<Type> workloadTypes;
  SmallVector<Location> workloadLocs;
  for (auto argument : workgroupsOp.getArguments()) {
    Type argumentType = argument.getType();
    if (!llvm::isa<IndexType>(argumentType))
      continue;
    workload.push_back(argument);
    workloadTypes.push_back(argumentType);
    workloadLocs.push_back(argument.getLoc());
  }

  // Populate the count region.
  Block *block =
      rewriter.createBlock(&workgroupCountBody, workgroupCountBody.end(),
                           workloadTypes, workloadLocs);
  Location loc = workgroupsOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(block);
  auto defaultCountOp =
      rewriter.create<IREE::Flow::DispatchWorkgroupCountFromSliceOp>(
          loc, block->getArguments());
  rewriter.create<IREE::Flow::ReturnOp>(loc, defaultCountOp.getResults());

  // Update the `workgroupsOp` region.
  rewriter.modifyOpInPlace(workgroupsOp, [&]() {
    // Update the workload of the op.
    workgroupsOp.getWorkloadMutable().assign(workload);

    // Annotate the values captures as workload with their position in the
    // workload list.
    Region &body = workgroupsOp.getWorkgroupBody();
    if (body.empty()) {
      return;
    }
    rewriter.setInsertionPointToStart(&body.front());
    int ordinalNumber = 0;
    for (auto [index, operand] : llvm::enumerate(workgroupsOp.getArguments())) {
      if (!llvm::isa<IndexType>(operand.getType()))
        continue;
      BlockArgument arg = workgroupsOp.getInputBlockArgument(index);
      auto ordinalOp = rewriter.create<IREE::Flow::DispatchWorkloadOrdinalOp>(
          loc, arg, rewriter.getIndexAttr(ordinalNumber++));
      rewriter.replaceAllUsesExcept(arg, ordinalOp, ordinalOp);
    }
  });
}

namespace {
struct MaterializeDefaultWorkgroupCountRegionPass
    : public IREE::Flow::impl::MaterializeDefaultWorkgroupCountRegionPassBase<
          MaterializeDefaultWorkgroupCountRegionPass> {
  using IREE::Flow::impl::MaterializeDefaultWorkgroupCountRegionPassBase<
      MaterializeDefaultWorkgroupCountRegionPass>::
      MaterializeDefaultWorkgroupCountRegionPassBase;
  void runOnOperation() override;
};
} // namespace

// populates the workgroup count region.
void MaterializeDefaultWorkgroupCountRegionPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  TensorDimTrackingRewriter rewriter(funcOp);

  // Populate the workgroup_count region of flow.dispatch.workgroups operation
  // that dont already have a region
  funcOp.walk([&](Flow::DispatchWorkgroupsOp workgroupsOp) {
    createDefaultWorkgroupCountRegion(rewriter, workgroupsOp);
  });
}

} // namespace mlir::iree_compiler::IREE::Flow
