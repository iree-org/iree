// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/MaterializedLayoutTarget.h"

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_VERIFYMATERIALIZEDLAYOUTTARGETPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

IREE::HAL::ExecutableTargetAttr getMaterializedLayoutTarget(Operation *op) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  if (!moduleOp) {
    return {};
  }
  return dyn_cast_if_present<IREE::HAL::ExecutableTargetAttr>(
      moduleOp->getDiscardableAttr(kMaterializedLayoutTargetAttrName));
}

void setMaterializedLayoutTarget(ModuleOp moduleOp,
                                 IREE::HAL::ExecutableTargetAttr target) {
  moduleOp->setDiscardableAttr(kMaterializedLayoutTargetAttrName, target);
}

LogicalResult verifyMaterializedLayoutTarget(ModuleOp moduleOp) {
  Attribute attr =
      moduleOp->getDiscardableAttr(kMaterializedLayoutTargetAttrName);
  if (!attr) {
    return success();
  }
  auto layoutTarget = dyn_cast<IREE::HAL::ExecutableTargetAttr>(attr);
  if (!layoutTarget) {
    return moduleOp.emitError()
           << "expected '" << kMaterializedLayoutTargetAttrName
           << "' to be a #hal.executable.target";
  }

  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return failure();
  }
  SetVector<IREE::HAL::ExecutableTargetAttr> targets;
  deviceAnalysis.gatherAllExecutableTargets(targets);
  if (targets.size() != 1) {
    return moduleOp.emitError()
           << "expected a single executable target for a module with "
              "materialized layouts, but found "
           << targets.size();
  }
  if (targets.front() != layoutTarget) {
    return moduleOp.emitError()
           << "cannot retarget a module with layouts materialized for "
           << layoutTarget << " to " << targets.front();
  }
  return success();
}

namespace {
struct VerifyMaterializedLayoutTargetPass final
    : impl::VerifyMaterializedLayoutTargetPassBase<
          VerifyMaterializedLayoutTargetPass> {
  void runOnOperation() override {
    if (failed(verifyMaterializedLayoutTarget(getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
