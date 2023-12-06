// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

class MaterializeHomogeneousEncodingsPass
    : public MaterializeHomogeneousEncodingsBase<
          MaterializeHomogeneousEncodingsPass> {
public:
  MaterializeHomogeneousEncodingsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runNopPipeline(ModuleOp &moduleOp) {
    OpPassManager passManager(moduleOp.getOperationName());
    passManager.addNestedPass<func::FuncOp>(
        createMaterializeEncodingIntoNopPass());
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto targetsAttr = moduleOp->getAttrOfType<ArrayAttr>("hal.device.targets");
    if (!targetsAttr || targetsAttr.size() != 1) {
      return runNopPipeline(moduleOp);
    }
    auto deviceTarget = cast<IREE::HAL::DeviceTargetAttr>(targetsAttr[0]);
    SmallVector<IREE::HAL::ExecutableTargetAttr, 4> executableTargets =
        deviceTarget.getExecutableTargets();
    if (executableTargets.size() != 1) {
      return runNopPipeline(moduleOp);
    }
    // TODO: vmvx has its own logic about supporting dynamic tile
    // sizes. It is not fully integrated into the pipeline, so we remain the
    // materialization to the end.
    auto executableTarget = executableTargets[0];
    if (executableTarget.getBackend() == "vmvx") {
      return;
    }

    // Only llvm-cpu backends handle encodings for now, others just go with nop.
    if (executableTarget.getBackend() != "llvm-cpu") {
      return runNopPipeline(moduleOp);
    }

    OpPassManager passManager(moduleOp.getOperationName());
    passManager.addNestedPass<func::FuncOp>(
        createCPUMaterializeUpperBoundTileSizePass(executableTargets));
    passManager.addNestedPass<func::FuncOp>(
        createCPUMaterializeEncodingPass(executableTarget));
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createMaterializeHomogeneousEncodingsPass() {
  return std::make_unique<MaterializeHomogeneousEncodingsPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
