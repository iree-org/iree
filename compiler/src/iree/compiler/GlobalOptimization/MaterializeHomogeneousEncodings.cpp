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
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

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
    FunctionLikeNest(passManager).addPass(createMaterializeEncodingIntoNopPass);
    FunctionLikeNest(passManager).addPass(createCanonicalizerPass);
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto executableTargets =
        IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(moduleOp);
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
    FunctionLikeNest(passManager).addPass([&]() {
      return createCPUMaterializeUpperBoundTileSizePass(executableTargets);
    });
    FunctionLikeNest(passManager).addPass([&]() {
      return createCPUMaterializeEncodingPass(executableTarget);
    });
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createMaterializeHomogeneousEncodingsPass() {
  return std::make_unique<MaterializeHomogeneousEncodingsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
