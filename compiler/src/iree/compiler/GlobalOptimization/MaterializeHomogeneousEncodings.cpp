// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_MATERIALIZEHOMOGENEOUSENCODINGSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

namespace {
class MaterializeHomogeneousEncodingsPass
    : public impl::MaterializeHomogeneousEncodingsPassBase<
          MaterializeHomogeneousEncodingsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect, tensor::TensorDialect>();
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
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run()))
      return signalPassFailure();

    SetVector<IREE::HAL::ExecutableTargetAttr> executableTargets;
    deviceAnalysis.gatherAllExecutableTargets(executableTargets);
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
    passManager.addPass(createCPUMaterializeHostEncodingPass());
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
