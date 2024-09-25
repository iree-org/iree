// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
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

// TODO: Remove the flag once the codegen can handle the late materialization
// path. This is mainly for testing.
static llvm::cl::opt<bool> clEnableExperimentalRocmDataTiling(
    "iree-global-opt-experimental-rocm-data-tiling",
    llvm::cl::desc("Enables data-tiling materializatino for rocm backends "
                   "(experimental)."),
    llvm::cl::init(false));

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

  void addNopPipeline(OpPassManager &passManager) {
    FunctionLikeNest(passManager).addPass(createMaterializeEncodingIntoNopPass);
    FunctionLikeNest(passManager).addPass(createCanonicalizerPass);
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run()))
      return signalPassFailure();

    SetVector<IREE::HAL::ExecutableTargetAttr> executableTargets;
    deviceAnalysis.gatherAllExecutableTargets(executableTargets);
    OpPassManager passManager(moduleOp.getOperationName());
    if (executableTargets.size() != 1) {
      addNopPipeline(passManager);
      if (failed(runPipeline(passManager, moduleOp))) {
        return signalPassFailure();
      }
      return;
    }

    // TODO: vmvx has its own logic about supporting dynamic tile
    // sizes. It is not fully integrated into the pipeline, so we remain the
    // materialization to the end.
    auto executableTarget = executableTargets[0];
    if (executableTarget.getBackend() == "vmvx") {
      return;
    }

    // Only llvm-cpu and rocm backends handle encodings for now, others just go
    // with nop.
    if (executableTarget.getBackend() == "llvm-cpu") {
      passManager.addPass(createCPUMaterializeHostEncodingPass());
    } else if (clEnableExperimentalRocmDataTiling &&
               executableTarget.getBackend() == "rocm") {
      passManager.addPass(createGPUMaterializeHostEncodingPass());
      FunctionLikeNest(passManager).addPass([&]() {
        return createDecomposePackUnPackOpsPass(/*tileOuterToOne=*/false,
                                                /*useOnlyReshapes=*/true,
                                                /*controlFn=*/std::nullopt);
      });
    } else {
      addNopPipeline(passManager);
    }
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
