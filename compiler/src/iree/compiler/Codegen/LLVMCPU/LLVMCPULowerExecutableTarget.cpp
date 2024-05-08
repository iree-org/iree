// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to LLVM dialect.
/// In due course this could be used to generate code for all backends.
class LLVMCPULowerExecutableTargetPass
    : public LLVMCPULowerExecutableTargetBase<
          LLVMCPULowerExecutableTargetPass> {
public:
  LLVMCPULowerExecutableTargetPass() = default;
  LLVMCPULowerExecutableTargetPass(
      const LLVMCPULowerExecutableTargetPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;
};
} // namespace

// TODO(dcaballe): We temporarily need this utility to retrieve a valid
// lowering config. We should be able to remove this once we have a lowering
// config attribute per op.
static FailureOr<LoweringConfigAttr>
getRootLoweringConfig(FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  // Check for self first.
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  auto rootLoweringConfig = iree_compiler::getLoweringConfig(rootOp.value());
  if (rootLoweringConfig) {
    return rootLoweringConfig;
  }

  return failure();
}

static TilingConfig getTilingConfigForPipeline(FunctionOpInterface funcOp) {
  auto maybeLoweringConfig = getRootLoweringConfig(funcOp);
  assert(succeeded(maybeLoweringConfig) &&
         "Pipeline requires a lowering config");
  return TilingConfig(*maybeLoweringConfig);
}

void LLVMCPULowerExecutableTargetPass::runOnOperation() {
  auto funcOp = getOperation();
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    // Do nothing without target
    return;
  }

  LLVMCPUPipelineOptions pipelineOpts;
  if (isX86(target) || isRISCV(target)) {
    pipelineOpts.useConfiguredVectorSizes = false;
  }
  pipelineOpts.lowerToAVX2 = hasAVX2Feature(target);
  pipelineOpts.enableVectorMasking =
      isX86(target) || isRISCV(target) ||
      (isAArch64(target) && hasAnySVEFeature(target));
  pipelineOpts.enableUkernels = hasUkernel(target);
  pipelineOpts.enableAArch64SSVE =
      isAArch64(target) && hasAnySVEFeature(target) && hasSMEFeature(target);

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo)
    return;

  OpPassManager pipeline(func::FuncOp::getOperationName());
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  // No pipleline specified, nothing to do.
  case IREE::Codegen::DispatchLoweringPassPipeline::None:
    return;
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault:
    addCPUDefaultPassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUBufferOpsTileAndVectorize: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    addCPUBufferOpsTileAndVectorizePipeline(pipeline, tilingConfig,
                                            pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    addMultiTilingExpertPassPipeline(pipeline, tilingConfig, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUDoubleTilingPeelingExpert: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    pipelineOpts.enablePeeling = true;
    addMultiTilingExpertPassPipeline(pipeline, tilingConfig, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    pipelineOpts.enablePeeling = isLoopPeelingEnabled(&funcOp);
    addConvTileAndDecomposeExpertPassPipeline(pipeline, tilingConfig,
                                              pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    addMmt4dTilingExpertPassPipeline(pipeline, tilingConfig, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDataTiling: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    addCPUDataTilingPipeline(pipeline, tilingConfig, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPULinalgExtTileAndVectorize: {
    TilingConfig tilingConfig = getTilingConfigForPipeline(funcOp);
    addCPULinalgExtTileAndVectorizePipeline(pipeline, tilingConfig,
                                            pipelineOpts);
    break;
  }
  default:
    funcOp.emitOpError("Unsupported pipeline on CPU target.");
    return signalPassFailure();
  }

  if (failed(runPipeline(pipeline, funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMCPULowerExecutableTargetPass() {
  return std::make_unique<LLVMCPULowerExecutableTargetPass>();
}

} // namespace mlir::iree_compiler
