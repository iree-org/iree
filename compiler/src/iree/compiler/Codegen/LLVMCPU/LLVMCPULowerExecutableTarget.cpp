// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
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

#define GEN_PASS_DEF_LLVMCPULOWEREXECUTABLETARGETPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to LLVM dialect.
/// In due course this could be used to generate code for all backends.
class LLVMCPULowerExecutableTargetPass
    : public impl::LLVMCPULowerExecutableTargetPassBase<
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

static FailureOr<TilingConfig>
getTilingConfigForPipeline(FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp) || !rootOp.value()) {
    return failure();
  }
  auto rootLoweringConfig =
      iree_compiler::getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(
          rootOp.value());
  if (!rootLoweringConfig) {
    return failure();
  }
  return TilingConfig(rootLoweringConfig);
}

void LLVMCPULowerExecutableTargetPass::runOnOperation() {
  auto funcOp = getOperation();
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    // Do nothing without target
    return;
  }

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo)
    return;

  LLVMCPUPipelineOptions pipelineOpts;
  if (isX86(target) || isRISCV(target)) {
    pipelineOpts.useConfiguredVectorSizes = false;
  }
  pipelineOpts.decomposePackUnPackOps =
      isOptEnabled(funcOp, getEnableDecompositionStr());
  pipelineOpts.lowerToAVX2 = hasAVX2Feature(target);
  pipelineOpts.enableVectorMasking =
      isX86(target) || isRISCV(target) ||
      (isAArch64(target) && hasAnySVEFeature(target));
  pipelineOpts.enableAArch64SME =
      isAArch64(target) && hasAnySVEFeature(target) && hasSMEFeature(target);
  pipelineOpts.enableAArch64I8mm = isAArch64(target) && hasI8mmFeature(target);
  pipelineOpts.enablePeeling = isOptEnabled(funcOp, getEnableLoopPeelingStr());

  OpPassManager pipeline(func::FuncOp::getOperationName());
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  // No pipleline specified, nothing to do.
  case IREE::Codegen::DispatchLoweringPassPipeline::None:
    return;
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    addCPUDefaultPassPipeline(pipeline, maybeTilingConfig);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUBufferOpsTileAndVectorize: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(maybeTilingConfig)) {
      funcOp.emitOpError("Tiling Config is necessary for "
                         "CPUBufferOpsTileAndVectorize pipeline.");
      return signalPassFailure();
    }
    addCPUBufferOpsTileAndVectorizePipeline(pipeline, *maybeTilingConfig,
                                            pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(maybeTilingConfig)) {
      funcOp.emitOpError(
          "Tiling Config is necessary for CPUDoubleTilingExpert pipeline.");
      return signalPassFailure();
    }
    addMultiTilingExpertPassPipeline(pipeline, *maybeTilingConfig,
                                     pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(maybeTilingConfig)) {
      funcOp.emitOpError("Tiling Config is necessary for "
                         "CPUConvTileAndDecomposeExpert pipeline.");
      return signalPassFailure();
    }
    addConvTileAndDecomposeExpertPassPipeline(pipeline, *maybeTilingConfig,
                                              pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(maybeTilingConfig)) {
      funcOp.emitOpError(
          "Tiling Config is necessary for Mmt4dTilingExpert pipeline.");
      return signalPassFailure();
    }
    addMmt4dTilingExpertPassPipeline(pipeline, *maybeTilingConfig,
                                     pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDataTiling: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(maybeTilingConfig)) {
      funcOp.emitOpError(
          "Tiling Config is necessary for CPUDataTiling pipeline.");
      return signalPassFailure();
    }
    addCPUDataTilingPipeline(pipeline, *maybeTilingConfig, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPULinalgExtTileAndVectorize: {
    auto maybeTilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(maybeTilingConfig)) {
      funcOp.emitOpError("Tiling Config is necessary for "
                         "CPULinalgExtTileAndVectorize pipeline.");
      return signalPassFailure();
    }
    addCPULinalgExtTileAndVectorizePipeline(pipeline, *maybeTilingConfig,
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
} // namespace mlir::iree_compiler
