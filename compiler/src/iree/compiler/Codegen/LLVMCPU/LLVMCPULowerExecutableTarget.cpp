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
  // Check for self first.
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) {
    return funcOp.emitOpError(
        "failed to get tiling configuration for pipeline");
  }
  // In presence of custom_op, need to look for root op within the custom op
  // to get the root lowering config.
  if (auto customOp = dyn_cast<IREE::LinalgExt::CustomOp>(rootOp.value())) {
    SmallVector<Operation *> nestedCustomOps = getComputeOps(customOp);
    FailureOr<Operation *> nestedRootOp = getRootOperation(nestedCustomOps);
    if (succeeded(nestedRootOp)) {
      rootOp = nestedRootOp.value();
    }
  }

  auto rootLoweringConfig =
      iree_compiler::getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(
          rootOp.value());
  if (!rootLoweringConfig) {
    return funcOp.emitOpError("expected root op to have a lowering config");
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
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault:
    addCPUDefaultPassPipeline(pipeline);
    break;
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUBufferOpsTileAndVectorize: {
    FailureOr<TilingConfig> tilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(tilingConfig)) {
      return signalPassFailure();
    }
    addCPUBufferOpsTileAndVectorizePipeline(pipeline, tilingConfig.value(),
                                            pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert: {
    FailureOr<TilingConfig> tilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(tilingConfig)) {
      return signalPassFailure();
    }
    addMultiTilingExpertPassPipeline(pipeline, tilingConfig.value(),
                                     pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert: {
    FailureOr<TilingConfig> tilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(tilingConfig)) {
      return signalPassFailure();
    }
    addConvTileAndDecomposeExpertPassPipeline(pipeline, tilingConfig.value(),
                                              pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert: {
    FailureOr<TilingConfig> tilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(tilingConfig)) {
      return signalPassFailure();
    }
    addMmt4dTilingExpertPassPipeline(pipeline, tilingConfig.value(),
                                     pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDataTiling: {
    FailureOr<TilingConfig> tilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(tilingConfig)) {
      return signalPassFailure();
    }
    addCPUDataTilingPipeline(pipeline, tilingConfig.value(), pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPULinalgExtTileAndVectorize: {
    FailureOr<TilingConfig> tilingConfig = getTilingConfigForPipeline(funcOp);
    if (failed(tilingConfig)) {
      return signalPassFailure();
    }
    addCPULinalgExtTileAndVectorizePipeline(pipeline, tilingConfig.value(),
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
