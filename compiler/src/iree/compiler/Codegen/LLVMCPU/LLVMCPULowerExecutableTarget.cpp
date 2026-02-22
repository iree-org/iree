// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/CodegenOptions.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/DebugLog.h"
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

#define DEBUG_TYPE "kernel-dispatch"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttrInterface;

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
  using Base::Base;
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

/// Returns a new lowering config with distribution tile sizes set to zeros.
static IREE::CPU::LoweringConfigAttr
getConfigWithZeroDistributionTiles(IREE::CPU::LoweringConfigAttr config) {
  using TilingLevel = IREE::CPU::TilingLevel;
  MLIRContext *ctx = config.getContext();
  SmallVector<NamedAttribute> items;
  for (int i : IREE::CPU::getTilingLevelsAsInts()) {
    if (!config.hasTilingLevel(i)) {
      continue;
    }
    auto level = static_cast<TilingLevel>(i);
    if (level != TilingLevel::DistributionTiles) {

      items.emplace_back(IREE::CPU::getTilingLevelName(level),
                         config.getTilingLevelAttr(i));
      continue;
    }
    // Reset distribution tiles to zeros.
    SmallVector<int64_t> origSizes = config.getWorkgroupTileSizes();
    SmallVector<int64_t> zeroSizes(origSizes.size(), 0);
    auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        ctx, zeroSizes, /*interchange=*/{}, /*scalableFlags=*/{});
    items.emplace_back(IREE::CPU::getTilingLevelName(level), newLevel);
  }
  return IREE::CPU::LoweringConfigAttr::get(ctx, items);
}

static LoweringConfigAttrInterface
getRootLoweringConfig(FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  for (Operation *op : computeOps) {
    LoweringConfigAttrInterface loweringConfig = getLoweringConfig(op);
    if (loweringConfig && loweringConfig.hasWorkgroupTilingLevel()) {
      return loweringConfig;
    }
  }
  return nullptr;
}

void LLVMCPULowerExecutableTargetPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    // Do nothing without target
    return;
  }
  DictionaryAttr targetConfig = target.getConfiguration();

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return;
  }

  auto pipeline = translationInfo.getDispatchLoweringPassPipeline();
  LLVMCPUPipelineOptions pipelineOpts;
  pipelineOpts.cpuOpts = cpuOptions.getValue();

  // If distribution is disabled, reset distribution tile sizes to zeros in the
  // lowering config so the IR reflects the actual behavior.
  if (pipelineOpts.cpuOpts.disableDistribution) {
    LDBG() << "Distribution is disabled, resetting distribution tile sizes to "
              "zeros in lowering configs.";
    funcOp.walk([&](Operation *op) {
      auto config = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
      if (config && config.hasWorkgroupTilingLevel()) {
        setLoweringConfig(op, getConfigWithZeroDistributionTiles(config));
      }
    });
  }

  if (isX86(targetConfig) || isRISCV(targetConfig)) {
    pipelineOpts.useConfiguredVectorSizes = false;
  }
  pipelineOpts.decomposePackUnPackOps =
      isOptEnabled(funcOp, getEnableDecompositionStr());
  pipelineOpts.lowerToAVX2 = hasAVX2Feature(targetConfig);
  pipelineOpts.enableVectorMasking =
      isX86(targetConfig) || isRISCV(targetConfig) ||
      (isAArch64(targetConfig) && hasAnySVEFeature(targetConfig));
  // TODO(#16956): The decomposition of attention op leads to complex control
  // flow, which leads to non-trivial stack allocation. Enforcing masking for
  // targets that do not support native vector masking enables the
  // functionality, though the performance may be suboptimal.
  auto hasAttentionOp = [](FunctionOpInterface funcOp) {
    bool hasAttention = false;
    funcOp.walk([&](IREE::LinalgExt::AttentionOp) {
      hasAttention = true;
      return WalkResult::interrupt();
    });
    return hasAttention;
  };
  if (isAArch64(targetConfig) && hasAttentionOp(funcOp)) {
    pipelineOpts.enableVectorMasking = true;
  }
  pipelineOpts.enableAArch64SME = isAArch64(targetConfig) &&
                                  hasAnySVEFeature(targetConfig) &&
                                  hasSMEFeature(targetConfig);
  pipelineOpts.enableAArch64I8mm =
      isAArch64(targetConfig) && hasI8mmFeature(targetConfig);
  pipelineOpts.enablePeeling = isOptEnabled(funcOp, getEnableLoopPeelingStr());

  LoweringConfigAttrInterface loweringConfig = getRootLoweringConfig(funcOp);
  OpPassManager passManager(func::FuncOp::getOperationName());
  switch (pipeline) {
  // No pipleline specified, nothing to do.
  case IREE::Codegen::DispatchLoweringPassPipeline::None:
    return;
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault: {
    addCPUDefaultPassPipeline(passManager, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUBufferOpsTileAndVectorize: {
    addCPUBufferOpsTileAndVectorizePipeline(passManager, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert: {
    assert(loweringConfig && "expected a valid lowering config");
    addMultiTilingExpertPassPipeline(passManager, loweringConfig, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert: {
    addConvTileAndDecomposeExpertPassPipeline(passManager, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::Mmt4dTilingExpert: {
    addMmt4dTilingExpertPassPipeline(passManager, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::CPUDataTiling: {
    addCPUDataTilingPipeline(passManager, pipelineOpts);
    break;
  }
  case IREE::Codegen::DispatchLoweringPassPipeline::
      CPULinalgExtTileAndVectorize: {
    addCPULinalgExtTileAndVectorizePipeline(passManager, pipelineOpts);
    break;
  }
  default:
    funcOp.emitOpError("Unsupported pipeline on CPU target.");
    return signalPassFailure();
  }

  if (failed(runPipeline(passManager, funcOp))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
