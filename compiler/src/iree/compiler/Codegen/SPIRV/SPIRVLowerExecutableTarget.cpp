// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lower-executable-target-pass"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVLOWEREXECUTABLETARGETPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

/// Lowers a hal.executable.variant inner module to SPIR-V scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to SPIRV dialect.
class SPIRVLowerExecutableTargetPass final
    : public impl::SPIRVLowerExecutableTargetPassBase<
          SPIRVLowerExecutableTargetPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, affine::AffineDialect,
                gpu::GPUDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
                IREE::LinalgExt::IREELinalgExtDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, scf::SCFDialect,
                spirv::SPIRVDialect, transform::TransformDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void SPIRVLowerExecutableTargetPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) {
    return;
  }

  std::optional<OpPassManager> maybePipeline =
      getFunctionOpInterfacePassManager(funcOp);
  if (!maybePipeline) {
    funcOp.emitOpError(
        "unhandled function-like container during executable lowering");
    return signalPassFailure();
  }
  OpPassManager &pipeline = maybePipeline.value();

  Attribute pipelineAttr = translationInfo.getPassPipeline();
  // No pipeline specified, nothing to do.
  if (isa<IREE::Codegen::NoPipelineAttr>(pipelineAttr)) {
    return;
  }

  // Check for PipelineAttrInterface first (covers GPU::SPIRVPipelineAttr via
  // external model and any custom pipeline attrs).
  auto pipelineIface =
      dyn_cast<IREE::Codegen::PipelineAttrInterface>(pipelineAttr);

  if (!pipelineIface) {
    // Not an interface implementor -- reject any remaining legacy pipeline.
    funcOp.emitOpError("unsupported pipeline on SPIR-V target");
    return signalPassFailure();
  }

  // Build pipeline options for pipelines that need software pipelining config.
  std::unique_ptr<SPIRVCodegenPipelineOptions> spirvOpts;
  if (auto spirvPipeline =
          dyn_cast<IREE::GPU::SPIRVPipelineAttr>(pipelineAttr)) {
    auto value = spirvPipeline.getValue();
    if (value == IREE::GPU::SPIRVLoweringPipeline::CooperativeMatrixVectorize ||
        value == IREE::GPU::SPIRVLoweringPipeline::MatmulPromoteVectorize) {
      FailureOr<int64_t> maybeDepth =
          getSoftwarePipelineDepth(translationInfo.getConfiguration());
      FailureOr<int64_t> maybeStage =
          getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
      if (failed(maybeDepth) || failed(maybeStage)) {
        funcOp.emitOpError("invalid pipeline without software pipelining "
                           "configuration");
        return signalPassFailure();
      }
      spirvOpts = std::make_unique<SPIRVCodegenPipelineOptions>(*maybeDepth,
                                                                *maybeStage);
    }
  }

  if (failed(pipelineIface.buildPipeline(pipeline, spirvOpts.get()))) {
    funcOp.emitOpError("failed to build pass pipeline");
    return signalPassFailure();
  }

  LDBG() << "Using SPIR-V lowering pass pipeline: ";
  LLVM_DEBUG(pipeline.dump());

  if (failed(runPipeline(pipeline, funcOp))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
