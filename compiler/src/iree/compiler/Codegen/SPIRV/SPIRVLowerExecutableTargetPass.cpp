// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/Support/Debug.h"
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

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

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

  switch (translationInfo.getDispatchLoweringPassPipeline()) {
  case CodeGenPipeline::SPIRVBaseLowering:
    addSPIRVBaseLoweringPassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVBaseDistribute:
    addSPIRVBaseDistributePassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVBaseVectorize:
    addSPIRVBaseVectorizePassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVSubgroupReduce:
    addSPIRVSubgroupReducePassPipeline(pipeline);
    break;
  case CodeGenPipeline::SPIRVCooperativeMatrixVectorize: {
    FailureOr<int64_t> maybeDepth =
        getSoftwarePipelineDepth(translationInfo.getConfiguration());
    FailureOr<int64_t> maybeStage =
        getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
    if (failed(maybeDepth) || failed(maybeStage)) {
      funcOp.emitOpError("invalid cooperative matrix pipeline without "
                         "software pipelining configuration.");
      return signalPassFailure();
    }
    addSPIRVCooperativeMatrixVectorizePassPipeline(pipeline, *maybeDepth,
                                                   *maybeStage);
    break;
  }
  case CodeGenPipeline::SPIRVMatmulPromoteVectorize: {
    FailureOr<int64_t> maybeDepth =
        getSoftwarePipelineDepth(translationInfo.getConfiguration());
    FailureOr<int64_t> maybeStage =
        getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
    if (failed(maybeDepth) || failed(maybeStage)) {
      funcOp.emitOpError("invalid matmul pipeline without software "
                         "pipelining configuration.");
      return signalPassFailure();
    }
    addSPIRVMatmulPromoteVectorizePassPipeline(pipeline, *maybeDepth,
                                               *maybeStage);
    break;
  }
  case CodeGenPipeline::SPIRVWinogradVectorize:
    addSPIRVWinogradVectorizePassPipeline(pipeline);
    break;
  // No pipeline specified, nothing to do.
  case CodeGenPipeline::None:
    return;
  default:
    funcOp.emitOpError("unsupported pipeline on GPU target.");
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V lowering pass pipeline:\n";
    pipeline.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  if (failed(runPipeline(pipeline, funcOp))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
