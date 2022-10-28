// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct SPIRVVectorToGPUPass
    : public SPIRVVectorToGPUBase<SPIRVVectorToGPUPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect, AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    RewritePatternSet flatternpatterns(funcOp.getContext());
    populateVectorTransferToGPUMMAPreparationPatterns(flatternpatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(flatternpatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(funcOp.getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populatePrepareVectorToMMAPatterns(patterns, /*useNvGpu=*/false);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    convertVectorToMMAOps(funcOp);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVVectorToGPUPass() {
  return std::make_unique<SPIRVVectorToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
