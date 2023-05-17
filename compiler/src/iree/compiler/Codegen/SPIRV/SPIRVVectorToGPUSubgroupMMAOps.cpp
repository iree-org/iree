// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/SPIRVPasses.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct SPIRVVectorToGPUSubgroupMMAPass final
    : public SPIRVVectorToGPUSubgroupMMABase<SPIRVVectorToGPUSubgroupMMAPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect,
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

    IRRewriter rewriter(&getContext());
    if (failed(convertVectorToMMAOps(rewriter, funcOp))) {
      funcOp->emitError("failed conversion to GPU subgroup MMA ops");
      return signalPassFailure();
    }

    // Make sure we actually generate GPU subgroup mma ops.
    WalkResult result = funcOp.walk([](Operation* op) {
      return isa<gpu::SubgroupMmaComputeOp>(op) ? WalkResult::interrupt()
                                                : WalkResult::advance();
    });
    if (!result.wasInterrupted()) {
      funcOp->emitError("no GPU subgroup mma compute ops generated");
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorToGPUSubgroupMMAOpsPass() {
  return std::make_unique<SPIRVVectorToGPUSubgroupMMAPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
