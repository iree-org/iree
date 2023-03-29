// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

static void swizzleSharedMemory(func::FuncOp funcOp) {
  SmallVector<memref::AllocOp> shmAllocOps;
  funcOp->walk([&](memref::AllocOp allocOp) {
    auto memrefType = allocOp.getMemref().getType().cast<MemRefType>();
    auto addressSpaceAttr =
        memrefType.getMemorySpace().dyn_cast_or_null<gpu::AddressSpaceAttr>();
    // Only apply it to shared memory of input operands.
    if (!addressSpaceAttr ||
        addressSpaceAttr.getValue() !=
            gpu::GPUDialect::getWorkgroupAddressSpace() ||
        memrefType.getRank() < 3) {
      return;
    }
    shmAllocOps.push_back(allocOp);
  });
  for (auto allocOp : shmAllocOps) {
    (void)nvgpu::optimizeSharedMemoryReadsAndWrites(funcOp,
                                                    allocOp.getMemref());
  }
}

namespace {
struct LLVMGPUVectorToGPUPass
    : public LLVMGPUVectorToGPUBase<LLVMGPUVectorToGPUPass> {
  LLVMGPUVectorToGPUPass(GPUTensorCoreType tensorCoreType)
      : tensorCoreType(tensorCoreType) {}
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect, AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    bool targetMmaSync = tensorCoreType == GPUTensorCoreType::MMA_SYNC;
    RewritePatternSet flatternpatterns(funcOp.getContext());
    populateVectorTransferToGPUMMAPreparationPatterns(flatternpatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(flatternpatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(funcOp.getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populatePrepareVectorToMMAPatterns(patterns, targetMmaSync);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (targetMmaSync) {
      if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp))) {
        return signalPassFailure();
      }
      // Using TF32 for Float.
      RewritePatternSet f32ToTF32patterns(funcOp.getContext());
      nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns,
                                              nvgpu::MmaSyncF32Lowering::TF32);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(f32ToTF32patterns)))) {
        return signalPassFailure();
      }
    } else {
      if (failed(convertVectorToMMAOps(rewriter, funcOp))) {
        return signalPassFailure();
      }
    }
    createAsyncGroups(rewriter, funcOp, targetMmaSync);

    if (targetMmaSync) {
      swizzleSharedMemory(funcOp);
    }
  }

 private:
  GPUTensorCoreType tensorCoreType;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorToGPU(
    GPUTensorCoreType tensorCoreType) {
  return std::make_unique<LLVMGPUVectorToGPUPass>(tensorCoreType);
}

}  // namespace iree_compiler
}  // namespace mlir
