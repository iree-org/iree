// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORTOGPUPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

static void swizzleSharedMemory(mlir::FunctionOpInterface funcOp) {
  SmallVector<memref::AllocOp> shmAllocOps;
  funcOp->walk([&](memref::AllocOp allocOp) {
    // Only apply it to shared memory of input operands.
    if (!hasSharedMemoryAddressSpace(allocOp.getType()) ||
        allocOp.getType().getRank() < 3) {
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
struct LLVMGPUVectorToGPUPass final
    : impl::LLVMGPUVectorToGPUPassBase<LLVMGPUVectorToGPUPass> {
  using Base::Base;
  LLVMGPUVectorToGPUPass(GPUTensorCoreType tensorCoreType)
      : tensorCoreType(tensorCoreType) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect, affine::AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    bool targetMmaSync = tensorCoreType == GPUTensorCoreType::MMA_SYNC;
    RewritePatternSet flatternpatterns(funcOp.getContext());
    populateVectorTransferToGPUMMAPreparationPatterns(flatternpatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(flatternpatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(funcOp.getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populatePrepareVectorToMMAPatterns(patterns, targetMmaSync);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
      if (failed(applyPatternsGreedily(getOperation(),
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
      // Fold subview on memory copy to enable the application of shared memory
      // swizzling optimization.
      RewritePatternSet pattern(funcOp.getContext());
      memref::populateFoldMemRefAliasOpPatterns(pattern);
      if (failed(applyPatternsGreedily(funcOp, std::move(pattern)))) {
        return signalPassFailure();
      }
      swizzleSharedMemory(funcOp);
    }
  }

private:
  GPUTensorCoreType tensorCoreType;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUVectorToGPUPass(GPUTensorCoreType tensorCoreType) {
  return std::make_unique<LLVMGPUVectorToGPUPass>(tensorCoreType);
}

} // namespace mlir::iree_compiler
