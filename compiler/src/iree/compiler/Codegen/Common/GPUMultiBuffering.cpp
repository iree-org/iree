// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct GPUMultiBufferingPass
    : public GPUMultiBufferingBase<GPUMultiBufferingPass> {
  GPUMultiBufferingPass(unsigned numBuffers) : numBuffers(numBuffers) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<memref::AllocOp> allocs;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      // Skip allocations not used in a loop.
      for (Operation* user : allocOp->getUsers()) {
        auto loop = user->getParentOfType<scf::ForOp>();
        if (!loop) return WalkResult::advance();
      }
      allocs.push_back(allocOp);
      return WalkResult::advance();
    });
    // Apply multi-buffering to all of them.
    for (memref::AllocOp alloc : allocs) {
      if (failed(memref::multiBuffer(alloc, numBuffers)))
        // Stop if any buffer cannot be multi buffered as pipelining will assume
        // this happened.
        return signalPassFailure();
    }
  }

 private:
  unsigned numBuffers;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUMultiBuffering(
    unsigned numBuffers) {
  return std::make_unique<GPUMultiBufferingPass>(numBuffers);
}

}  // namespace iree_compiler
}  // namespace mlir
