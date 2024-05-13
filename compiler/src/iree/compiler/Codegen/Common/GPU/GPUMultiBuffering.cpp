// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUMULTIBUFFERINGPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUMultiBufferingPass final
    : impl::GPUMultiBufferingPassBase<GPUMultiBufferingPass> {
  using GPUMultiBufferingPassBase::GPUMultiBufferingPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // First hoist all shared memory allocations to the entry block of the
    // function. We can see memref.alloc in loops after bufferizing scf.forall
    // with promoted shared memory usage inside.

    SmallVector<memref::AllocOp> allocs;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      if (hasSharedMemoryAddressSpace(allocOp.getType()))
        allocs.push_back(allocOp);
    });

    assert(funcOp.getBlocks().size() == 1);
    for (memref::AllocOp allocOp : allocs) {
      if (allocOp->getParentOp() != funcOp)
        allocOp->moveBefore(&*funcOp.begin()->begin());
    }

    // Then perform multibuffering transformations.

    allocs.clear();
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      // Skip allocations not used in a loop.
      for (Operation *user : allocOp->getUsers()) {
        auto loop = user->getParentOfType<scf::ForOp>();
        if (!loop)
          return WalkResult::advance();
      }
      allocs.push_back(allocOp);
      return WalkResult::advance();
    });
    // Apply multi-buffering to all of them.
    for (memref::AllocOp alloc : allocs) {
      if (failed(memref::multiBuffer(alloc, numBuffers))) {
        // Error out and stop if any buffer cannot be multi buffered, as future
        // software pipelining transformations will assume this happened.
        alloc.emitOpError("cannot be multi-buffered");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
