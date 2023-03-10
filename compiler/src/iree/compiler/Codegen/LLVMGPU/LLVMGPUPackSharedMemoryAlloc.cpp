// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace iree_compiler {

/// Insert barriers and wait operations if there are allocs of a different alias
/// group before the given alloc.
static void addBarrier(func::FuncOp funcOp, Operation *alloc,
                       ArrayRef<Operation *> aliasGroup) {
  Block *entryBlock = &(*funcOp.getBlocks().begin());
  bool needBarrier = false;
  if (alloc->getBlock() != entryBlock) {
    needBarrier = true;
  } else {
    for (Operation &op : entryBlock->getOperations()) {
      if (&op == alloc) break;
      if (op.getNumRegions() != 0) {
        needBarrier = true;
        break;
      }
      if (isa<memref::AllocOp>(&op)) {
        if (std::find(aliasGroup.begin(), aliasGroup.end(), &op) ==
            aliasGroup.end()) {
          needBarrier = true;
          break;
        }
      }
    }
  }
  if (!needBarrier) return;
  OpBuilder builder(alloc);
  // TODO: make it a option if needed.
  bool hasAsyncCopies = true;
  if (hasAsyncCopies) {
    Value groupToken = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        SmallVector<Value>());
    builder.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                             builder.getI32IntegerAttr(0));
  }
  builder.create<gpu::BarrierOp>(alloc->getLoc());
}

namespace {

struct LLVMGPUPackSharedMemoryAllocPass
    : public LLVMGPUPackSharedMemoryAllocBase<
          LLVMGPUPackSharedMemoryAllocPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    DominanceInfo dominators(funcOp);
    SmallVector<Operation *> allocs;
    funcOp.walk([&](memref::AllocOp alloc) {
      if (hasSharedMemoryAddressSpace(alloc.getType())) {
        allocs.push_back(alloc);
      }
    });
    // First sink the alloc as low as possible in the CFG.
    sinkOpsInCFG(allocs, dominators);
    SmallVector<AliasGroup> aliasGroups;
    analyseAllocsForPacking(funcOp, allocs, aliasGroups);
    // If there is 1 or less alias group there is nothing to do.
    if (aliasGroups.size() <= 1) return;

    // Pack all the allocations into one i8 alloc.
    // We may need to add extra barriers to make sure we are done writting or
    // reading from the previous alias group before starting a new one.
    for (size_t i = 0; i < aliasGroups.size(); i++) {
      for (Operation *alloc : aliasGroups[i]) {
        addBarrier(funcOp, alloc, aliasGroups[i]);
      }
    }

    OpBuilder builder(funcOp.getContext());
    packAllocs(builder, funcOp, aliasGroups);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUPackSharedMemoryAlloc() {
  return std::make_unique<LLVMGPUPackSharedMemoryAllocPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
