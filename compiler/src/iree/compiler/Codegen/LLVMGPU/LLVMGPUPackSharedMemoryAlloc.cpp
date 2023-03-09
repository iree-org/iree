// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "iree-llvmgpu-pack-shared-memory-alloc"

namespace mlir {
namespace iree_compiler {

using AliasGroup = SmallVector<Operation *>;

/// Analyze the liverange of allocs and set them in individual groups if
/// possible and mark allocs with an attributes.
/// The algorithm is a simplistic memory allocation solution. It sorts
/// allocations into alias groups. Everytime two alloc's liverange interfers
/// they are merge into the same group. If a new alloc is part of multiple alias
/// groups all those are merged into one. At the end we are left with groups of
/// allocations that are disjoint and can use the same memory.
// TODO: Move this to a common place if needed by other backends.
static void analyzeSharedMemoryAlloc(func::FuncOp funcOp,
                                     const SmallVector<Operation *> &allocs,
                                     SmallVector<AliasGroup> &aliasGroups) {
  struct AllocGroup {
    SmallVector<Operation *> allocs;
    // Keep track of every operation where any of the alloc in the group is
    // live.
    llvm::DenseSet<Operation *> liveness;
  };
  Liveness liveness(funcOp);
  SmallVector<AllocGroup> groups;
  for (Operation *alloc : allocs) {
    SmallVector<size_t> aliasGroups;
    for (size_t i : llvm::seq<size_t>(0, groups.size())) {
      AllocGroup &group = groups[i];
      for (Operation *user : alloc->getUsers()) {
        // Skip the whole analysis if any user is a subview.
        // TODO: This could be extended if needed by recursively merging
        // liveness.
        if (isa<memref::SubViewOp>(user)) return;
        if (group.liveness.count(user)) {
          aliasGroups.push_back(i);
          break;
        }
      }
    }
    if (aliasGroups.empty()) {
      // If we didn't find any alias group create a new one.
      AllocGroup &newGroup = groups.emplace_back();
      newGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo =
          liveness.resolveLiveness(alloc->getResult(0));
      newGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
    } else {
      // Merge the alloc into the first alias group it interfers with.
      AllocGroup &mergeGroup = groups[aliasGroups[0]];
      mergeGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo =
          liveness.resolveLiveness(alloc->getResult(0));
      mergeGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
      // Then merge all the other alias groups into the first group.
      for (size_t i = 1, e = aliasGroups.size(); i < e; i++) {
        AllocGroup &group = groups[aliasGroups[i]];
        mergeGroup.allocs.insert(mergeGroup.allocs.end(), group.allocs.begin(),
                                 group.allocs.end());
        mergeGroup.liveness.insert(group.liveness.begin(),
                                   group.liveness.end());
        // For simplicity we leave the group empty and don't remove it.
        group.allocs.clear();
        group.liveness.clear();
      }
    }
  }

  LLVM_DEBUG({
    for (size_t i = 0; i < groups.size(); i++) {
      llvm::dbgs() << "Alias group " << i << ":\n";
      for (Operation *op : groups[i].allocs) op->dump();
    }
  });

  for (size_t i = 0; i < groups.size(); i++) {
    if (groups[i].allocs.empty()) continue;
    aliasGroups.push_back(std::move(groups[i].allocs));
  }
}

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

static int64_t getAllocSize(Operation *op, DataLayout &dataLayout) {
  auto allocOp = cast<memref::AllocOp>(op);
  int64_t numElements = allocOp.getType().getNumElements();
  return (dataLayout.getTypeSizeInBits(allocOp.getType().getElementType()) *
          numElements) /
         8;
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
    analyzeSharedMemoryAlloc(funcOp, allocs, aliasGroups);
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
    DataLayout dataLayout = DataLayout::closest(funcOp);
    builder.setInsertionPointToStart(&(*funcOp.getBody().begin()));
    int64_t maxAlloc = 0;
    for (size_t i = 0; i < aliasGroups.size(); i++) {
      int64_t allocSize = 0;
      for (Operation *alloc : aliasGroups[i]) {
        allocSize += getAllocSize(alloc, dataLayout);
      }
      maxAlloc = std::max(maxAlloc, allocSize);
    }

    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    MemRefType allocType = MemRefType::get({maxAlloc}, builder.getI8Type(),
                                           AffineMap(), workgroupSpace);
    Value packedAlloc =
        builder.create<memref::AllocOp>(funcOp.getLoc(), allocType);
    for (size_t i = 0; i < aliasGroups.size(); i++) {
      int64_t offset = 0;
      for (Operation *alloc : aliasGroups[i]) {
        Location loc = alloc->getLoc();
        builder.setInsertionPoint(alloc);
        Value offsetValue = builder.create<arith::ConstantIndexOp>(loc, offset);
        Value newAlloc = builder.create<memref::ViewOp>(
            packedAlloc.getLoc(), alloc->getResultTypes()[0], packedAlloc,
            offsetValue, ArrayRef<Value>({}));
        offset += getAllocSize(alloc, dataLayout);
        alloc->replaceAllUsesWith(ArrayRef<Value>({newAlloc}));
        alloc->erase();
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUPackSharedMemoryAlloc() {
  return std::make_unique<LLVMGPUPackSharedMemoryAllocPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
