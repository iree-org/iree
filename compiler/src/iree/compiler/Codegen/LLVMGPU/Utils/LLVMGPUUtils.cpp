// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;

#define DEBUG_TYPE "llvm-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
namespace iree_compiler {

static bool isContiguousStore(Operation* write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0)) {
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  return false;
}

static bool isContiguousRead(Operation* read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  return false;
}

static Value getMemrefOperand(Operation* op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getSource();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getSource();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

static Value getValueStored(Operation* writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

static Operation::operand_range getIndices(Operation* op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

void createAsyncGroups(func::FuncOp funcOp, bool useMMASync) {
  LLVM_DEBUG(DBGS() << "Start asyncGroups: useMMASync=" << useMMASync << "\n");
  llvm::SmallSetVector<Operation*, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation* writeOp) {
    if (!isContiguousStore(writeOp)) {
      return WalkResult::advance();
    }
    LLVM_DEBUG(DBGS() << "--candidate writeOp: " << writeOp << "\n");
    Value vectorVal = getValueStored(writeOp);
    if (vectorVal.getType().cast<VectorType>().getRank() != 1) {
      LLVM_DEBUG(
          DBGS()
          << "----writeOp is not an inbounds 1-D minor identity -> Skip \n");
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    auto addressSpaceAttr = memrefOperand.getType()
                                .cast<MemRefType>()
                                .getMemorySpace()
                                .dyn_cast_or_null<gpu::AddressSpaceAttr>();
    if (!addressSpaceAttr || addressSpaceAttr.getValue() !=
                                 gpu::GPUDialect::getWorkgroupAddressSpace()) {
      LLVM_DEBUG(DBGS() << "----address space is not workgroup -> Skip \n");
      return WalkResult::advance();
    }
    Operation* readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LLVM_DEBUG(DBGS() << "----no readOp defining the writeOp -> Skip \n");
      return WalkResult::advance();
    }

    VectorType vecType = vectorVal.getType().cast<VectorType>();
    if (!((vecType.getElementType().isF32() && vecType.getNumElements() <= 4) ||
          (vecType.getElementType().isF16() &&
           vecType.getNumElements() <= 8))) {
      LLVM_DEBUG(
          DBGS() << "----readOp is not (<=4)xf32 or (<=8)xf16 -> Skip \n");
      return WalkResult::advance();
    }

    LLVM_DEBUG(DBGS() << "--writeOp can be made async -> SUCCESS\n");
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation*> group;
    Operation* writeOp = *copyToSharedMem.begin();
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation* nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation* readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        auto addressSpaceAttr = memrefOperand.getType()
                                    .cast<MemRefType>()
                                    .getMemorySpace()
                                    .dyn_cast_or_null<gpu::AddressSpaceAttr>();
        if (!addressSpaceAttr ||
            addressSpaceAttr.getValue() !=
                gpu::GPUDialect::getWorkgroupAddressSpace()) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    OpBuilder builder(funcOp.getContext());
    for (Operation* writeOp : group) {
      builder.setInsertionPoint(writeOp);
      Value vectorVal = getValueStored(writeOp);
      Operation* readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value token = builder.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          builder.getIndexAttr(
              vectorVal.getType().cast<VectorType>().getNumElements()),
          Value(),
          /*bypassL1=*/useMMASync ? builder.getUnitAttr() : UnitAttr());
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    builder.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                             nullptr);
    // Clean up old stores.
    for (Operation* writeOp : group) writeOp->erase();
  }
}

}  // namespace iree_compiler
}  // namespace mlir
