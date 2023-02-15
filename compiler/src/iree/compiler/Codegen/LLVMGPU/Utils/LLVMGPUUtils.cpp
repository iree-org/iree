// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace iree_compiler {

void createAsyncGroups(func::FuncOp funcOp, bool useMMASync) {
  llvm::SmallSetVector<vector::TransferWriteOp, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](vector::TransferWriteOp writeOp) {
    if (!writeOp.getPermutationMap().isMinorIdentity() ||
        writeOp.getVectorType().getRank() != 1 || !writeOp.isDimInBounds(0)) {
      return WalkResult::advance();
    }
    auto addressSpaceAttr = writeOp.getShapedType()
                                .cast<MemRefType>()
                                .getMemorySpace()
                                .dyn_cast_or_null<gpu::AddressSpaceAttr>();
    if (!addressSpaceAttr || addressSpaceAttr.getValue() !=
                                 gpu::GPUDialect::getWorkgroupAddressSpace()) {
      return WalkResult::advance();
    }
    auto read = writeOp.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!read || read.getVectorType() != writeOp.getVectorType() ||
        !read.isDimInBounds(0) || !read.getPermutationMap().isMinorIdentity())
      return WalkResult::advance();
    if (!((read.getVectorType().getElementType().isF32() &&
           read.getVectorType().getNumElements() <= 4) ||
          (read.getVectorType().getElementType().isF16() &&
           read.getVectorType().getNumElements() <= 8)))
      return WalkResult::advance();
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<vector::TransferWriteOp> group;
    vector::TransferWriteOp writeOp = *copyToSharedMem.begin();
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation* nextNode = writeOp.getOperation();
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      auto readOp = dyn_cast<vector::TransferReadOp>(nextNode);
      // ignore read from a different address space.
      if (readOp) {
        auto addressSpaceAttr = readOp.getShapedType()
                                    .cast<MemRefType>()
                                    .getMemorySpace()
                                    .dyn_cast_or_null<gpu::AddressSpaceAttr>();
        if (!addressSpaceAttr ||
            addressSpaceAttr.getValue() !=
                gpu::GPUDialect::getWorkgroupAddressSpace()) {
          continue;
        }
      }
      auto nextWriteOp = dyn_cast<vector::TransferWriteOp>(nextNode);
      if (nextWriteOp && copyToSharedMem.count(nextWriteOp)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextWriteOp);
        group.push_back(nextWriteOp);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    OpBuilder builder(funcOp.getContext());
    for (vector::TransferWriteOp writeOp : group) {
      builder.setInsertionPoint(writeOp);
      auto readOp = writeOp.getVector().getDefiningOp<vector::TransferReadOp>();
      Value token = builder.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp.getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
          writeOp.getSource(), writeOp.getIndices(), readOp.getSource(),
          readOp.getIndices(),
          builder.getIndexAttr(readOp.getVectorType().getNumElements()),
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
    for (vector::TransferWriteOp writeOp : group) writeOp.erase();
  }
}

}  // namespace iree_compiler
}  // namespace mlir
