// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

/// Helper to convert copy to shared memory to async copy. This creates groups
/// of consecutive copies and emit wait operation right after.
static void createAsyncGroups(FuncOp funcOp) {
  llvm::SmallSetVector<vector::TransferWriteOp, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](vector::TransferWriteOp writeOp) {
    if (!writeOp.permutation_map().isMinorIdentity() ||
        writeOp.getVectorType().getRank() != 1 || !writeOp.isDimInBounds(0) ||
        writeOp.getShapedType().cast<MemRefType>().getMemorySpaceAsInt() !=
            gpu::GPUDialect::getWorkgroupAddressSpace())
      return WalkResult::advance();
    auto read = writeOp.vector().getDefiningOp<vector::TransferReadOp>();
    if (!read || read.getVectorType() != writeOp.getVectorType() ||
        !read.isDimInBounds(0) || !read.permutation_map().isMinorIdentity())
      return WalkResult::advance();
    if (read.getVectorType().getNumElements() > 4 ||
        !read.getVectorType().getElementType().isF32())
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
          !nextNode->hasTrait<OpTrait::HasRecursiveSideEffects>())
        continue;
      auto readOp = dyn_cast<vector::TransferReadOp>(nextNode);
      // ignore read from a different address space.
      if (readOp &&
          readOp.getShapedType().cast<MemRefType>().getMemorySpaceAsInt() !=
              gpu::GPUDialect::getWorkgroupAddressSpace()) {
        continue;
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
      auto readOp = writeOp.vector().getDefiningOp<vector::TransferReadOp>();
      Value token = builder.create<gpu::DeviceAsyncCopyOp>(
          writeOp.getLoc(), gpu::DeviceAsyncTokenType::get(funcOp.getContext()),
          writeOp.source(), writeOp.indices(), readOp.source(),
          readOp.indices(),
          builder.getIndexAttr(readOp.getVectorType().getNumElements()));
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = builder.create<gpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), gpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    builder.create<gpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                           nullptr);
    // Clean up old stores.
    for (vector::TransferWriteOp writeOp : group) writeOp.erase();
  }
}

namespace {

struct LLVMGPUVectorToGPUPass
    : public LLVMGPUVectorToGPUBase<LLVMGPUVectorToGPUPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populatePrepareVectorToMMAPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    convertVectorToMMAOps(getOperation());
    createAsyncGroups(getOperation());
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUVectorToGPU() {
  return std::make_unique<LLVMGPUVectorToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
