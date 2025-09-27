// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-convert-to-coalesced-dma"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTTOCOALESCEDDMAPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// Pattern to convert tensor.parallel_insert_slice operations to coalesced DMA
struct ConvertParallelInsertSliceToCoalescedDMA
    : public OpRewritePattern<tensor::ParallelInsertSliceOp> {
  using OpRewritePattern<tensor::ParallelInsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ParallelInsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Analyzing tensor.parallel_insert_slice for coalesced DMA: "
               << insertOp << "\n");

    // Analyze the source and destination patterns
    if (!hasGatherLikePattern(insertOp)) {
      LLVM_DEBUG(llvm::dbgs() << "  No gather-like pattern detected\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Found gather-like pattern in parallel_insert_slice\n");

    // TODO: Implement the transformation logic
    // 1. Create IREE::GPU::CoalescedGatherDMAOp to replace the insert operation
    // 2. Handle the source tensor and destination tensor appropriately
    // 3. Preserve the parallel semantics through proper DMA operations

    // For now, just mark this as a potential transformation point
    LLVM_DEBUG(llvm::dbgs() << "  Would convert to coalesced DMA here\n");

    return failure(); // TODO: Return success() when implementation is complete
  }

private:
  // Check if the insert operation has gather-like characteristics
  bool hasGatherLikePattern(tensor::ParallelInsertSliceOp insertOp) const {
    Value source = insertOp.getSource();

    // Check if source comes from operations that suggest gather patterns
    if (auto definingOp = source.getDefiningOp()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Source defined by: " << definingOp->getName() << "\n");

      // Look for tensor.extract operations (common in gather patterns)
      if (isa<tensor::ExtractOp>(definingOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Source from tensor.extract - potential gather\n");
        return true;
      }

      // Look for tensor.extract_slice operations
      if (isa<tensor::ExtractSliceOp>(definingOp)) {
        LLVM_DEBUG(llvm::dbgs() << "    Source from tensor.extract_slice\n");
        return true;
      }

      // Check for other operations that might indicate scattered access
      if (isa<tensor::InsertOp, tensor::FromElementsOp>(definingOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Source from tensor composition operation\n");
        return true;
      }
    }

    // Check the insertion pattern itself
    auto offsets = insertOp.getMixedOffsets();
    auto sizes = insertOp.getMixedSizes();
    auto strides = insertOp.getMixedStrides();

    LLVM_DEBUG(llvm::dbgs() << "    Insert pattern - offsets: "
                            << offsets.size() << ", sizes: " << sizes.size()
                            << ", strides: " << strides.size() << "\n");

    // For now, consider any non-trivial insertion pattern as potential for DMA
    if (!offsets.empty() || !sizes.empty()) {
      return true;
    }

    return false;
  }
};

struct ConvertToCoalescedDMAPass final
    : impl::ConvertToCoalescedDMAPassBase<ConvertToCoalescedDMAPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running ConvertToCoalescedDMAPass on function: "
                            << funcOp.getName() << "\n");

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add the tensor.parallel_insert_slice conversion pattern
    patterns.add<ConvertParallelInsertSliceToCoalescedDMA>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern application failed\n");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "ConvertToCoalescedDMAPass completed\n");
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertToCoalescedDMAPass() {
  return std::make_unique<ConvertToCoalescedDMAPass>();
}

} // namespace mlir::iree_compiler
