// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <deque>

#define DEBUG_TYPE "iree-codegen-gpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

static void
debugPrintUniqueOperationNames(const std::deque<Operation *> &worklist) {
  DenseSet<StringRef> uniqueNames;
  for (Operation *op : worklist) {
    uniqueNames.insert(op->getName().getStringRef());
  }

  for (StringRef name : uniqueNames) {
    llvm::dbgs().indent(2) << "* " << name << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

/// A rewriter for the pattern rewriting driver.
struct VectorDistributionRewriter : PatternRewriter {
  VectorDistributionRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

/// Custom listener to store emitted ops that needs to be distributed.
struct VectorDistributionListener : RewriterBase::Listener {
  bool hasOpsToBeDistributed() { return !toBeDistributed.empty(); }

  void clearOpsToBeDistributed() { return toBeDistributed.clear(); }

  const std::deque<Operation *> &getOpsToBeDistributed() const {
    return toBeDistributed;
  }

  void notifyOperationModified(Operation *op) override {
    if (IREE::VectorExt::isMarkedForRedistribution(op)) {
      IREE::VectorExt::clearRedistributionMark(op);
      toBeDistributed.push_back(op);
    }
  }

private:
  std::deque<Operation *> toBeDistributed;
};

static void applyVectorDistribution(Operation *root,
                                    const FrozenRewritePatternSet &patterns) {

  VectorDistributionRewriter rewriter(root->getContext());
  VectorDistributionListener listener;
  rewriter.setListener(&listener);
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Collect all the operations to be distributed.
  std::deque<Operation *> worklist;
  LLVM_DEBUG(llvm::dbgs() << "Collecting operations to be distributed\n");
  root->walk([&](Operation *op) {
    // The distribution of mask op is special.
    // Although the signature set for visibility purposes
    // but it will be distributed when the body is
    // distributed. Therefore, we explicitly exclude
    // the yield and the mask op.
    if (IREE::VectorExt::hasOpSignature(op) &&
        !isa<vector::MaskOp, vector::YieldOp>(op)) {
      worklist.push_back(op);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Operations to be distributed:\n");
  LLVM_DEBUG(debugPrintUniqueOperationNames(worklist));

  // Note that the pattern application here never runs on a newly created
  // operation. It always runs on an existing operation. This ensures that no
  // invalidated state of the analysis is ever used.
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (op == nullptr) {
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Distributing: ");
    LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << ": Failed to distribute operation:\n");
      continue;
    }

    // Move recently emitted operations that needs to be distributed
    // from the local/rewriter worklist into the "global" worklist.
    if (listener.hasOpsToBeDistributed()) {
      auto opstoBeDistributed = listener.getOpsToBeDistributed();

      LLVM_DEBUG(llvm::dbgs()
                 << "Recently emitted operations to be distributed:\n");
      LLVM_DEBUG(debugPrintUniqueOperationNames(opstoBeDistributed));

      worklist.insert(worklist.end(), opstoBeDistributed.begin(),
                      opstoBeDistributed.end());
      listener.clearOpsToBeDistributed();
    }

    LLVM_DEBUG(llvm::dbgs().indent(2)
               << ": Successfully distributed operation:\n");
  }
}

LogicalResult distributeVectorOps(Operation *root,
                                  RewritePatternSet &distributionPatterns,
                                  VectorLayoutOptions &options) {
  // Run the analysis and determine the layouts.
  LLVM_DEBUG(llvm::dbgs() << "Running Layout Analysis\n");
  llvm::MapVector<Value, VectorLayoutInterface> layouts;
  propagateVectorLayoutInfo(root, layouts);
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Succeeded\n");
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  // Go to each operation, and set its distribution signature.
  LLVM_DEBUG(
      llvm::dbgs() << "Setting distribution signatures for operations\n");
  root->walk([&](Operation *op) {
    if (failed(IREE::VectorExt::setOpSignature(op, layouts, options))) {
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping operation because not all vector "
                        "operands/results have a layout:\n";
        op->print(llvm::dbgs());
      });
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Distribution signatures set\n");
  LLVM_DEBUG(root->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  FrozenRewritePatternSet frozenPatterns(std::move(distributionPatterns));
  applyVectorDistribution(root, frozenPatterns);

  RewritePatternSet patterns(root->getContext());
  IREE::VectorExt::ToSIMDOp::getCanonicalizationPatterns(patterns,
                                                         root->getContext());
  IREE::VectorExt::ToSIMTOp::getCanonicalizationPatterns(patterns,
                                                         root->getContext());
  if (failed(applyPatternsGreedily(root, std::move(patterns)))) {
    return failure();
  }

  // Remove signature after distribution.
  root->walk([](Operation *op) { IREE::VectorExt::removeOpSignature(op); });

  if (options.verifyConversion()) {
    WalkResult hasConversionOp = root->walk([](Operation *op) {
      if (isa<IREE::VectorExt::ToSIMDOp, IREE::VectorExt::ToSIMTOp>(op)) {
        for (auto user : op->getUsers()) {
          if (!isa<IREE::VectorExt::ToSIMDOp, IREE::VectorExt::ToSIMTOp>(
                  user)) {
            LLVM_DEBUG({
              llvm::dbgs() << "Found live cast op: " << *op << "\n";
              llvm::dbgs() << "With live user: " << *user << "\n";
            });
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (hasConversionOp.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

} // namespace mlir::iree_compiler
