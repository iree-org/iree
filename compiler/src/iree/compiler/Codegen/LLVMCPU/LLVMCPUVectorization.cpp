// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-vectorization"

namespace mlir {
namespace iree_compiler {
namespace {
/// Returns the op that contains lowering config. Checks whether the provided op
/// contains the lowering config and returns it. Otherwise, tries to find the
/// lowering config across the function. If there are multiple ops with the same
/// lowering configs, returns the first one found. Returns failure if there are
/// multiple op with different lowering config.
/// TODO(hanchung): This is copied from LinalgTensorCodegenDriver.cpp. We should
/// refactor it to Utils.h.
static FailureOr<Operation *> getRootOp(Operation *op) {
  // Check for self first.
  if (iree_compiler::getLoweringConfig(op)) {
    return op;
  }

  // Get the function op.
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    funcOp = op->getParentOfType<func::FuncOp>();
  }

  assert(funcOp && "Missing funcOp");

  Operation *rootOp = nullptr;
  mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr rootLoweringConfig;
  auto result = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig = iree_compiler::getLoweringConfig(op);
    if (!loweringConfig) {
      return WalkResult::advance();
    }
    if (rootLoweringConfig) {
      if (rootLoweringConfig != loweringConfig) {
        return WalkResult::interrupt();
      }
    } else {
      rootOp = op;
      rootLoweringConfig = loweringConfig;
    }
    return WalkResult::advance();
  });

  if (!rootOp || result.wasInterrupted()) {
    return failure();
  }
  return rootOp;
}

/// Computes the canonical shape used to vectorize this dispatch. Retrieves
/// the vectorization tile sizes (parallel and reduction levels) out of the
/// lowering config and adjusts them to the format expected by the Linalg
/// vectorizer.
static SmallVector<int64_t> getCanonicalVectorShape(func::FuncOp funcOp) {
  FailureOr<Operation *> rootOp = getRootOp(funcOp);
  if (failed(rootOp)) {
    return {};
  }

  unsigned numTileLevels =
      mlir::iree_compiler::getNumTileLevels(rootOp.value());
  if (numTileLevels < 3) {
    return {};
  }

  // Retrieve the tile sizes from the last two tiling levels (parallel and
  // reduction) used for vectorization.
  SmallVector<int64_t> canonicalVectorShape =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 2);
  SmallVector<int64_t> reductionTileSizes =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 1);

  if (!reductionTileSizes.empty()) {
    assert(canonicalVectorShape.size() == reductionTileSizes.size() &&
           "Unexpected tile sizes");

    // Combine the reduction tile sizes with the parallel tile sizes already in
    // the canonical vector shape.
    for (int i = 0, end = canonicalVectorShape.size(); i < end; ++i) {
      if (reductionTileSizes[i] > 0)
        canonicalVectorShape[i] = reductionTileSizes[i];
    }
  }

  // Replace zeros in canonical vector shape to turn it into a valid shape.
  std::replace(canonicalVectorShape.begin(), canonicalVectorShape.end(), 0, 1);
  return canonicalVectorShape;
}

// Give the canonical vector shape of a dispatch, returns the vector sizes for a
// particular linalg op within that dispatch.
static SmallVector<int64_t> getVectorSizes(
    linalg::LinalgOp linalgOp, ArrayRef<int64_t> canonicalVectorShape) {
  FailureOr<Operation *> rootOp = getRootOp(linalgOp);
  if (failed(rootOp)) {
    return {};
  }

  // TODO: Infer the tiles sizes for an op that is not the root op.
  if (*rootOp != linalgOp.getOperation()) {
    return {};
  }

  if (canonicalVectorShape.empty()) {
    return {};
  }

  assert(canonicalVectorShape.size() >= linalgOp.getNumLoops() &&
         "Unexpected canonical vector shape or number of loops");

  // Return the valid canonical vector shape subset based on the number of loops
  // of the linalg op.
  SmallVector<int64_t> vecSize(
      canonicalVectorShape.take_front(linalgOp.getNumLoops()));
  for (auto [idx, val] : llvm::enumerate(linalgOp.getStaticLoopRanges())) {
    if (ShapedType::isDynamic(val)) continue;
    vecSize[idx] = std::max(vecSize[idx], val);
  }

  return vecSize;
}

class LLVMCPUVectorizationPass
    : public LLVMCPUVectorizationBase<LLVMCPUVectorizationPass> {
 public:
  using LLVMCPUVectorizationBase::LLVMCPUVectorizationBase;
  LLVMCPUVectorizationPass(const LLVMCPUVectorizationPassOptions &options) {
    this->enableVectorMasking.setValue(options.enableVectorMasking);
    this->vectorizePadding.setValue(options.vectorizePadding);
    this->vectorizeGatherAccesses.setValue(options.vectorizeGatherAccesses);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  SmallVector<int64_t> canonicalVectorShape;
  if (enableVectorMasking) {
    canonicalVectorShape = getCanonicalVectorShape(funcOp);
  }

  IRRewriter rewriter(context);
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk(
      [&](linalg::LinalgOp linalgOp) { candidates.push_back(linalgOp); });
  for (auto linalgOp : candidates) {
    SmallVector<int64_t> vectorSizes;
    if (enableVectorMasking) {
      vectorSizes.append(getVectorSizes(linalgOp, canonicalVectorShape));
    }
    (void)linalg::vectorize(rewriter, linalgOp, vectorSizes,
                            vectorizeGatherAccesses);
  };

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
      vectorizationPatterns);
  vector::populateVectorTransferPermutationMapLoweringPatterns(
      vectorizationPatterns);
  vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                            linalg::LinalgCopyVTWForwardingPattern>(
      funcOp.getContext(), /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                      funcOp.getContext());
  vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                       funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  // Apply the pad tensor op vectorization separately to avoid running the
  // GenericPadOpVectorizationPattern too early.
  // TODO: Improve once we have better infrastructure to control pattern
  // application.
  if (vectorizePadding) {
    RewritePatternSet patterns(funcOp.getContext());
    linalg::populatePadOpVectorizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  // Gathers all innermost loops through a post order pruned walk.
  funcOp.walk([](Operation *op) {
    if (auto forOp = dyn_cast<AffineForOp>(op))
      (void)promoteIfSingleIteration(forOp);
    else if (auto forOp = dyn_cast<scf::ForOp>(op))
      (void)promoteIfSingleIteration(forOp);
  });
  linalg::hoistRedundantVectorTransfers(funcOp);
  linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorizationPass() {
  return std::make_unique<LLVMCPUVectorizationPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorizationPass(
    const LLVMCPUVectorizationPassOptions &options) {
  return std::make_unique<LLVMCPUVectorizationPass>(options);
}
}  // namespace iree_compiler
}  // namespace mlir
