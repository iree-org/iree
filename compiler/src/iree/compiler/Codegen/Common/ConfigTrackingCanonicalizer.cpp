// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-config-tracking-canonicalizer"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONFIGTRACKINGCANONICALIZERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static Operation *skipCastsDefiningOp(Value v) {
  auto producer = v.getDefiningOp();
  while (auto castProducer = dyn_cast<tensor::CastOp>(producer)) {
    producer = castProducer.getSource().getDefiningOp();
  }
  return producer;
}

void ConfigTrackingListener::notifyOperationReplaced(Operation *op,
                                                     ValueRange replacement) {
  // We have no way to track replacements without a producer.
  if (replacement.empty()) {
    return;
  }

  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(op);
  if (!loweringConfig) {
    return;
  }

  // Must have a producer of the same type to track the lowering config.
  auto producer = skipCastsDefiningOp(replacement.front());
  if (!producer || producer->getName() != op->getName()) {
    return;
  }

  for (auto v : replacement.drop_front()) {
    // Conservatively require that all replacements are produced by the same
    // operation.
    if (skipCastsDefiningOp(v) != producer) {
      return;
    }
  }

  // No need to add the lowering config if it's already present.
  if (getLoweringConfig(producer)) {
    return;
  }

  setLoweringConfig(producer, loweringConfig);
}

namespace {

/// Fold tensor.insert_slice into tensor.empty when the inserted slice covers
/// the entire destination tensor. This pattern handles both static and dynamic
/// dimensions by comparing mixed sizes.
///
/// Example:
///   %0 = tensor.empty(%d) : tensor<?x128xf32>
///   %1 = tensor.insert_slice %src into %0[0, 0][%d, 128][1, 1]
///        : tensor<?x128xf32> into tensor<?x128xf32>
/// Folds to:
///   %1 = tensor.cast %src : tensor<?x128xf32> to tensor<?x128xf32>
/// Or just %src if types match.
class FoldFullInsertSliceIntoEmpty final
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    // Check that the destination is a tensor.empty.
    auto emptyOp = insertOp.getDest().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) {
      return failure();
    }

    // Check that all offsets are zero and all strides are one.
    if (!areAllConstantIntValue(insertOp.getMixedOffsets(), 0) ||
        !areAllConstantIntValue(insertOp.getMixedStrides(), 1)) {
      return failure();
    }

    // Get the mixed sizes of both the insert_slice and the empty tensor.
    SmallVector<OpFoldResult> insertSizes = insertOp.getMixedSizes();
    SmallVector<OpFoldResult> emptySizes = emptyOp.getMixedSizes();

    // Check that the sizes are equal (including dynamic dimensions).
    // Use isEqualConstantIntOrValueArray which handles both static and dynamic
    // sizes correctly.
    if (!isEqualConstantIntOrValueArray(insertSizes, emptySizes)) {
      return failure();
    }

    // The insert_slice covers the entire tensor, so we can replace it with
    // the source (possibly with a cast if types differ).
    Value source = insertOp.getSource();
    if (source.getType() == insertOp.getType()) {
      rewriter.replaceOp(insertOp, source);
    } else {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(insertOp, insertOp.getType(),
                                                  source);
    }
    return success();
  }
};

/// Add the corresponding fast-math flags to operations given a floating-point
/// optimization mode.
// TODO: For now we only allow default flags, such as arithmetic reassociation.
struct ConfigTrackingCanonicalizerPass final
    : impl::ConfigTrackingCanonicalizerPassBase<
          ConfigTrackingCanonicalizerPass> {
public:
  using Base::Base;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Normal);

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // Add custom patterns.
    owningPatterns.add<FoldFullInsertSliceIntoEmpty>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    ConfigTrackingListener listener;
    {
      config.setListener(&listener);
      LogicalResult didConverge =
          applyPatternsGreedily(getOperation(), *patterns, config);
      config.setListener(nullptr);
      if (this->testConvergence && failed(didConverge)) {
        getOperation()->emitError("Canonicalizer failed to converge");
        return signalPassFailure();
      }
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace
} // namespace mlir::iree_compiler
