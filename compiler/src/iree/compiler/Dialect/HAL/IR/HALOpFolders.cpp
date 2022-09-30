// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// hal.tensor.import/export
//===----------------------------------------------------------------------===//

OpFoldResult TensorImportOp::fold(ArrayRef<Attribute> operands) {
  if (auto exportOp = getSource().getDefiningOp<TensorExportOp>()) {
    if (exportOp.getSource().getType() == getTarget().getType() &&
        exportOp.getSourceEncoding() == getTargetEncoding()) {
      return exportOp.getSource();
    }
  }
  return {};
}

OpFoldResult TensorExportOp::fold(ArrayRef<Attribute> operands) {
  if (auto importOp = getSource().getDefiningOp<TensorImportOp>()) {
    if (importOp.getSource().getType() == getTarget().getType() &&
        importOp.getTargetEncoding() == getSourceEncoding()) {
      return importOp.getSource();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.*
//===----------------------------------------------------------------------===//

namespace {

/// Folds hal.buffer.subspans into buffer view creation subspans.
struct FoldBufferViewCreateSubspan
    : public OpRewritePattern<BufferViewCreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferViewCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceBuffer = op.getSourceBuffer();
    auto newSourceOffset = op.getSourceOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getSourceBuffer().getDefiningOp())) {
      newSourceBuffer = subspanOp.getSourceBuffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getSourceOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceBufferMutable().assign(newSourceBuffer);
      op.getSourceOffsetMutable().assign(newSourceOffset);
    });
    return success();
  }
};

}  // namespace

void BufferViewCreateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<FoldBufferViewCreateSubspan>(context);
}

namespace {

/// Skips a hal.buffer_view.buffer accessor when the buffer view was created in
/// the same scope and we know the origin buffer.
struct SkipBufferViewBufferOp : public OpRewritePattern<BufferViewBufferOp> {
  using OpRewritePattern<BufferViewBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferViewBufferOp op,
                                PatternRewriter &rewriter) const override {
    if (auto createOp = dyn_cast_or_null<BufferViewCreateOp>(
            op.getBufferView().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.getSourceBuffer());
      return success();
    }
    return failure();
  }
};

}  // namespace

void BufferViewBufferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<SkipBufferViewBufferOp>(context);
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.*
//===----------------------------------------------------------------------===//

namespace {

/// Skips a hal.command_buffer.device accessor when the device was created in
/// the same scope.
struct SkipCommandBufferDeviceOp
    : public OpRewritePattern<CommandBufferDeviceOp> {
  using OpRewritePattern<CommandBufferDeviceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferDeviceOp op,
                                PatternRewriter &rewriter) const override {
    if (auto createOp = dyn_cast_or_null<CommandBufferCreateOp>(
            op.getCommandBuffer().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.getDevice());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CommandBufferDeviceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SkipCommandBufferDeviceOp>(context);
}

namespace {

/// Folds hal.buffer.subspans into buffer fill offsets.
struct FoldCommandBufferFillBufferSubspans
    : public OpRewritePattern<CommandBufferFillBufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferFillBufferOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newTargetBuffer = op.getTargetBuffer();
    auto newTargetOffset = op.getTargetOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getTargetBuffer().getDefiningOp())) {
      newTargetBuffer = subspanOp.getSourceBuffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getTargetOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetBufferMutable().assign(newTargetBuffer);
      op.getTargetOffsetMutable().assign(newTargetOffset);
    });
    return success();
  }
};

}  // namespace

void CommandBufferFillBufferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldCommandBufferFillBufferSubspans>(context);
}

namespace {

/// Folds hal.buffer.subspans into buffer copy offsets.
struct FoldCommandBufferCopyBufferSubspans
    : public OpRewritePattern<CommandBufferCopyBufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferCopyBufferOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceBuffer = op.getSourceBuffer();
    auto newSourceOffset = op.getSourceOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getSourceBuffer().getDefiningOp())) {
      newSourceBuffer = subspanOp.getSourceBuffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getSourceOffset());
      needsUpdate = true;
    }
    auto newTargetBuffer = op.getTargetBuffer();
    auto newTargetOffset = op.getTargetOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getTargetBuffer().getDefiningOp())) {
      newTargetBuffer = subspanOp.getSourceBuffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getTargetOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceBufferMutable().assign(newSourceBuffer);
      op.getSourceOffsetMutable().assign(newSourceOffset);
      op.getTargetBufferMutable().assign(newTargetBuffer);
      op.getTargetOffsetMutable().assign(newTargetOffset);
    });
    return success();
  }
};

}  // namespace

void CommandBufferCopyBufferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldCommandBufferCopyBufferSubspans>(context);
}

namespace {

/// Folds hal.buffer.subspans into push descriptor bindings.
/// The binding range is always equal to or a subset of the subspan.
struct FoldCommandBufferPushDescriptorSetBufferSubspan
    : public OpRewritePattern<CommandBufferPushDescriptorSetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferPushDescriptorSetOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto bindingBuffers = llvm::to_vector<4>(op.getBindingBuffers());
    auto bindingOffsets = llvm::to_vector<4>(op.getBindingOffsets());
    for (size_t i = 0; i < bindingBuffers.size(); ++i) {
      auto *definingOp = bindingBuffers[i].getDefiningOp();
      if (!definingOp) continue;
      if (auto subspanOp = dyn_cast<BufferSubspanOp>(definingOp)) {
        needsUpdate = true;
        bindingBuffers[i] = subspanOp.getSourceBuffer();
        bindingOffsets[i] = rewriter.createOrFold<mlir::arith::AddIOp>(
            subspanOp.getLoc(), subspanOp.getSourceOffset(), bindingOffsets[i]);
      }
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      auto mutableBindingBuffers = op.getBindingBuffersMutable();
      mutableBindingBuffers.clear();
      mutableBindingBuffers.append(bindingBuffers);
      auto mutableBindingOffsets = op.getBindingOffsetsMutable();
      mutableBindingOffsets.clear();
      mutableBindingOffsets.append(bindingOffsets);
    });
    return success();
  }
};

}  // namespace

void CommandBufferPushDescriptorSetOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldCommandBufferPushDescriptorSetBufferSubspan>(context);
}

//===----------------------------------------------------------------------===//
// hal.device.switch
//===----------------------------------------------------------------------===//

// TODO(benvanik): fold conditions with the same IR tree.
// TODO(benvanik): remove duplicate conditions.
// TODO(benvanik): fold condition expressions (any(always, ...) -> always, etc).
// TODO(benvanik): completely replace switches with just one always block.
// TODO(benvanik): remove conditions with no side-effects.

//===----------------------------------------------------------------------===//
// hal.device.match.id
//===----------------------------------------------------------------------===//

// TODO(benvanik): fold matches that are known true based on device config.

//===----------------------------------------------------------------------===//
// hal.fence.create
//===----------------------------------------------------------------------===//

namespace {

/// Replaces a fence with no timepoints with a null value.
struct ElideUnusedFenceCreate : public OpRewritePattern<FenceCreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    } else {
      return failure();
    }
  }
};

}  // namespace

void FenceCreateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ElideUnusedFenceCreate>(context);
}

//===----------------------------------------------------------------------===//
// hal.fence.join
//===----------------------------------------------------------------------===//

namespace {

/// Replaces a fence join with no operands with a null value.
struct ElideEmptyFenceJoin : public OpRewritePattern<FenceJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceJoinOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0) return failure();
    rewriter.replaceOpWithNewOp<IREE::Util::NullOp>(op,
                                                    op.getResult().getType());
    return success();
  }
};

// Produces a deduplicated and null-elided operand list.
// Returns None if nothing changed.
static Optional<std::vector<Value>> deduplicateFenceOperands(
    ValueRange operands) {
  SetVector<Value> newOperands;
  for (auto operand : operands) {
    if (isa_and_nonnull<IREE::Util::NullOp>(operand.getDefiningOp())) {
      // Drop null values as they don't mean anything. Ideally we'd reach back
      // a little further here but that's best done in an IPO pass.
      continue;
    }
    newOperands.insert(operand);
  }

  if (newOperands.size() == operands.size()) return None;
  return newOperands.takeVector();
}

/// Deduplicates fence join operands and drops nulls.
struct DeduplicateFenceJoinFences : public OpRewritePattern<FenceJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceJoinOp op,
                                PatternRewriter &rewriter) const override {
    auto newOperands = deduplicateFenceOperands(op.getFences());
    if (!newOperands) return failure();
    rewriter.replaceOpWithNewOp<FenceJoinOp>(op, op.getResult().getType(),
                                             newOperands.value());
    return success();
  }
};

}  // namespace

void FenceJoinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<ElideEmptyFenceJoin>(context);
  results.insert<DeduplicateFenceJoinFences>(context);
}

//===----------------------------------------------------------------------===//
// hal.fence.await
//===----------------------------------------------------------------------===//

namespace {

/// Elides a fence await with no fences.
struct ElideEmptyFenceAwait : public OpRewritePattern<FenceAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceAwaitOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getFences().empty()) return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, /*ok=*/0, 32);
    return success();
  }
};

/// Deduplicates fence await operands and drops nulls.
struct DeduplicateFenceAwaitFences : public OpRewritePattern<FenceAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceAwaitOp op,
                                PatternRewriter &rewriter) const override {
    auto newOperands = deduplicateFenceOperands(op.getFences());
    if (newOperands == None) return failure();
    rewriter.replaceOpWithNewOp<FenceAwaitOp>(op, op.getStatus().getType(),
                                              op.getTimeoutMillis(),
                                              newOperands.value());
    return success();
  }
};

}  // namespace

void FenceAwaitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ElideEmptyFenceAwait>(context);
  results.insert<DeduplicateFenceAwaitFences>(context);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
