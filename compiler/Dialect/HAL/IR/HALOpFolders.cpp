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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
  if (auto exportOp = source().getDefiningOp<TensorExportOp>()) {
    if (exportOp.source().getType() == target().getType() &&
        exportOp.source_encoding() == target_encoding()) {
      return exportOp.source();
    }
  }
  return {};
}

OpFoldResult TensorExportOp::fold(ArrayRef<Attribute> operands) {
  if (auto importOp = source().getDefiningOp<TensorImportOp>()) {
    if (importOp.source().getType() == target().getType() &&
        importOp.target_encoding() == source_encoding()) {
      return importOp.source();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.*
//===----------------------------------------------------------------------===//

namespace {

/// Skips a hal.buffer_view.buffer accessor when the buffer view was created in
/// the same scope and we know the origin buffer.
struct SkipBufferViewBufferOp : public OpRewritePattern<BufferViewBufferOp> {
  using OpRewritePattern<BufferViewBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferViewBufferOp op,
                                PatternRewriter &rewriter) const override {
    if (auto createOp = dyn_cast_or_null<BufferViewCreateOp>(
            op.buffer_view().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.buffer());
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

namespace {

/// Expands a hal.buffer_view.dims op into individual ops for each dimension.
struct ExpandBufferViewDimsOp : public OpRewritePattern<BufferViewDimsOp> {
  using OpRewritePattern<BufferViewDimsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferViewDimsOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> newDimValues;
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      newDimValues.push_back(rewriter.createOrFold<BufferViewDimOp>(
          op.getLoc(), rewriter.getIndexType(), op.buffer_view(),
          rewriter.getIndexAttr(i)));
    }
    rewriter.replaceOp(op, {newDimValues});
    return success();
  }
};

}  // namespace

void BufferViewDimsOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<ExpandBufferViewDimsOp>(context);
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
            op.command_buffer().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.device());
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
    auto newTargetBuffer = op.target_buffer();
    auto newTargetOffset = op.target_offset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.target_buffer().getDefiningOp())) {
      newTargetBuffer = subspanOp.source_buffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.source_offset(), op.target_offset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.target_bufferMutable().assign(newTargetBuffer);
      op.target_offsetMutable().assign(newTargetOffset);
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
    auto newSourceBuffer = op.source_buffer();
    auto newSourceOffset = op.source_offset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.source_buffer().getDefiningOp())) {
      newSourceBuffer = subspanOp.source_buffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.source_offset(), op.source_offset());
      needsUpdate = true;
    }
    auto newTargetBuffer = op.target_buffer();
    auto newTargetOffset = op.target_offset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.target_buffer().getDefiningOp())) {
      newTargetBuffer = subspanOp.source_buffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.source_offset(), op.target_offset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.source_bufferMutable().assign(newSourceBuffer);
      op.source_offsetMutable().assign(newSourceOffset);
      op.target_bufferMutable().assign(newTargetBuffer);
      op.target_offsetMutable().assign(newTargetOffset);
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
    auto bindingBuffers = llvm::to_vector<4>(op.binding_buffers());
    auto bindingOffsets = llvm::to_vector<4>(op.binding_offsets());
    for (size_t i = 0; i < bindingBuffers.size(); ++i) {
      auto *definingOp = bindingBuffers[i].getDefiningOp();
      if (!definingOp) continue;
      if (auto subspanOp = dyn_cast<BufferSubspanOp>(definingOp)) {
        needsUpdate = true;
        bindingBuffers[i] = subspanOp.source_buffer();
        bindingOffsets[i] = rewriter.createOrFold<mlir::arith::AddIOp>(
            subspanOp.getLoc(), subspanOp.source_offset(), bindingOffsets[i]);
      }
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      auto mutableBindingBuffers = op.binding_buffersMutable();
      mutableBindingBuffers.clear();
      mutableBindingBuffers.append(bindingBuffers);
      auto mutableBindingOffsets = op.binding_offsetsMutable();
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

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
