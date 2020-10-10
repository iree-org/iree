// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
// hal.variable.*
//===----------------------------------------------------------------------===//

namespace {

/// Converts variable initializer functions that evaluate to a constant to a
/// specified initial value.
struct InlineConstVariableOpInitializer : public OpRewritePattern<VariableOp> {
  using OpRewritePattern<VariableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VariableOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.initializer()) return failure();
    auto *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(op, op.initializer().getValue());
    auto initializer = cast<FuncOp>(symbolOp);
    if (initializer.getBlocks().size() == 1 &&
        initializer.getBlocks().front().getOperations().size() == 2 &&
        isa<mlir::ReturnOp>(
            initializer.getBlocks().front().getOperations().back())) {
      auto &primaryOp = initializer.getBlocks().front().getOperations().front();
      Attribute constResult;
      if (matchPattern(primaryOp.getResult(0), m_Constant(&constResult))) {
        auto newOp = rewriter.create<VariableOp>(op.getLoc(), op.sym_name(),
                                                 op.is_mutable(), op.type(),
                                                 constResult);
        SymbolTable::setSymbolVisibility(newOp,
                                         SymbolTable::getSymbolVisibility(op));
        rewriter.replaceOp(op, {});
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void VariableOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<InlineConstVariableOpInitializer>(context);
}

namespace {

class PropagateVariableLoadAddress
    : public OpRewritePattern<VariableLoadIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(VariableLoadIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<VariableAddressOp>(
            op.variable().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<VariableLoadOp>(op, op.result().getType(),
                                                  addressOp.variable());
      return success();
    }
    return failure();
  }
};

}  // namespace

void VariableLoadIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateVariableLoadAddress>(context);
}

namespace {

/// Erases hal.variable.store ops that are no-ops.
/// This can happen if there was a variable load, some DCE'd usage, and a
/// store back to the same variable: we want to be able to elide the entire load
/// and store.
struct EraseUnusedVariableStoreOp : public OpRewritePattern<VariableStoreOp> {
  using OpRewritePattern<VariableStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VariableStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto loadOp =
            dyn_cast_or_null<VariableLoadOp>(op.value().getDefiningOp())) {
      if (loadOp.variable() == op.variable()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void VariableStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedVariableStoreOp>(context);
}

namespace {

class PropagateVariableStoreAddress
    : public OpRewritePattern<VariableStoreIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(VariableStoreIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<VariableAddressOp>(
            op.variable().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<VariableStoreOp>(op, op.value(),
                                                   addressOp.variable());
      return success();
    }
    return failure();
  }
};

}  // namespace

void VariableStoreIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateVariableStoreAddress>(context);
}

//===----------------------------------------------------------------------===//
// hal.allocator.*
//===----------------------------------------------------------------------===//

// Computes the element count of a possibly-dynamic shaped tensor.
static Value getElementCount(Location loc, Value baseValue,
                             ValueRange shapeDims, OpBuilder &builder) {
  Value value = baseValue;
  for (auto dim : shapeDims) {
    value = builder.createOrFold<mlir::MulIOp>(loc, value, dim);
  }
  return value;
}

namespace {

/// Expands hal.allocator.compute_size to IR performing the math.
struct ExpandAllocatorComputeSizeOp
    : public OpRewritePattern<AllocatorComputeSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocatorComputeSizeOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): use buffer constraints for alignment.
    BufferConstraintsAdaptor bufferConstraints(op.getLoc(), op.allocator());

    auto elementSize = rewriter.createOrFold<mlir::ConstantIndexOp>(
        op.getLoc(), getElementByteCount(op.element_typeAttr()));
    auto byteSize =
        getElementCount(op.getLoc(), elementSize, op.shape(), rewriter);

    rewriter.replaceOp(op, {byteSize});
    return success();
  }
};

}  // namespace

void AllocatorComputeSizeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandAllocatorComputeSizeOp>(context);
}

namespace {

/// Expands hal.allocator.compute_offset to IR performing the math.
struct ExpandAllocatorComputeOffsetOp
    : public OpRewritePattern<AllocatorComputeOffsetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocatorComputeOffsetOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): use buffer constraints.
    BufferConstraintsAdaptor bufferConstraints(op.getLoc(), op.allocator());

    auto offset = rewriter.createOrFold<mlir::ConstantIndexOp>(op.getLoc(), 0);
    for (size_t i = 0; i < op.indices().size(); ++i) {
      // TODO(benvanik): check error case in debug builds.
      // if (indices[i] >= shape[i]) {
      //   return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
      //                           "index[%zu] out of bounds: %d >= %d", i,
      //                           indices[i], shape[i]);
      // }
      auto axisOffset = op.indices()[i];
      for (size_t j = i + 1; j < op.shape().size(); ++j) {
        axisOffset = rewriter.createOrFold<mlir::MulIOp>(
            op.getLoc(), axisOffset, op.shape()[j]);
      }
      offset =
          rewriter.createOrFold<mlir::AddIOp>(op.getLoc(), offset, axisOffset);
    }
    auto elementSize = rewriter.createOrFold<mlir::ConstantIndexOp>(
        op.getLoc(), getElementByteCount(op.element_typeAttr()));
    auto byteOffset =
        rewriter.createOrFold<mlir::MulIOp>(op.getLoc(), offset, elementSize);

    rewriter.replaceOp(op, {byteOffset});
    return success();
  }
};

}  // namespace

void AllocatorComputeOffsetOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandAllocatorComputeOffsetOp>(context);
}

namespace {

/// Expands hal.allocator.compute_range to IR performing the math.
struct ExpandAllocatorComputeRangeOp
    : public OpRewritePattern<AllocatorComputeRangeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocatorComputeRangeOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): use buffer constraints.
    BufferConstraintsAdaptor bufferConstraints(op.getLoc(), op.allocator());

    SmallVector<Value, 6> endIndices(op.shape().size());
    auto one = rewriter.createOrFold<mlir::ConstantIndexOp>(op.getLoc(), 1);
    for (size_t i = 0; i < endIndices.size(); ++i) {
      endIndices[i] = rewriter.createOrFold<mlir::SubIOp>(
          op.getLoc(),
          rewriter.createOrFold<mlir::AddIOp>(op.getLoc(), op.indices()[i],
                                              op.lengths()[i]),
          one);
    }

    auto startByteOffset = rewriter.createOrFold<AllocatorComputeOffsetOp>(
        op.getLoc(), rewriter.getIndexType(), op.allocator(), op.shape(),
        op.element_typeAttr(), op.indices());
    auto endByteOffset = rewriter.createOrFold<AllocatorComputeOffsetOp>(
        op.getLoc(), rewriter.getIndexType(), op.allocator(), op.shape(),
        op.element_typeAttr(), endIndices);

    auto elementSize = rewriter.createOrFold<mlir::ConstantIndexOp>(
        op.getLoc(), getElementByteCount(op.element_typeAttr()));
    auto offsetLength = rewriter.createOrFold<mlir::AddIOp>(
        op.getLoc(),
        rewriter.createOrFold<mlir::SubIOp>(op.getLoc(), endByteOffset,
                                            startByteOffset),
        elementSize);

    rewriter.replaceOp(op, {startByteOffset, offsetLength});
    return success();
  }
};

}  // namespace

void AllocatorComputeRangeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandAllocatorComputeRangeOp>(context);
}

namespace {

/// Expands hal.allocator.allocate.const to an allocation and data write.
struct ExpandAllocatorAllocateConstOp
    : public OpRewritePattern<AllocatorAllocateConstOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocatorAllocateConstOp op,
                                PatternRewriter &rewriter) const override {
    auto hostBuffer = rewriter.createOrFold<IREE::ByteBufferConstantOp>(
        op.getLoc(), IREE::ByteBufferType::get(rewriter.getContext()),
        op.value());
    auto zero = rewriter.createOrFold<mlir::ConstantIndexOp>(op.getLoc(), 0);
    auto neg1 = rewriter.createOrFold<mlir::ConstantIndexOp>(op.getLoc(), -1);
    auto deviceBuffer = rewriter.createOrFold<AllocatorMapOp>(
        op.getLoc(), op.allocator(), op.memory_types(), op.buffer_usage(),
        hostBuffer, zero, neg1);
    rewriter.replaceOp(op, {deviceBuffer});
    return success();
  }
};

}  // namespace

void AllocatorAllocateConstOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandAllocatorAllocateConstOp>(context);
}

//===----------------------------------------------------------------------===//
// hal.buffer.*
//===----------------------------------------------------------------------===//

namespace {

/// Skips a hal.buffer.allocator accessor when the buffer view was created in
/// the same scope and we know the origin buffer.
struct SkipBufferAllocatorOp : public OpRewritePattern<BufferAllocatorOp> {
  using OpRewritePattern<BufferAllocatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferAllocatorOp op,
                                PatternRewriter &rewriter) const override {
    if (auto allocateOp = dyn_cast_or_null<AllocatorAllocateOp>(
            op.buffer().getDefiningOp())) {
      rewriter.replaceOp(op, allocateOp.allocator());
      return success();
    } else if (auto allocateOp = dyn_cast_or_null<AllocatorAllocateConstOp>(
                   op.buffer().getDefiningOp())) {
      rewriter.replaceOp(op, allocateOp.allocator());
      return success();
    } else if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
                   op.buffer().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<BufferAllocatorOp>(op,
                                                     subspanOp.source_buffer());
      return success();
    }
    return failure();
  }
};

}  // namespace

void BufferAllocatorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SkipBufferAllocatorOp>(context);
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.*
//===----------------------------------------------------------------------===//

namespace {

/// Expands hal.buffer_view.const to an allocation and buffer view wrapper.
struct ExpandBufferViewConstOp : public OpRewritePattern<BufferViewConstOp> {
  using OpRewritePattern<BufferViewConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferViewConstOp op,
                                PatternRewriter &rewriter) const override {
    auto shapedType = op.value().getType();
    auto elementType = getElementTypeValue(shapedType.getElementType());
    if (!elementType.hasValue()) {
      return failure();
    }

    auto buffer = rewriter.createOrFold<AllocatorAllocateConstOp>(
        op.getLoc(), op.allocator(), op.memory_types(), op.buffer_usage(),
        op.value());

    SmallVector<Value, 4> shape;
    if (shapedType.getRank() >= 1) {
      for (auto dim : shapedType.getShape()) {
        shape.push_back(
            rewriter.createOrFold<mlir::ConstantIndexOp>(op.getLoc(), dim));
      }
    }

    rewriter.replaceOpWithNewOp<BufferViewCreateOp>(op, buffer, shape,
                                                    elementType.getValue());
    return success();
  }
};

}  // namespace

void BufferViewConstOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandBufferViewConstOp>(context);
}

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

void BufferViewBufferOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
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
            op.command_buffer().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.device());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CommandBufferDeviceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SkipCommandBufferDeviceOp>(context);
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
        bindingOffsets[i] = rewriter.createOrFold<mlir::AddIOp>(
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
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldCommandBufferPushDescriptorSetBufferSubspan>(context);
}

//===----------------------------------------------------------------------===//
// hal.constant_pool.*
//===----------------------------------------------------------------------===//

namespace {

// Resolves hal.constant.buffer ops to their runtime hal.variable buffer.
struct ResolveConstantPoolLoadToRuntimeBuffer
    : public OpRewritePattern<ConstantPoolLoadOp> {
  using OpRewritePattern<ConstantPoolLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantPoolLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto *constOp = SymbolTable::lookupNearestSymbolFrom(op, op.constant());
    SymbolRefAttr runtimeBufferSymRef;
    ByteRangeAttr runtimeBufferRange;
    if (auto spanOp = dyn_cast<ConstantPoolSpanOp>(constOp)) {
      runtimeBufferSymRef = spanOp.runtime_bufferAttr();
      runtimeBufferRange = spanOp.runtime_rangeAttr();
    } else if (auto splatOp = dyn_cast<ConstantPoolSplatOp>(constOp)) {
      runtimeBufferSymRef = splatOp.runtime_bufferAttr();
      runtimeBufferRange = splatOp.runtime_rangeAttr();
    }
    if (!runtimeBufferSymRef || !runtimeBufferRange) return failure();
    rewriter.replaceOpWithNewOp<IREE::HAL::ConstantSubspanOp>(
        op, op.getType(), runtimeBufferSymRef, runtimeBufferRange);
    return success();
  }
};

}  // namespace

void ConstantPoolLoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ResolveConstantPoolLoadToRuntimeBuffer>(context);
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
