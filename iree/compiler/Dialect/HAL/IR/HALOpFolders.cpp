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
// Variables
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
        rewriter.replaceOpWithNewOp<VariableOp>(
            op, op.sym_name(), op.is_mutable(), op.type(), constResult);
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
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

namespace {

/// Skips a hal.buffer_view.buffer accessor when the buffer view was created in
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
// iree::hal::BufferView
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
        shape.push_back(rewriter.createOrFold<mlir::ConstantOp>(
            op.getLoc(),
            rewriter.getI32IntegerAttr(static_cast<int32_t>(dim))));
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

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
