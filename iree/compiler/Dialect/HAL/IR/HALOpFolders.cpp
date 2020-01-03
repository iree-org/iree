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
#include "mlir/Dialect/StandardOps/Ops.h"
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

  PatternMatchResult matchAndRewrite(VariableOp op,
                                     PatternRewriter &rewriter) const override {
    if (!op.initializer()) return matchFailure();
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
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

}  // namespace

void VariableOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<InlineConstVariableOpInitializer>(context);
}

namespace {

/// Erases hal.variable.load ops whose values are unused.
/// We have to do this manually as the load op cannot be marked pure and have it
/// done automatically.
struct EraseUnusedVariableLoadOp : public OpRewritePattern<VariableLoadOp> {
  using OpRewritePattern<VariableLoadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(VariableLoadOp op,
                                     PatternRewriter &rewriter) const override {
    if (op.result().use_empty()) {
      rewriter.eraseOp(op);
      return matchSuccess();
    }
    return matchFailure();
  }
};

}  // namespace

void VariableLoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedVariableLoadOp>(context);
}

namespace {

/// Erases hal.variable.store ops that are no-ops.
/// This can happen if there was a variable load, some DCE'd usage, and a
/// store back to the same variable: we want to be able to elide the entire load
/// and store.
struct EraseUnusedVariableStoreOp : public OpRewritePattern<VariableStoreOp> {
  using OpRewritePattern<VariableStoreOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(VariableStoreOp op,
                                     PatternRewriter &rewriter) const override {
    if (auto loadOp =
            dyn_cast_or_null<VariableLoadOp>(op.value().getDefiningOp())) {
      if (loadOp.variable() == op.variable()) {
        rewriter.eraseOp(op);
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

}  // namespace

void VariableStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedVariableStoreOp>(context);
}

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

namespace {

/// Simplifies a hal.allocator.compute_size + hal.allocator.allocate pair into
/// a single hal.allocator.allocate.shaped when there are no other paired
/// allocations.
struct SimplifyAllocatorAllocateShapedOp
    : public OpRewritePattern<AllocatorAllocateOp> {
  using OpRewritePattern<AllocatorAllocateOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AllocatorAllocateOp op,
                                     PatternRewriter &rewriter) const override {
    if (auto computeSizeOp = dyn_cast_or_null<AllocatorComputeSizeOp>(
            op.allocation_size().getDefiningOp())) {
      if (op.memory_types() == computeSizeOp.memory_types() &&
          op.buffer_usage() == computeSizeOp.buffer_usage()) {
        rewriter.replaceOpWithNewOp<AllocatorAllocateShapedOp>(
            op, op.allocator(), op.memory_types(), op.buffer_usage(),
            computeSizeOp.shape(), computeSizeOp.element_size().getZExtValue());
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

}  // namespace

void AllocatorAllocateOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAllocatorAllocateShapedOp>(context);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
