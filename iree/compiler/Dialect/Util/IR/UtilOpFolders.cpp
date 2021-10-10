// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

//===----------------------------------------------------------------------===//
// util.cmp.eq
//===----------------------------------------------------------------------===//

OpFoldResult CmpEQOp::fold(ArrayRef<Attribute> operands) {
  auto makeBool = [&](bool value) {
    return IntegerAttr::get(IntegerType::get(getContext(), 1), value ? 1 : 0);
  };
  if (lhs() == rhs()) {
    // SSA values are exactly the same.
    return makeBool(true);
  } else if (operands[0] && operands[1] && operands[0] == operands[1]) {
    // Folded attributes are equal but may come from separate ops.
    return makeBool(true);
  }
  // TODO(benvanik): we could add some interfaces for comparing, but this is
  // likely good enough for now.
  return {};
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

namespace {

// Deletes empty vm.initializer ops.
struct DropEmptyInitializerOp : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    if (op.body().getBlocks().size() != 1) return failure();
    auto &block = op.body().front();
    if (block.empty() || isa<InitializerReturnOp>(block.front())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Inlines constant stores from initializers into the global initializer.
// This is not strictly required but can help our initialization code perform
// more efficient initialization of large numbers of primitive values.
struct InlineConstantGlobalInitializer
    : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> deadOps;
    op.walk([&](GlobalStoreOp storeOp) {
      Attribute valueAttr;
      if (!matchPattern(storeOp.value(), m_Constant(&valueAttr))) return;
      auto globalOp = storeOp.getGlobalOp();
      rewriter.updateRootInPlace(globalOp, [&]() {
        if (valueAttr && !valueAttr.isa<UnitAttr>()) {
          globalOp.initial_valueAttr(valueAttr);
        } else {
          globalOp.clearInitialValue();
        }
      });
      deadOps.push_back(storeOp);
    });
    if (deadOps.empty()) return failure();
    for (auto deadOp : deadOps) rewriter.eraseOp(deadOp);
    return success();
  }
};

}  // namespace

void InitializerOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<DropEmptyInitializerOp, InlineConstantGlobalInitializer>(
      context);
}

void GlobalOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {}

namespace {

/// Turns util.global.address -> util.global.load.indirect into a direct load.
class PropagateGlobalLoadAddress
    : public OpRewritePattern<GlobalLoadIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GlobalLoadIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp =
            dyn_cast_or_null<GlobalAddressOp>(op.global().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<GlobalLoadOp>(op, op.result().getType(),
                                                addressOp.global());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalLoadIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateGlobalLoadAddress>(context);
}

namespace {

/// Erases util.global.store ops that are no-ops.
/// This can happen if there was a global load, some DCE'd usage, and a
/// store back to the same global: we want to be able to elide the entire load
/// and store.
struct EraseUnusedGlobalStoreOp : public OpRewritePattern<GlobalStoreOp> {
  using OpRewritePattern<GlobalStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto loadOp =
            dyn_cast_or_null<GlobalLoadOp>(op.value().getDefiningOp())) {
      if (loadOp.global() == op.global()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void GlobalStoreOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<EraseUnusedGlobalStoreOp>(context);
}

namespace {

/// Turns util.global.address -> util.global.store.indirect into a direct store.
class PropagateGlobalStoreAddress
    : public OpRewritePattern<GlobalStoreIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GlobalStoreIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp =
            dyn_cast_or_null<GlobalAddressOp>(op.global().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<GlobalStoreOp>(op, op.value(),
                                                 addressOp.global());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalStoreIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PropagateGlobalStoreAddress>(context);
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
