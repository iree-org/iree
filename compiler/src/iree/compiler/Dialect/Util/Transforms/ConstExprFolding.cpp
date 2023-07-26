// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {

void addEscapingInputsToParent(Location loc, SmallVector<Value> &parentOperands,
                               Block *block, IRMapping mapping, Operation *op) {
  for (Value input : op->getOperands()) {
    if (mapping.contains(input))
      continue;
    BlockArgument blockArg = block->addArgument(input.getType(), loc);
    parentOperands.push_back(input);
    mapping.map(input, blockArg);
  }
}

ConstExprOp formSingleRootConstExprOp(OpBuilder &builder, Operation *rootOp,
                                      SmallVector<Operation *> producers) {
  OpBuilder::InsertionGuard guard(builder);
  auto constExprOp = builder.create<ConstExprOp>(
      rootOp->getLoc(), rootOp->getResultTypes(), ValueRange{});
  SmallVector<Value> constExprOperands;
  Block *body = builder.createBlock(&constExprOp.getBody());

  IRMapping mapping;
  mlir::computeTopologicalSorting(producers);
  for (Operation *producer : producers) {
    addEscapingInputsToParent(producer->getLoc(), constExprOperands, body,
                              mapping, producer);
    builder.clone(*producer, mapping);
  }

  // Move the root operation into the constexpr.
  addEscapingInputsToParent(rootOp->getLoc(), constExprOperands, body, mapping,
                            rootOp);
  Operation *newRoot = builder.clone(*rootOp, mapping);
  builder.create<YieldConstOp>(rootOp->getLoc(), TypeRange{},
                               newRoot->getResults());

  constExprOp->setOperands(constExprOperands);
  return constExprOp;
}

class ConstExprLinalgOps : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  ConstExprLinalgOps(MLIRContext *context, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (linalgOp->getParentOfType<ConstExprOp>())
      return failure();

    // Mixed and buffer sematics aren't supported.
    if (!linalgOp.hasTensorSemantics())
      return failure();

    // Never constexpr fold certain ops on their own.
    if (llvm::isa<linalg::FillOp>(linalgOp)) {
      return failure();
    }

    // Collect constant operands, failing if any are not const.
    SmallVector<Operation *> outlineOps;
    for (const OpOperand *opOperand : linalgOp.getDpsInputOperands()) {
      if (!consumeConstantProducer(opOperand->get(), outlineOps)) {
        return failure();
      }
    }
    for (const OpOperand *opOperand : linalgOp.getDpsInitOperands()) {
      if (!consumeConstantProducer(opOperand->get(), outlineOps)) {
        return failure();
      }
    }

    auto constExprOp =
        formSingleRootConstExprOp(rewriter, linalgOp, outlineOps);
    rewriter.replaceOp(linalgOp, constExprOp);
    dceUnusedOps(rewriter, outlineOps);

    return success();
  }

  void dceUnusedOps(PatternRewriter &rewriter,
                    SmallVector<Operation *> liveOps) const {
    bool modified;
    SmallVector<Operation *> worklist;
    do {
      modified = false;
      worklist.swap(liveOps);
      liveOps.clear();
      for (Operation *op : worklist) {
        if (op->use_empty()) {
          rewriter.eraseOp(op);
          modified = true;
        } else {
          liveOps.push_back(op);
        }
      }
    } while (modified);
  }

  bool consumeConstantProducer(Value producer,
                               SmallVector<Operation *> &outlineOps) const {
    Operation *producerOp = producer.getDefiningOp();
    if (!producerOp) {
      // Not constant.
      return false;
    }

    // Allow one-use, free-standing constant producers. We do
    // not consider multi-use in this simple folding heuristic
    // because it is tricky to balance and we cannot be 100%
    // certain locally that it is profitable to fold.
    if (producerOp->hasTrait<OpTrait::ConstantLike>()) {
      if (!producerOp->getResult(0).hasOneUse()) {
        return false;
      }
      outlineOps.push_back(producerOp);
      return true;
    }

    // Allow util.constexpr but do not (nested) outline it: Leave
    // it for general constexpr folding.
    if (llvm::isa<IREE::Util::ConstExprOp>(producerOp)) {
      return true;
    }

    // Always outline tensor.empty ops.
    if (llvm::isa<tensor::EmptyOp>(producerOp)) {
      outlineOps.push_back(producerOp);
      return true;
    }

    return false;
  }
};

class TestTrivialConstExprFoldingPass
    : public TestTrivialConstExprFoldingBase<TestTrivialConstExprFoldingPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<UtilDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateTrivialLinalgConstexprFoldingOperations(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "FAILURE!\n";
      return signalPassFailure();
    }
  }
};

} // namespace

void populateTrivialLinalgConstexprFoldingOperations(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ConstExprLinalgOps>(context);
}

std::unique_ptr<OperationPass<void>> createTestTrivialConstExprFoldingPass() {
  return std::make_unique<TestTrivialConstExprFoldingPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
