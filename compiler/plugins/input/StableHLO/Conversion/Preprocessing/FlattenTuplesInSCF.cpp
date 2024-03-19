// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific preprocessing for XLA inputs.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_FLATTENTUPLESINSCF
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {
// Given a set of types, unpack to a list of a types, removing all tuples.
void untupleTypes(TypeRange types, llvm::SmallVectorImpl<Type> &newTypes) {
  for (Type type : types) {
    if (auto tupleTy = dyn_cast<TupleType>(type)) {
      untupleTypes(tupleTy.getTypes(), newTypes);
    } else {
      newTypes.push_back(type);
    }
  }
}

void recursiveUntuple(Value value, ImplicitLocOpBuilder b, IRMapping &mapping,
                      llvm::SmallVectorImpl<Value> &newValues) {
  auto tupleType = dyn_cast<TupleType>(value.getType());
  if (!tupleType) {
    // We can return the value as is.
    newValues.push_back(value);
    return;
  }

  for (auto [idx, subType] : llvm::enumerate(tupleType.getTypes())) {
    auto elementOp = b.create<mlir::stablehlo::GetTupleElementOp>(
        subType, value, b.getI32IntegerAttr(idx));
    recursiveUntuple(elementOp.getResult(), b, mapping, newValues);
  }
}

Value recursiveRetuple(Type oldType, ArrayRef<Value> *values,
                       ImplicitLocOpBuilder &b) {
  auto tupleType = dyn_cast<TupleType>(oldType);
  if (!tupleType) {
    Value returnValue = *values->begin();
    *values = {values->begin() + 1, values->end()};
    return returnValue;
  }

  llvm::SmallVector<Value> subValues;
  for (Type subType : tupleType.getTypes()) {
    subValues.push_back(recursiveRetuple(subType, values, b));
  }

  return b.create<mlir::stablehlo::TupleOp>(tupleType, subValues).getResult();
}

void DetupleRegion(Region &srcRegion, Region &destRegion, ArrayRef<Type> types,
                   IRMapping &mapping, ImplicitLocOpBuilder &b) {
  auto beforeB = b.createBlock(&destRegion);
  beforeB->addArguments(types, SmallVector<Location>(types.size(), b.getLoc()));
  b.setInsertionPointToStart(beforeB);

  llvm::SmallVector<Value> beforeResultVals;
  for (auto argument : beforeB->getArguments()) {
    beforeResultVals.push_back(argument);
  }

  llvm::ArrayRef<Value> beforeResults(beforeResultVals);
  for (auto oldResult : srcRegion.front().getArguments()) {
    auto newResult = recursiveRetuple(oldResult.getType(), &beforeResults, b);
    mapping.map(oldResult, newResult);
  }

  b.setInsertionPointToEnd(&destRegion.front());
  for (auto &srcop : srcRegion.front()) {
    auto cloned = b.clone(srcop, mapping);
    for (int i = 0; i < cloned->getNumResults(); i++) {
      mapping.map(srcop.getResult(i), cloned->getResult(i));
    }
  }
}

class DetupleYieldOp : public OpRewritePattern<scf::YieldOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::YieldOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    bool hasTuples = false;
    IRMapping mapping;

    llvm::SmallVector<Value> operands;
    for (auto operand : op.getOperands()) {
      hasTuples |= isa<TupleType>(operand.getType());
      recursiveUntuple(operand, b, mapping, operands);
    }

    if (!hasTuples)
      return rewriter.notifyMatchFailure(op, "no tupled arguments");

    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, operands);
    return success();
  }
};

class DetupleConditionOp : public OpRewritePattern<scf::ConditionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ConditionOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    bool hasTuples = false;
    IRMapping mapping;

    llvm::SmallVector<Value> operands;
    for (auto operand : op.getArgs()) {
      hasTuples |= isa<TupleType>(operand.getType());
      recursiveUntuple(operand, b, mapping, operands);
    }

    if (!hasTuples)
      return rewriter.notifyMatchFailure(op, "no tupled arguments");

    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(op, op.getCondition(),
                                                        operands);
    return success();
  }
};

class DetupleIfOp : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    bool hasTuples = false;
    IRMapping mapping;

    for (auto type : op.getResultTypes()) {
      hasTuples |= isa<TupleType>(type);
    }

    if (!hasTuples)
      return rewriter.notifyMatchFailure(op, "no tupled arguments");

    llvm::SmallVector<Type> types;
    untupleTypes(op.getResultTypes(), types);

    auto newOp = b.create<mlir::scf::IfOp>(types, op.getOperand());

    DetupleRegion(op.getThenRegion(), newOp.getThenRegion(), {}, mapping, b);
    DetupleRegion(op.getElseRegion(), newOp.getElseRegion(), {}, mapping, b);

    b.setInsertionPoint(op);
    llvm::SmallVector<Value> newResultVals;
    for (auto result : newOp.getResults()) {
      newResultVals.push_back(result);
    }

    llvm::ArrayRef<Value> newResults(newResultVals);
    llvm::SmallVector<Value, 10> retupledValues;
    for (auto oldResult : op.getResults()) {
      auto newResult = recursiveRetuple(oldResult.getType(), &newResults, b);
      retupledValues.push_back(newResult);
      mapping.map(oldResult, newResult);
    }

    rewriter.replaceOp(op, retupledValues);
    return success();
  }
};

class DetupleWhileOp : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    bool hasTuples = false;
    IRMapping mapping;

    llvm::SmallVector<Value> operands;
    for (auto operand : op.getOperands()) {
      hasTuples |= isa<TupleType>(operand.getType());
      recursiveUntuple(operand, b, mapping, operands);
    }

    if (!hasTuples)
      return rewriter.notifyMatchFailure(op, "no tupled arguments");

    llvm::SmallVector<Type> types;
    untupleTypes(op.getResultTypes(), types);

    auto newOp = b.create<mlir::scf::WhileOp>(types, operands);

    DetupleRegion(op.getBefore(), newOp.getBefore(), types, mapping, b);
    DetupleRegion(op.getAfter(), newOp.getAfter(), types, mapping, b);

    b.setInsertionPoint(op);
    llvm::SmallVector<Value> newResultVals;
    for (auto result : newOp.getResults()) {
      newResultVals.push_back(result);
    }

    llvm::ArrayRef<Value> newResults(newResultVals);
    llvm::SmallVector<Value, 10> retupledValues;
    for (auto oldResult : op.getResults()) {
      auto newResult = recursiveRetuple(oldResult.getType(), &newResults, b);
      retupledValues.push_back(newResult);
      mapping.map(oldResult, newResult);
    }

    rewriter.replaceOp(op, retupledValues);
    return success();
  }
};

struct FlattenTuplesInSCF final
    : impl::FlattenTuplesInSCFBase<FlattenTuplesInSCF> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Builder b(ctx);

    // Run canonicalization patterns to cancel out remaining tuple ops. We need
    // to run these manually here because StableHLO does not define
    // folds/canonicalization patterns for its ops.
    RewritePatternSet patterns(ctx);
    populateCanonicalizationPatterns(ctx, &patterns);
    patterns
        .add<DetupleYieldOp, DetupleConditionOp, DetupleIfOp, DetupleWhileOp>(
            ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
