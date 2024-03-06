// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO dialect ops to the SCF dialect.

#include "compiler/plugins/input/StableHLO/stablehlo-iree/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/stablehlo-iree/Conversion/Rewriters.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZECONTROLFLOW
#include "compiler/plugins/input/StableHLO/stablehlo-iree/Conversion/Passes.h.inc"

namespace {
// All transformations in this file take stablehlo blocks which end with
// stablehlo::ReturnOp and lower to SCF ops which end with scf::YieldOp. Inline
// an entire block with the only change being return -> yield.
void inlineStableHloRegionIntoSCFRegion(PatternRewriter &rewriter, Region &hlo,
                                        Region &scf) {
  // Remove an existing block, then move the region over.
  if (!scf.empty()) {
    rewriter.eraseBlock(&scf.back());
  }
  rewriter.inlineRegionBefore(hlo, scf, scf.end());
  // Fix up the terminator.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&scf.back());
  Operation *terminator = scf.back().getTerminator();
  rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator,
                                            terminator->getOperands());
}

// stablehlo ops need inputs to be tensors, but scalar values can be a scalar
// tensor or a 1 element tensor. To handle this, collapse shape before
// extracting the scalar value when necessary.
Value extractTensorValue(OpBuilder &b, Value tensor) {
  Location loc = tensor.getLoc();
  if (auto rankedTy = dyn_cast<RankedTensorType>(tensor.getType())) {
    if (rankedTy.getRank() != 0) {
      tensor = b.create<tensor::CollapseShapeOp>(
          loc, tensor, SmallVector<ReassociationIndices>());
    }
  }

  return b.create<tensor::ExtractOp>(loc, tensor, ValueRange());
}

struct ScfForBounds {
  Value lb;
  Value ub;
  Value step;
  unsigned indexArgIndex;
};

std::optional<ScfForBounds> extractForBounds(mlir::stablehlo::WhileOp op) {
  Block &cond = op.getCond().front();
  Block &body = op.getBody().front();
  if (cond.getOperations().size() != 2)
    return std::nullopt;

  auto matchBbArg = [](Value v, Block &block) -> std::optional<unsigned> {
    if (!isa<BlockArgument>(v) || v.getParentBlock() != &block)
      return std::nullopt;
    return cast<BlockArgument>(v).getArgNumber();
  };

  auto compare = dyn_cast<mlir::stablehlo::CompareOp>(cond.front());
  // If the rhs of the comapare is defined outside the block, it's a constant
  // within the loop.
  if (!compare ||
      compare.getComparisonDirection() !=
          mlir::stablehlo::ComparisonDirection::LT ||
      compare.getRhs().getParentBlock() == &cond ||
      !getElementTypeOrSelf(compare.getLhs().getType())
           .isSignlessIntOrIndex()) {
    return std::nullopt;
  }

  std::optional<unsigned> iterArg = matchBbArg(compare.getLhs(), cond);
  if (!iterArg)
    return std::nullopt;

  auto add = dyn_cast_or_null<mlir::stablehlo::AddOp>(
      body.getTerminator()->getOperand(*iterArg).getDefiningOp());
  if (!add || matchBbArg(add.getLhs(), body) != iterArg ||
      add.getRhs().getParentBlock() == &body) {
    return std::nullopt;
  }

  ScfForBounds bounds;
  bounds.ub = compare.getRhs();
  bounds.step = add.getRhs();
  bounds.lb = op->getOperand(*iterArg);
  bounds.indexArgIndex = *iterArg;
  return bounds;
}

// Rewrites `stablehlo.while` to `scf.while` or `scf.for`.
struct WhileOpPattern final : OpConversionPattern<mlir::stablehlo::WhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (std::optional<ScfForBounds> bounds = extractForBounds(op)) {
      auto newForOp = rewriter.create<scf::ForOp>(
          loc, extractTensorValue(rewriter, bounds->lb),
          extractTensorValue(rewriter, bounds->ub),
          extractTensorValue(rewriter, bounds->step), adaptor.getOperands());

      rewriter.setInsertionPointToEnd(newForOp.getBody());
      // Inline while body, and only replace the stablehlo.return with an
      // scf.yield.
      inlineStableHloRegionIntoSCFRegion(rewriter, op.getBody(),
                                         newForOp.getRegion());
      BlockArgument indexArg = newForOp.getRegion().insertArgument(
          unsigned{0}, newForOp.getLowerBound().getType(), loc);
      BlockArgument oldIndexArg =
          newForOp.getRegion().getArgument(1 + bounds->indexArgIndex);
      rewriter.setInsertionPointToStart(&newForOp.getRegion().front());
      auto indexArgTensor = rewriter.create<tensor::FromElementsOp>(
          loc, oldIndexArg.getType(), indexArg);
      oldIndexArg.replaceAllUsesWith(indexArgTensor);

      rewriter.replaceOp(op, newForOp.getResults());
      return success();
    }

    auto newWhileOp = rewriter.create<scf::WhileOp>(loc, op.getResultTypes(),
                                                    adaptor.getOperands());

    // Inline while condition. The block is the same, except the boolean result
    // needs to be extracted and used with an scf.condition.
    rewriter.inlineRegionBefore(op.getCond(), newWhileOp.getBefore(),
                                newWhileOp.getBefore().end());
    auto conditionReturn = cast<mlir::stablehlo::ReturnOp>(
        newWhileOp.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&newWhileOp.getBefore().front());
    Value i1 = extractTensorValue(rewriter, conditionReturn->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        conditionReturn, i1, newWhileOp.getBeforeArguments());

    // Inline while body, and only replace the stablehlo.return with an
    // scf.yield.
    inlineStableHloRegionIntoSCFRegion(rewriter, op.getBody(),
                                       newWhileOp.getAfter());

    rewriter.replaceOp(op, newWhileOp.getResults());
    return success();
  }
};

// Rewrites `stablehlo.if` to `scf.if`.
struct IfOpPattern final : OpConversionPattern<mlir::stablehlo::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto scfIf = rewriter.create<scf::IfOp>(
        op.getLoc(), op.getResultTypes(),
        extractTensorValue(rewriter, adaptor.getPred()),
        /*withElseRegion=*/true);
    inlineStableHloRegionIntoSCFRegion(rewriter, op.getTrueBranch(),
                                       scfIf.getThenRegion());
    inlineStableHloRegionIntoSCFRegion(rewriter, op.getFalseBranch(),
                                       scfIf.getElseRegion());
    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};

// Rewrites `stablehlo.case` to a nested `scf.if`.
struct CaseOpPattern : public OpConversionPattern<mlir::stablehlo::CaseOp> {
  using OpConversionPattern::OpConversionPattern;

  // Recursively create if/else ops to handle each possible value in a case op.
  scf::IfOp createNestedCases(int currentIdx, mlir::stablehlo::CaseOp op,
                              OpAdaptor adaptor,
                              PatternRewriter &outerBuilder) const {
    Location loc = op.getLoc();
    Value idxValue = adaptor.getIndex();
    size_t finalIdx = op.getBranches().size() - 2;

    // Determine if the current index matches the case index.
    Type scalarType = idxValue.getType();
    auto shapedType = cast<ShapedType>(scalarType);
    auto constAttr = DenseElementsAttr::get(
        shapedType,
        {cast<Attribute>(outerBuilder.getI32IntegerAttr(currentIdx))});
    Value currentIdxVal = outerBuilder.create<mlir::stablehlo::ConstantOp>(
        loc, idxValue.getType(), constAttr);

    auto scfIf = outerBuilder.create<scf::IfOp>(
        loc, op.getResultTypes(),
        extractTensorValue(outerBuilder,
                           outerBuilder.create<mlir::stablehlo::CompareOp>(
                               loc, idxValue, currentIdxVal,
                               mlir::stablehlo::ComparisonDirection::EQ)),
        /*withElseRegion=*/true);
    inlineStableHloRegionIntoSCFRegion(
        outerBuilder, op.getBranches()[currentIdx], scfIf.getThenRegion());
    int nextIdx = currentIdx + 1;
    // Don't recurse for the final default block.
    if (currentIdx == static_cast<int64_t>(finalIdx)) {
      inlineStableHloRegionIntoSCFRegion(
          outerBuilder, op.getBranches()[nextIdx], scfIf.getElseRegion());
      return scfIf;
    }

    PatternRewriter::InsertionGuard guard(outerBuilder);
    outerBuilder.setInsertionPointToEnd(&scfIf.getElseRegion().back());
    auto innerIf = createNestedCases(nextIdx, op, adaptor, outerBuilder);
    outerBuilder.create<scf::YieldOp>(op.getLoc(), innerIf.getResults());
    return scfIf;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the op if there is only a default block.
    if (op.getBranches().size() == 1) {
      Block &block = op.getBranches().front().front();
      OperandRange results = block.getTerminator()->getOperands();
      // Remove the stablehlo.return terminator, then inline the block.
      rewriter.eraseOp(block.getTerminator());
      rewriter.inlineBlockBefore(/*source=*/&block, /*dest=*/op.getOperation(),
                                 /*argValues=*/{});
      rewriter.replaceOp(op, results);
      return success();
    }

    // Begin recursion with case 0.
    rewriter.replaceOp(
        op, createNestedCases(0, op, adaptor, rewriter).getResults());
    return success();
  }
};

struct LegalizeControlFlow final
    : impl::LegalizeControlFlowBase<LegalizeControlFlow> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto f = getOperation();
    MLIRContext *ctx = f.getContext();

    RewritePatternSet patterns(ctx);
    populateLegalizeControlFlowPatterns(ctx, &patterns);

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<mlir::stablehlo::IfOp, mlir::stablehlo::WhileOp,
                        mlir::stablehlo::CaseOp>();

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

void populateLegalizeControlFlowPatterns(MLIRContext *context,
                                         RewritePatternSet *patterns) {
  patterns->add<WhileOpPattern, IfOpPattern, CaseOpPattern>(context);
}

} // namespace mlir::iree_compiler::stablehlo
