// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/Patterns.h"

#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

struct ConvertTensorConstantOp
    : public AffinityOpConversionPattern<arith::ConstantOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      arith::ConstantOp constantOp, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle tensor types - other arith.constant types (like i32) are
    // ignored.
    if (!llvm::isa<TensorType>(constantOp.getType())) {
      return failure();
    }

    auto constantType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Constant);
    auto newOp = IREE::Stream::TensorConstantOp::create(
        rewriter, constantOp.getLoc(), constantType,
        convertAttributeToStream(constantOp.getValue()),
        TypeAttr::get(constantOp.getType()),
        /*result_encoding_dims=*/ValueRange{}, executionAffinityAttr);

    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto constantSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        constantOp.getLoc(), rewriter.getIndexType(), newOp.getResult());
    auto transferOp = IREE::Stream::AsyncTransferOp::create(
        rewriter, constantOp.getLoc(), unknownType, newOp.getResult(),
        constantSize, constantSize,
        /*source_affinity=*/executionAffinityAttr,
        /*result_affinity=*/executionAffinityAttr);
    rewriter.replaceOpWithMultiple(constantOp,
                                   {{transferOp.getResult(), constantSize}});
    return success();
  }
};

struct BranchOpConversion
    : public AffinityAwareConversionPattern<mlir::cf::BranchOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
        op, op.getDest(), flattenValues(adaptor.getOperands()));
    return success();
  }
};

struct CondBranchOpConversion
    : public AffinityAwareConversionPattern<mlir::cf::CondBranchOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto trueDestOperands = flattenValues(adaptor.getTrueDestOperands());
    auto falseDestOperands = flattenValues(adaptor.getFalseDestOperands());
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition().front(), op.getTrueDest(), trueDestOperands,
        op.getFalseDest(), falseDestOperands);
    return success();
  }
};

static ValueRange asValueRange(ArrayRef<Value> values) { return values; }

struct SwitchOpConversion
    : public AffinityAwareConversionPattern<mlir::cf::SwitchOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::SwitchOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto defaultOperands = flattenValues(adaptor.getDefaultOperands());
    auto caseOperands = llvm::to_vector(llvm::map_range(
        adaptor.getCaseOperands(), [&](ArrayRef<ValueRange> operands) {
          return flattenValues(operands);
        }));
    rewriter.replaceOpWithNewOp<mlir::cf::SwitchOp>(
        op, adaptor.getFlag().front(), op.getDefaultDestination(),
        defaultOperands, op.getCaseValuesAttr(), op.getCaseDestinations(),
        llvm::to_vector(llvm::map_range(caseOperands, asValueRange)));
    return success();
  }
};

struct SelectOpConversion
    : public AffinityAwareConversionPattern<mlir::arith::SelectOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle selects where the operands are tensors (resources).
    if (!llvm::isa<TensorType>(op.getTrueValue().getType()))
      return failure();
    auto trueOperand = resolveTensorOperands(op.getLoc(), op.getTrueValue(),
                                             adaptor.getTrueValue(), rewriter);
    auto falseOperand = resolveTensorOperands(
        op.getLoc(), op.getFalseValue(), adaptor.getFalseValue(), rewriter);
    auto resourceSelectOp = mlir::arith::SelectOp::create(
        rewriter, op.getLoc(), adaptor.getCondition().front(),
        trueOperand.resource, falseOperand.resource);
    auto sizeSelectOp = mlir::arith::SelectOp::create(
        rewriter, op.getLoc(), adaptor.getCondition().front(),
        trueOperand.resourceSize, falseOperand.resourceSize);
    rewriter.replaceOpWithMultiple(op, {ValueRange{resourceSelectOp.getResult(),
                                                   sizeSelectOp.getResult()}});
    return success();
  }
};

struct ScfIfOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::IfOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource results to resource + size.
    SmallVector<Type> expandedTypes;
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    for (auto originalType : llvm::enumerate(op.getResultTypes())) {
      SmallVector<Type> newTypes;
      if (failed(getTypeConverter()->convertType(originalType.value(),
                                                 newTypes))) {
        return rewriter.notifyMatchFailure(op,
                                           "unable to convert result types");
      }
      resultMap.push_back(
          Result{originalType.index(), expandedTypes.size(), newTypes.front()});
      expandedTypes.append(newTypes);
    }

    // Create a new call that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original call as
    // the result counts differ.
    auto ifOp = mlir::scf::IfOp::create(rewriter, op.getLoc(), expandedTypes,
                                        op.getCondition());

    ifOp.getThenRegion().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());

    ifOp.getElseRegion().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
                                ifOp.getElseRegion().end());

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    SmallVector<Value> resultSizes;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto resource = ifOp.getResult(result.newIndex + 0);
        auto resourceSize = ifOp.getResult(result.newIndex + 1);
        results.push_back(resource);
        resultSizes.push_back(resourceSize);
      } else {
        results.push_back(ifOp.getResult(result.newIndex));
        resultSizes.push_back(nullptr);
      }
    }
    replaceOpWithMultiple(op, results, resultSizes, rewriter);
    return success();
  }
};

struct ScfForOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::ForOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Expand any resource operands to resource + size.
    auto expandedOperands = flattenValues(adaptor.getInitArgs());

    // Expand any resource results to resource + size.
    SmallVector<Type> expandedTypes;
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    for (auto originalType : llvm::enumerate(op.getResultTypes())) {
      SmallVector<Type> newTypes;
      if (failed(getTypeConverter()->convertType(originalType.value(),
                                                 newTypes))) {
        return rewriter.notifyMatchFailure(op,
                                           "unable to convert result types");
      }
      resultMap.push_back(
          Result{originalType.index(), expandedTypes.size(), newTypes.front()});
      expandedTypes.append(newTypes);
    }

    auto &block = op.getRegion().front();
    TypeConverter::SignatureConversion newSignature(block.getNumArguments());
    for (auto arg : llvm::enumerate(block.getArgumentTypes())) {
      if (failed(typeConverter.convertSignatureArg(arg.index(), arg.value(),
                                                   newSignature))) {
        return failure();
      }
    }

    // Create a new loop that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original loop as
    // the result counts differ.
    auto forOp = mlir::scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound().front(),
        adaptor.getUpperBound().front(), adaptor.getStep().front(),
        expandedOperands);

    // Inline the block and update the block arguments.
    rewriter.eraseBlock(forOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), forOp.getRegion(),
                                forOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&forOp.getRegion(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    SmallVector<Value> resultSizes;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto resource = forOp.getResult(result.newIndex + 0);
        auto resourceSize = forOp.getResult(result.newIndex + 1);
        results.push_back(resource);
        resultSizes.push_back(resourceSize);
      } else {
        results.push_back(forOp.getResult(result.newIndex));
        resultSizes.push_back(nullptr);
      }
    }
    replaceOpWithMultiple(op, results, resultSizes, rewriter);
    return success();
  }
};

struct ScfWhileOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::WhileOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Expand any resource operands to resource + size.
    auto expandedOperands = flattenValues(adaptor.getOperands());

    // Expand any resource results to resource + size.
    SmallVector<Type> expandedTypes;
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    for (auto originalType : llvm::enumerate(op.getResultTypes())) {
      SmallVector<Type> newTypes;
      if (failed(getTypeConverter()->convertType(originalType.value(),
                                                 newTypes))) {
        return rewriter.notifyMatchFailure(op,
                                           "unable to convert result types");
      }
      resultMap.push_back(
          Result{originalType.index(), expandedTypes.size(), newTypes.front()});
      expandedTypes.append(newTypes);
    }

    TypeConverter::SignatureConversion newSignature(op.getNumOperands());
    for (auto argType : llvm::enumerate(op.getOperandTypes())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return failure();
      }
    }

    // Create a new call that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original call as
    // the result counts differ.
    auto whileOp = mlir::scf::WhileOp::create(rewriter, op.getLoc(),
                                              expandedTypes, expandedOperands);

    // Inline the `before` block and update the block arguments.
    whileOp.getBefore().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getBefore(), whileOp.getBefore(),
                                whileOp.getBefore().end());
    if (failed(rewriter.convertRegionTypes(&whileOp.getBefore(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Inline the `after` block and update the block arguments.
    whileOp.getAfter().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getAfter(), whileOp.getAfter(),
                                whileOp.getAfter().end());
    if (failed(rewriter.convertRegionTypes(&whileOp.getAfter(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    SmallVector<Value> resultSizes;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto resource = whileOp.getResult(result.newIndex + 0);
        auto resourceSize = whileOp.getResult(result.newIndex + 1);
        results.push_back(resource);
        resultSizes.push_back(resourceSize);
      } else {
        results.push_back(whileOp.getResult(result.newIndex));
        resultSizes.push_back(nullptr);
      }
    }
    replaceOpWithMultiple(op, results, resultSizes, rewriter);
    return success();
  }
};

struct ScfConditionOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::ConditionOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::ConditionOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands = flattenValues(adaptor.getArgs());
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        op, adaptor.getCondition().front(), expandedOperands);
    return success();
  }
};

struct ScfYieldOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::YieldOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands = flattenValues(adaptor.getOperands());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, expandedOperands);
    return success();
  }
};

template <typename OpT>
static inline void addGenericLegalOp(ConversionTarget &conversionTarget,
                                     TypeConverter &typeConverter) {
  conversionTarget.addDynamicallyLegalOp<OpT>([&](OpT op) {
    return llvm::all_of(
               op->getOperandTypes(),
               [&typeConverter](Type t) { return typeConverter.isLegal(t); }) &&
           llvm::all_of(op->getResultTypes(), [&typeConverter](Type t) {
             return typeConverter.isLegal(t);
           });
  });
}

} // namespace

void populateStandardToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns) {
  typeConverter.addConversion([](IndexType type) { return type; });
  typeConverter.addConversion([](IntegerType type) { return type; });
  typeConverter.addConversion([](FloatType type) { return type; });

  // Ensure all shape related ops are fully converted as we should no longer
  // have any types they are valid to be used on after this conversion.
  conversionTarget.addIllegalOp<memref::DimOp, memref::RankOp, tensor::DimOp,
                                tensor::RankOp>();

  conversionTarget.addDynamicallyLegalOp<arith::ConstantOp>(
      [](arith::ConstantOp op) {
        return !llvm::isa<TensorType>(op.getType());
      });
  patterns.insert<ConvertTensorConstantOp>(typeConverter, context,
                                           affinityAnalysis);

  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.

  addGenericLegalOp<mlir::cf::BranchOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::cf::CondBranchOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::cf::SwitchOp>(conversionTarget, typeConverter);
  patterns
      .insert<BranchOpConversion, CondBranchOpConversion, SwitchOpConversion>(
          typeConverter, context, affinityAnalysis);

  addGenericLegalOp<mlir::arith::SelectOp>(conversionTarget, typeConverter);
  patterns.insert<SelectOpConversion>(typeConverter, context, affinityAnalysis);

  addGenericLegalOp<mlir::scf::IfOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::ForOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::WhileOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::ConditionOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::YieldOp>(conversionTarget, typeConverter);
  patterns
      .insert<ScfConditionOpConversion, ScfIfOpConversion, ScfForOpConversion,
              ScfWhileOpConversion, ScfYieldOpConversion>(
          typeConverter, context, affinityAnalysis);
}

} // namespace mlir::iree_compiler
