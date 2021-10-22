// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/ConvertStandardToStream.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class FuncOpSignatureConversion : public OpConversionPattern<mlir::FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::FuncOp funcOp, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Convert the input signature types.
    // TODO(benvanik): dynamic shapes by passing in tensor dynamic dims.
    auto originalType = funcOp.getType();
    TypeConverter::SignatureConversion newSignature(
        originalType.getNumInputs());
    for (auto argType : llvm::enumerate(originalType.getInputs())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return failure();
      }
    }
    SmallVector<Type, 4> newResultTypes;
    if (failed(typeConverter.convertTypes(originalType.getResults(),
                                          newResultTypes))) {
      return failure();
    }

    // Replace function.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                               newResultTypes));
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

static SmallVector<Value> expandResourceOperands(
    Location loc, ValueRange operands, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> expandedOperands;
  expandedOperands.reserve(operands.size());
  for (auto operand : operands) {
    if (operand.getType().isa<IREE::Stream::ResourceType>()) {
      expandedOperands.push_back(operand);
      expandedOperands.push_back(
          rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(loc, operand));
    } else {
      expandedOperands.push_back(operand);
    }
  }
  return expandedOperands;
}

class CallOpConversion : public OpConversionPattern<mlir::CallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::CallOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), newOperands, rewriter);

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
    auto callOp = rewriter.create<mlir::CallOp>(op.getLoc(), expandedTypes,
                                                op.callee(), expandedOperands);

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (result.newType.isa<IREE::Stream::ResourceType>()) {
        auto resource = callOp.getResult(result.newIndex + 0);
        auto resourceSize = callOp.getResult(result.newIndex + 1);
        results.push_back(rewriter.create<IREE::Stream::AsyncTransferOp>(
            op.getLoc(), result.newType, resource, resourceSize, resourceSize,
            /*source_affinity=*/nullptr, /*result_affinity=*/nullptr));
      } else {
        results.push_back(callOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), newOperands, rewriter);
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, expandedOperands);
    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<mlir::BranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::BranchOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::BranchOpAdaptor operands(newOperands);
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), operands.destOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, op.dest(),
                                                expandedOperands);
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<mlir::CondBranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::CondBranchOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::CondBranchOpAdaptor operands(newOperands,
                                       op.getOperation()->getAttrDictionary());
    // Expand any resource operands to resource + size.
    auto trueDestOperands = expandResourceOperands(
        op.getLoc(), operands.trueDestOperands(), rewriter);
    auto falseDestOperands = expandResourceOperands(
        op.getLoc(), operands.falseDestOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        op, operands.condition(), op.trueDest(), trueDestOperands,
        op.falseDest(), falseDestOperands);
    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<mlir::SelectOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::SelectOp op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::SelectOp::Adaptor operands(newOperands);
    rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, operands.condition(),
                                                operands.true_value(),
                                                operands.false_value());
    return success();
  }
};

}  // namespace

void populateStandardStructuralToStreamPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns) {
  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.

  conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  conversionTarget.addDynamicallyLegalOp<mlir::CallOp>([&](mlir::CallOp op) {
    return llvm::all_of(
               op.getOperandTypes(),
               [&](Type type) { return typeConverter.isLegal(type); }) &&
           llvm::all_of(op.getResultTypes(),
                        [&](Type type) { return typeConverter.isLegal(type); });
  });
  conversionTarget.addDynamicallyLegalOp<mlir::ReturnOp>(
      [&](mlir::ReturnOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });

  conversionTarget.addDynamicallyLegalOp<mlir::BranchOp>(
      [&](mlir::BranchOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::CondBranchOp>(
      [&](mlir::CondBranchOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });

  patterns
      .insert<FuncOpSignatureConversion, CallOpConversion, ReturnOpConversion,
              BranchOpConversion, CondBranchOpConversion, SelectOpConversion>(
          typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
