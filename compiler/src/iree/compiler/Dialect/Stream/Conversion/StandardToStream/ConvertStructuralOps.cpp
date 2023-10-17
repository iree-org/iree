// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/Patterns.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

struct FuncOpSignatureConversion
    : public OpConversionPattern<mlir::func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Convert the input signature types.
    // TODO(benvanik): dynamic shapes by passing in tensor dynamic dims.
    auto originalType = funcOp.getFunctionType();
    TypeConverter::SignatureConversion newSignature(
        originalType.getNumInputs());
    for (auto argType : llvm::enumerate(originalType.getInputs())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return failure();
      }
    }
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter.convertTypes(originalType.getResults(),
                                          newResultTypes))) {
      return failure();
    }

    // Replace function.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getFunctionBody(),
                                newFuncOp.getFunctionBody(), newFuncOp.end());
    newFuncOp.setType(rewriter.getFunctionType(newSignature.getConvertedTypes(),
                                               newResultTypes));
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter, &newSignature))) {
      return failure();
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

static SmallVector<Value>
expandResourceOperands(Location loc, ValueRange operands,
                       ConversionPatternRewriter &rewriter) {
  SmallVector<Value> expandedOperands;
  expandedOperands.reserve(operands.size());
  for (auto operand : operands) {
    if (llvm::isa<TensorType>(operand.getType())) {
      auto value = consumeTensorOperand(loc, operand, rewriter);
      expandedOperands.push_back(value.resource);
      expandedOperands.push_back(value.resourceSize);
    } else if (llvm::isa<IREE::Stream::ResourceType>(operand.getType())) {
      expandedOperands.push_back(operand);
      expandedOperands.push_back(
          rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(loc, operand));
    } else {
      expandedOperands.push_back(operand);
    }
  }
  return expandedOperands;
}

struct CallOpConversion : public OpConversionPattern<mlir::func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);

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
    auto callOp = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), expandedTypes, op.getCallee(), expandedOperands);

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = callOp.getResult(result.newIndex + 0);
        auto resourceSize = callOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(callOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, expandedOperands);
    return success();
  }
};

struct BranchOpConversion : public OpConversionPattern<mlir::cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands = expandResourceOperands(
        op.getLoc(), adaptor.getDestOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                    expandedOperands);
    return success();
  }
};

struct CondBranchOpConversion
    : public OpConversionPattern<mlir::cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto trueDestOperands = expandResourceOperands(
        op.getLoc(), adaptor.getTrueDestOperands(), rewriter);
    auto falseDestOperands = expandResourceOperands(
        op.getLoc(), adaptor.getFalseDestOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(), trueDestOperands,
        op.getFalseDest(), falseDestOperands);
    return success();
  }
};

static ValueRange asValueRange(ArrayRef<Value> values) { return values; }

struct SwitchOpConversion : public OpConversionPattern<mlir::cf::SwitchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto defaultOperands = expandResourceOperands(
        op.getLoc(), adaptor.getDefaultOperands(), rewriter);
    auto caseOperands = llvm::to_vector(
        llvm::map_range(adaptor.getCaseOperands(), [&](ValueRange operands) {
          return expandResourceOperands(op.getLoc(), operands, rewriter);
        }));
    rewriter.replaceOpWithNewOp<mlir::cf::SwitchOp>(
        op, adaptor.getFlag(), op.getDefaultDestination(), defaultOperands,
        op.getCaseValuesAttr(), op.getCaseDestinations(),
        llvm::to_vector(llvm::map_range(caseOperands, asValueRange)));
    return success();
  }
};

struct SelectOpConversion : public OpConversionPattern<mlir::arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle selects where the operands are tensors (resources).
    if (!llvm::isa<TensorType>(op.getTrueValue().getType()))
      return failure();
    auto trueOperand =
        consumeTensorOperand(op.getLoc(), adaptor.getTrueValue(), rewriter);
    auto falseOperand =
        consumeTensorOperand(op.getLoc(), adaptor.getFalseValue(), rewriter);
    auto resourceSelectOp = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), adaptor.getCondition(), trueOperand.resource,
        falseOperand.resource);
    auto sizeSelectOp = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), adaptor.getCondition(), trueOperand.resourceSize,
        falseOperand.resourceSize);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, adaptor.getTrueValue().getType(),
        ValueRange{resourceSelectOp.getResult(), sizeSelectOp.getResult()});
    return success();
  }
};

struct ScfIfOpConversion : public OpConversionPattern<mlir::scf::IfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);

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
    auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), expandedTypes,
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
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = ifOp.getResult(result.newIndex + 0);
        auto resourceSize = ifOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(ifOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ScfForOpConversion : public OpConversionPattern<mlir::scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getInitArgs(), rewriter);

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

    // Create a new call that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original loop as
    // the result counts differ.
    auto forOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), expandedOperands);

    // Inline the block and update the block arguments.
    forOp.getRegion().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getRegion(), forOp.getRegion(),
                                forOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&forOp.getRegion(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = forOp.getResult(result.newIndex + 0);
        auto resourceSize = forOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(forOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ScfWhileOpConversion : public OpConversionPattern<mlir::scf::WhileOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);

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
    auto whileOp = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), expandedTypes, expandedOperands);

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
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = whileOp.getResult(result.newIndex + 0);
        auto resourceSize = whileOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(whileOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ScfConditionOpConversion
    : public OpConversionPattern<mlir::scf::ConditionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getArgs(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        op, adaptor.getCondition(), expandedOperands);
    return success();
  }
};

struct ScfYieldOpConversion : public OpConversionPattern<mlir::scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, expandedOperands);
    return success();
  }
};

} // namespace

void populateStandardStructuralToStreamPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.

  conversionTarget.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
  conversionTarget.addDynamicallyLegalOp<mlir::func::CallOp>(
      [&](mlir::func::CallOp op) {
        return llvm::all_of(
                   op.getOperandTypes(),
                   [&](Type type) { return typeConverter.isLegal(type); }) &&
               llvm::all_of(op.getResultTypes(), [&](Type type) {
                 return typeConverter.isLegal(type);
               });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });

  conversionTarget.addDynamicallyLegalOp<mlir::cf::BranchOp>(
      [&](mlir::cf::BranchOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::cf::CondBranchOp>(
      [&](mlir::cf::CondBranchOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::cf::SwitchOp>(
      [&](mlir::cf::SwitchOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::scf::IfOp>(
      [&](mlir::scf::IfOp op) {
        return llvm::all_of(op.getResultTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::scf::ForOp>(
      [&](mlir::scf::ForOp op) {
        return llvm::all_of(op.getResultTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::scf::WhileOp>(
      [&](mlir::scf::WhileOp op) {
        return llvm::all_of(op.getResultTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::scf::ConditionOp>(
      [&](mlir::scf::ConditionOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });
  conversionTarget.addDynamicallyLegalOp<mlir::scf::YieldOp>(
      [&](mlir::scf::YieldOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
          return typeConverter.isLegal(type);
        });
      });

  patterns
      .insert<FuncOpSignatureConversion, CallOpConversion, ReturnOpConversion,
              BranchOpConversion, CondBranchOpConversion, SwitchOpConversion,
              SelectOpConversion, ScfConditionOpConversion, ScfIfOpConversion,
              ScfForOpConversion, ScfWhileOpConversion, ScfYieldOpConversion>(
          typeConverter, context);
}

} // namespace iree_compiler
} // namespace mlir
