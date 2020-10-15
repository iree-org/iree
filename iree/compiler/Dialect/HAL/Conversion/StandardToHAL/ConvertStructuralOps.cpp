// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/ConvertStandardToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
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

class BranchOpConversion : public OpConversionPattern<mlir::BranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::BranchOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::BranchOpAdaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, op.dest(),
                                                adaptor.destOperands());
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<mlir::CondBranchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::CondBranchOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::CondBranchOpAdaptor adaptor(operands,
                                      op.getOperation()->getAttrDictionary());
    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        op, adaptor.condition(), op.trueDest(), adaptor.trueDestOperands(),
        op.falseDest(), op.falseDestOperands());
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp returnOp, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(returnOp, operands);
    return success();
  }
};

}  // namespace

void populateStandardStructuralToHALPatterns(MLIRContext *context,
                                             OwningRewritePatternList &patterns,
                                             TypeConverter &converter) {
  patterns.insert<FuncOpSignatureConversion, BranchOpConversion,
                  CondBranchOpConversion, ReturnOpConversion>(converter,
                                                              context);
}

}  // namespace iree_compiler
}  // namespace mlir
