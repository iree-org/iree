// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ImportMLProgramPass : public ImportMLProgramBase<ImportMLProgramPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }
  void runOnOperation() override;
};

class IREETypeConverter : public TypeConverter {
 public:
  IREETypeConverter();
};

// Generic 1:1 conversion pattern which effectively just renames an op.
// It does not support regions or ops with successors.
class OneToOneConversionPattern : public ConversionPattern {
 public:
  OneToOneConversionPattern(TypeConverter &converter, StringRef srcName,
                            StringRef targetName, MLIRContext *context,
                            PatternBenefit benefit)
      : ConversionPattern(converter, srcName, benefit, context),
        targetName(targetName) {}
  LogicalResult matchAndRewrite(
      Operation *srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(srcOp->getResultTypes(),
                                           resultTypes))) {
      return srcOp->emitError()
             << "could not convert result types to IREE internal types";
    }

    OperationState state(srcOp->getLoc(), targetName, operands, resultTypes,
                         srcOp->getAttrs());
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }

 private:
  StringRef targetName;
};

class MLProgramGlobalOpPattern
    : public OpConversionPattern<ml_program::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ml_program::GlobalOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type newType = typeConverter->convertType(srcOp.getType());
    if (!newType) return failure();
    auto globalOp = rewriter.replaceOpWithNewOp<IREE::Util::GlobalOp>(
        srcOp, srcOp.getName(), srcOp.getIsMutable(), newType,
        srcOp.getValue());
    globalOp.setVisibility(srcOp.getVisibility());
    return success();
  }
};

}  // namespace

IREETypeConverter::IREETypeConverter() {
  addConversion([](Type t) { return t; });
}

void ImportMLProgramPass::runOnOperation() {
  auto &context = getContext();
  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<IREE::Util::UtilDialect>();
  target.addIllegalDialect<ml_program::MLProgramDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  IREETypeConverter typeConverter;
  patterns.insert<MLProgramGlobalOpPattern>(typeConverter, &getContext(), 0);

  PatternBenefit specific_benefit = 100;
#define ONE_TO_ONE(SrcOpTy, TargetOpTy)           \
  patterns.insert<OneToOneConversionPattern>(     \
      typeConverter, SrcOpTy::getOperationName(), \
      TargetOpTy::getOperationName(), &context, specific_benefit)

  ONE_TO_ONE(ml_program::GlobalLoadOp, IREE::Util::GlobalLoadOp);
  ONE_TO_ONE(ml_program::GlobalLoadConstOp, IREE::Util::GlobalLoadOp);
  ONE_TO_ONE(ml_program::GlobalStoreOp, IREE::Util::GlobalStoreOp);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createImportMLProgramPass() {
  return std::make_unique<ImportMLProgramPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
