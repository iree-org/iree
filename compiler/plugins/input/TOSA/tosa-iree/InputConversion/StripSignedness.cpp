// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tosa-iree/InputConversion/PassDetail.h"
#include "tosa-iree/InputConversion/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

class StripSignednessPass : public StripSignednessBase<StripSignednessPass> {
public:
  explicit StripSignednessPass() {}
  void runOnOperation() override;
};

class IntegerTypeConverter : public TypeConverter {
public:
  static Type convertType(Type type) {
    if (auto iType = llvm::dyn_cast<IntegerType>(type)) {
      if (!iType.isSignless()) {
        return IntegerType::get(type.getContext(),
                                iType.getIntOrFloatBitWidth());
      }
    }
    return type;
  }
  static Type convertTensor(RankedTensorType type) {
    auto newType = RankedTensorType::get(type.getShape(),
                                         convertType(type.getElementType()));
    return newType;
  }
  explicit IntegerTypeConverter() {
    addConversion([](Type type) { return convertType(type); });
    addConversion(convertTensor);
  }
};

// Handles the type conversion component of the TypeConversion. This updates
// conversion patterns that used the original Quant types to be updated to
// the non-quant variants.
class GenericTypeConvert : public ConversionPattern {
public:
  GenericTypeConvert(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> newResults;
    if (isa<FunctionOpInterface>(op)) {
      return failure();
    }

    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, op->getAttrs(), op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

static bool isIllegalType(Type type) {
  if (IntegerType ity = llvm::dyn_cast<IntegerType>(type))
    return !ity.isSignless();
  if (auto shapedType = llvm::dyn_cast<ShapedType>(type)) {
    return isIllegalType(shapedType.getElementType());
  }
  return false;
}

void StripSignednessPass::runOnOperation() {
  IntegerTypeConverter converter;
  ConversionTarget target(getContext());

  // Operations are legal if they don't contain any illegal type.
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      for (Type type : funcOp.getArgumentTypes()) {
        if (isIllegalType(type))
          return false;
      }
      for (Type type : funcOp.getResultTypes()) {
        if (isIllegalType(type))
          return false;
      }
    }
    for (Type type : op->getResultTypes()) {
      if (type && isIllegalType(type))
        return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (type && isIllegalType(type))
        return false;
    }
    return true;
  });

  auto *ctx = &getContext();

  RewritePatternSet patterns(&getContext());
  patterns.insert<GenericTypeConvert>(ctx, converter);
  populateFunctionOpInterfaceTypeConversionPattern(
      getOperation()->getName().getStringRef(), patterns, converter);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createStripSignednessPass() {
  return std::make_unique<StripSignednessPass>();
}

} // namespace iree_compiler
} // namespace mlir
