// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTI48TOI64PASS
#include "compiler/plugins/input/TOSA/InputConversion/Passes.h.inc"

namespace {

class Converti48Toi64Pass final
    : public impl::Converti48Toi64PassBase<Converti48Toi64Pass> {
public:
  explicit Converti48Toi64Pass() = default;
  void runOnOperation() override;
};

struct i48Toi64Converter : TypeConverter {
public:
  static Type convertType(Type type) {
    if (type.isInteger(48)) {
      return IntegerType::get(type.getContext(), /*width=*/64);
    }
    return type;
  }
  static Type convertTensor(RankedTensorType type) {
    auto newType = RankedTensorType::get(type.getShape(),
                                         convertType(type.getElementType()));
    return newType;
  }
  explicit i48Toi64Converter() {
    addConversion([](Type type) { return convertType(type); });
    addConversion(convertTensor);
  }
};

static Attribute convertIntegerAttribute(Attribute attr,
                                         const TypeConverter &converter) {
  auto typedAttr = dyn_cast<TypedAttr>(attr);
  if (!typedAttr) {
    return attr;
  }
  Type newType = converter.convertType(typedAttr.getType());
  if (!newType) {
    return {};
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto intType = dyn_cast<IntegerType>(newType)) {
      return IntegerAttr::get(intType, intAttr.getValue().getZExtValue());
    }
  }
  if (auto shapedType = dyn_cast<ShapedType>(newType)) {
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
      auto elementType = dyn_cast<IntegerType>(shapedType.getElementType());
      if (elementType) {
        return denseAttr.mapValues(elementType, [&elementType](APInt value) {
          return APInt(elementType.getWidth(), value.getZExtValue());
        });
      }
    }
  }
  return {};
}

// Handles the type conversion component of the TypeConversion. This updates
// conversion patterns that used the original i48 tensor types to be
// updated to the i64 variants.
class GenericTypeConvert : public ConversionPattern {
public:
  GenericTypeConvert(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type, 4> newResults;
    if (isa<mlir::FunctionOpInterface>(op)) {
      return rewriter.notifyMatchFailure(op, "is a func op");
    }

    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResults))) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    Operation *newOp = op->clone(
        Operation::CloneOptions().withResultTypes(llvm::to_vector(newResults)));
    bool conversionFailed = false;
    newOp->getName().walkInherentAttrs(newOp, [&](StringRef, Attribute &attr) {
      if (Attribute converted =
              convertIntegerAttribute(attr, *getTypeConverter())) {
        attr = converted;
      } else {
        conversionFailed = true;
      }
    });
    SmallVector<NamedAttribute> attrs;
    for (NamedAttribute attr : op->getDiscardableAttrs()) {
      if (Attribute converted =
              convertIntegerAttribute(attr.getValue(), *getTypeConverter())) {
        attrs.emplace_back(attr.getName(), converted);
      } else {
        conversionFailed = true;
      }
    }
    if (conversionFailed) {
      newOp->destroy();
      return rewriter.notifyMatchFailure(op, "unsupported attribute type");
    }
    newOp->setDiscardableAttrs(attrs);
    newOp->setOperands(operands);
    rewriter.insert(newOp);
    for (auto [r, newRegion] :
         llvm::zip_equal(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(r, newRegion, newRegion.begin());
      TypeConverter::SignatureConversion result(newRegion.getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion.getArgumentTypes(), result);
      rewriter.applySignatureConversion(&newRegion.front(), result);
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

static bool isIllegalType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return isIllegalType(shapedType.getElementType());
  }
  return type.isInteger(48);
}

void Converti48Toi64Pass::runOnOperation() {
  i48Toi64Converter converter;
  ConversionTarget target(getContext());

  // Operations are legal if they don't contain any illegal type.
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op)) {
      for (Type type : funcOp.getArgumentTypes()) {
        if (isIllegalType(type)) {
          return false;
        }
      }
      for (Type type : funcOp.getResultTypes()) {
        if (isIllegalType(type)) {
          return false;
        }
      }
    }
    for (Type type : op->getResultTypes()) {
      if (type && isIllegalType(type)) {
        return false;
      }
    }
    for (Type type : op->getOperandTypes()) {
      if (type && isIllegalType(type)) {
        return false;
      }
    }
    bool legal = true;
    auto check = [&](Attribute attr) {
      if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
        legal &= !isIllegalType(typedAttr.getType());
      }
    };
    op->getName().walkInherentAttrs(
        op, [&](StringRef, Attribute &attr) { check(attr); });
    for (NamedAttribute attr : op->getDiscardableAttrs()) {
      check(attr.getValue());
    }
    return legal;
  });

  auto *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<GenericTypeConvert>(ctx, converter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);

  if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace
} // namespace mlir::iree_compiler
