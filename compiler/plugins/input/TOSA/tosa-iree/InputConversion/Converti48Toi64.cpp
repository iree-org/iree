//===- Converti48Toi64.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert all i48 types to i64.
//
//===----------------------------------------------------------------------===//

#include "tosa-iree/InputConversion/PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::iree_compiler {

class Converti48Toi64Pass : public Converti48Toi64Base<Converti48Toi64Pass> {
public:
  explicit Converti48Toi64Pass() = default;
  void runOnOperation() override;
};

struct i48Toi64Converter : public TypeConverter {
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
    if (isa<func::FuncOp>(op)) {
      return rewriter.notifyMatchFailure(op, "is a func op");
    }

    llvm::SmallVector<Type, 4> oldAttrTypes;
    llvm::SmallVector<unsigned, 4> typedIndices;

    // Extract the typed attributes for conversion.
    for (auto [index, attr] : llvm::enumerate(op->getAttrs())) {
      if (auto typedAttr = attr.getValue().dyn_cast<TypedAttr>()) {
        oldAttrTypes.push_back(typedAttr.getType());
        typedIndices.push_back(index);
      }
    }

    llvm::SmallVector<Type, 4> newAttrTypes;
    (void)getTypeConverter()->convertTypes(oldAttrTypes, newAttrTypes);

    llvm::SmallVector<NamedAttribute, 4> newAttrs(op->getAttrs());
    for (auto [idx, typedIndex] : llvm::enumerate(typedIndices)) {
      auto attrValue = newAttrs[typedIndex].getValue();
      auto newAttrType = newAttrTypes[idx];

      // For integer attributes, create a new integer of new width.
      if (auto intAttr = dyn_cast<IntegerAttr>(attrValue)) {
        if (auto intType = dyn_cast<IntegerType>(newAttrType)) {
          auto value =
              IntegerAttr::get(intType, intAttr.getValue().getZExtValue());
          newAttrs[typedIndex] =
              NamedAttribute(newAttrs[typedIndex].getName(), value);
          continue;
        }
      }

      // For shaped types, map the values to the new types.
      if (auto shapedType = dyn_cast<ShapedType>(newAttrType)) {
        if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attrValue)) {
          auto eType = shapedType.getElementType().dyn_cast<IntegerType>();
          auto cast = [&](APInt value) {
            return APInt(eType.getWidth(), value.getZExtValue());
          };
          auto newDenseAttr = denseAttr.mapValues(eType, cast);
          newAttrs[typedIndex] =
              NamedAttribute(newAttrs[typedIndex].getName(), newDenseAttr);
          continue;
        }
      }
      return rewriter.notifyMatchFailure(op, "Unsupported input type");
    }

    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());
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
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isIllegalType(type))
          return false;
      }
      for (Type type : funcOp.getFunctionType().getResults()) {
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
    for (auto attr : op->getAttrs()) {
      if (auto typedAttr = attr.getValue().dyn_cast<TypedAttr>()) {
        if (isIllegalType(typedAttr.getType())) {
          return false;
        }
      }
    }
    return true;
  });

  auto *ctx = &getContext();
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<GenericTypeConvert>(ctx, converter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);

  if (failed(applyFullConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createConverti48Toi64() {
  return std::make_unique<Converti48Toi64Pass>();
}

} // namespace mlir::iree_compiler
