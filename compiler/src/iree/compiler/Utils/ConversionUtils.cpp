// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ConversionUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

static void emitLegalizationErrors(Location loc,
                                   const DenseSet<Operation *> &illegalOps) {
  // Print op errors for each of the illegal ops that still remain.
  llvm::MapVector<StringRef, int> opNameCounts;
  for (Operation *illegalOp : illegalOps) {
    StringRef opName = illegalOp->getName().getStringRef();
    opNameCounts[opName]++;
    illegalOp->emitOpError() << ": illegal op still exists";
  }

  std::vector<std::string> errorMessages;
  errorMessages.reserve(opNameCounts.size());
  for (const auto &opInfo : opNameCounts) {
    errorMessages.push_back(
        llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
  }
  emitError(loc) << "The following illegal operations still remain: \n"
                 << llvm::join(errorMessages, "\n") << "\n";
}

LogicalResult verifyAllOperationsAreLegal(Operation *op,
                                          const ConversionTarget &target) {
  // We don't just use applyPartialConversion with no patterns because this pass
  // shouldn't alter the IR at all (including via folding or canonicalizations
  // that dialect conversion does automatically).
  DenseSet<Operation *> illegalOps;
  op->walk([&](Operation *op) {
    if (!target.isLegal(op)) {
      illegalOps.insert(op);
    }
  });
  if (illegalOps.empty())
    return success();
  emitLegalizationErrors(op->getLoc(), illegalOps);
  return failure();
}

Attribute convertAttribute(Location loc, Attribute oldAttr,
                           const TypeConverter &typeConverter) {
  // Type attributes get their nested type converted.
  if (auto oldTypeAttr = llvm::dyn_cast<TypeAttr>(oldAttr)) {
    return TypeAttr::get(typeConverter.convertType(oldTypeAttr.getValue()));
  }

  // Return the same attribute if it doesn't have a type.
  auto typedOldAttr = llvm::dyn_cast<TypedAttr>(oldAttr);
  if (!typedOldAttr)
    return oldAttr;

  // Convert the attribute type - if it's the same then it's already legal.
  auto oldType = typedOldAttr.getType();
  auto newType = typeConverter.convertType(oldType);
  if (oldType == newType)
    return typedOldAttr;

  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(typedOldAttr)) {
    APInt value = intAttr.getValue();
    if (newType.isSignedInteger()) {
      value = value.truncSSat(newType.getIntOrFloatBitWidth());
    } else if (newType.isUnsignedInteger()) {
      value = value.truncUSat(newType.getIntOrFloatBitWidth());
    } else {
      value = value.trunc(newType.getIntOrFloatBitWidth());
    }
    return IntegerAttr::get(newType, value);
  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(typedOldAttr)) {
    auto newFloatType = llvm::cast<FloatType>(newType);
    APFloat value = floatAttr.getValue();
    bool losesInfo = false;
    value.convert(newFloatType.getFloatSemantics(), APFloat::rmTowardZero,
                  &losesInfo);
    return FloatAttr::get(newType, value);
  } else if (auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(typedOldAttr)) {
    // NOTE: splats are also dense but this way we avoid needing to convert the
    // same splat value N times.
    return SplatElementsAttr::get(
        llvm::cast<ShapedType>(newType),
        convertAttribute(loc, splatAttr.getSplatValue<Attribute>(),
                         typeConverter));
  } else if (auto denseAttr =
                 llvm::dyn_cast<DenseIntElementsAttr>(typedOldAttr)) {
    auto newElementType = llvm::cast<ShapedType>(newType).getElementType();
    auto newElementBitWidth = newElementType.getIntOrFloatBitWidth();
    if (newElementType.isSignedInteger()) {
      return denseAttr.mapValues(newElementType, [&](APInt src) {
        return src.truncSSat(newElementBitWidth);
      });
    } else if (newElementType.isUnsignedInteger()) {
      return denseAttr.mapValues(newElementType, [&](APInt src) {
        return src.truncUSat(newElementBitWidth);
      });
    } else {
      return denseAttr.mapValues(newElementType, [&](APInt src) {
        return src.trunc(newElementBitWidth);
      });
    }
  } else if (auto denseAttr =
                 llvm::dyn_cast<DenseFPElementsAttr>(typedOldAttr)) {
    auto newElementType =
        llvm::cast<FloatType>(newType.cast<ShapedType>().getElementType());
    const auto &newFloatSemantics = newElementType.getFloatSemantics();
    return denseAttr.mapValues(newElementType, [&](APFloat src) {
      bool losesInfo = false;
      src.convert(newFloatSemantics, APFloat::rmTowardZero, &losesInfo);
      return src.bitcastToAPInt();
    });
  }

  return oldAttr;
}

} // namespace mlir::iree_compiler
