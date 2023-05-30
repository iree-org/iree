// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/StableHLO/TypeConversion.h"

#include <optional>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler::stablehlo {

namespace {

Type convertInteger(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

Type convertShapedType(ShapedType shapedType) {
  if (auto intType = llvm::dyn_cast<IntegerType>(shapedType.getElementType()))
    return shapedType.clone(convertInteger(intType));
  return shapedType;
}

std::optional<Value> materializeCastFromIllegal(OpBuilder& builder, Type type,
                                                ValueRange inputs,
                                                Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
      !toType.isSignlessInteger())
    return std::nullopt;
  // Use unrealized conversion casts to do signful->signless conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

std::optional<Value> materializeCastToIllegal(OpBuilder& builder, Type type,
                                              ValueRange inputs, Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if (!fromType.isSignlessInteger() ||
      (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
    return std::nullopt;
  // Use unrealized conversion casts to do signless->signful conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

std::optional<Value> scalarToTensor(OpBuilder& builder, Type /*type*/,
                                    ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  if (llvm::isa<ShapedType>(inputs.front().getType())) {
    return std::nullopt;
  }
  return builder
      .create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({}, inputs.front().getType()),
          inputs.front())
      .getResult();
}

}  // namespace

RemoveSignTypeConverter::RemoveSignTypeConverter() {
  addConversion([](Type type) { return type; });

  addConversion(convertInteger);
  addConversion(convertShapedType);

  addArgumentMaterialization(materializeCastFromIllegal);
  addSourceMaterialization(materializeCastToIllegal);
  addTargetMaterialization(materializeCastFromIllegal);
}

LinalgTypeConverter::LinalgTypeConverter() : RemoveSignTypeConverter() {
  addArgumentMaterialization(scalarToTensor);
}

}  // namespace mlir::iree_compiler::stablehlo
