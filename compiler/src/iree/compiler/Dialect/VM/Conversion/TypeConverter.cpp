// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-vm"

namespace mlir::iree_compiler::IREE::VM {

TypeConverter::TypeConverter(TargetOptions targetOptions)
    : targetOptions_(targetOptions) {
  addConversion([](IREE::Util::ObjectType type) {
    // Objects are always opaque ref types.
    return IREE::VM::RefType::get(IREE::VM::OpaqueType::get(type.getContext()));
  });
  addConversion([](IREE::Util::VariantType type) {
    // Variant means opaque in VM.
    return IREE::VM::OpaqueType::get(type.getContext());
  });

  // All ref types are passed through unmodified.
  addConversion([](IREE::VM::RefType type) { return type; });

  // Wrap ref types.
  addConversion([](Type type) -> std::optional<Type> {
    if (IREE::VM::RefType::isCompatible(type)) {
      return IREE::VM::RefType::get(type);
    }
    return std::nullopt;
  });

  // Pointer types remain as pointer types types are passed through unmodified.
  addConversion([this](IREE::Util::PtrType type) -> std::optional<Type> {
    // Recursively handle pointer target types (we want to convert ptr<index> to
    // ptr<i32>, for example).
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return std::nullopt;
    }
    return IREE::Util::PtrType::get(targetType);
  });

  // Convert integer types.
  addConversion([](IntegerType integerType) -> std::optional<Type> {
    if (integerType.isInteger(32) || integerType.isInteger(64)) {
      // i32 and i64 are always supported by the runtime.
      return integerType;
    } else if (integerType.getIntOrFloatBitWidth() < 32) {
      // Promote i1/i8/i16 -> i32.
      return IntegerType::get(integerType.getContext(), 32);
    }
    return std::nullopt;
  });

  // Convert floating-point types.
  addConversion([this](FloatType floatType) -> std::optional<Type> {
    if (floatType.getIntOrFloatBitWidth() < 32) {
      if (targetOptions_.f32Extension) {
        // Promote f16 -> f32.
        return FloatType::getF32(floatType.getContext());
      } else {
        // f32 is not supported; can't compile.
        return std::nullopt;
      }
    } else if (floatType.isF32()) {
      if (targetOptions_.f32Extension) {
        return floatType;
      } else {
        // f32 is not supported; can't compile.
        return std::nullopt;
      }
    } else if (floatType.isF64()) {
      if (targetOptions_.f64Extension) {
        // f64 is supported by the VM, use directly.
        return floatType;
      } else if (targetOptions_.f32Extension &&
                 targetOptions_.truncateUnsupportedFloats) {
        // f64 is not supported and we still want to compile, so truncate to
        // f32 (unsafe if all bits are actually required!).
        return FloatType::getF32(floatType.getContext());
      }
    }
    return std::nullopt;
  });

  // Convert index types to the target bit width.
  addConversion([this](IndexType indexType) -> std::optional<Type> {
    return IntegerType::get(indexType.getContext(), targetOptions_.indexBits);
  });

  // Vectors are used for arbitrary byte storage.
  addConversion([](VectorType vectorType) -> std::optional<Type> {
    return IREE::VM::RefType::get(
        IREE::VM::BufferType::get(vectorType.getContext()));
  });

  addSourceMaterialization([](OpBuilder &builder, IndexType type,
                              ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1 ||
        !llvm::isa<IntegerType>(inputs.front().getType())) {
      return nullptr;
    }
    return builder.create<arith::IndexCastOp>(loc, type, inputs.front());
  });

  addTargetMaterialization(
      [](OpBuilder &builder, IntegerType type, ValueRange inputs,
         Location loc) -> Value { return inputs.front(); });
}

} // namespace mlir::iree_compiler::IREE::VM
