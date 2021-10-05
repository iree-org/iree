// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREEPyDM/Transforms/ToIREE/TypeConverter.h"

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::iree_pydm;

namespace iree_d = mlir::iree;
namespace builtin_d = mlir;
namespace pydm_d = mlir::iree_pydm;

static Type getVariantListType(Builder &builder) {
  return builder.getType<iree_d::ListType>(
      builder.getType<iree_d::VariantType>());
}

LoweringTypeConverter::LoweringTypeConverter() {
  addConversion([](pydm_d::NoneType t) -> Optional<Type> {
    // TODO: This should really be a zero-width opaque value in the VM. Just
    // making it an integer now.
    return builtin_d::IntegerType::get(t.getContext(), 32);
  });
  addConversion([](pydm_d::ExceptionResultType t) -> Optional<Type> {
    return builtin_d::IntegerType::get(t.getContext(), 32);
  });
  addConversion([](pydm_d::ObjectType t) -> Optional<Type> {
    Builder b(t.getContext());
    return getVariantListType(b);
  });

  // Bool.
  addConversion([&](pydm_d::BoolType t) -> Optional<Type> {
    return builtin_d::IntegerType::get(t.getContext(), 1);
  });

  // Integer type hierarchy.
  addConversion([&](pydm_d::IntegerType t) -> Optional<Type> {
    Builder b(t.getContext());
    if (t.isWeak()) {
      return getWeakIntegerType(b);
    }
    return b.getIntegerType(t.getBitWidth());
  });

  // Real type hierarchy.
  addConversion([&](pydm_d::RealType t) -> Optional<Type> {
    Builder b(t.getContext());
    if (t.isWeak()) {
      return getWeakFloatType(b);
    }
    return t.getFloatType();
  });

  // Variable references.
  addConversion([](pydm_d::FreeVarRefType t) -> Optional<Type> {
    // Just an object record.
    Builder b(t.getContext());
    return getVariantListType(b);
  });

  // Explicit conversions for allowed built-in types (avoids default conversion
  // which can mask issues).
  addConversion([](builtin_d::IntegerType t) -> Optional<Type> { return t; });
  addConversion([](builtin_d::FloatType t) -> Optional<Type> { return t; });
  addConversion([](builtin_d::IndexType t) -> Optional<Type> { return t; });
}

Type LoweringTypeConverter::getBoolType(Builder b) const {
  return b.getIntegerType(boolBits);
}

Type LoweringTypeConverter::getWeakIntegerType(Builder b) const {
  return b.getIntegerType(weakIntegerBits);
}

Type LoweringTypeConverter::getWeakFloatType(Builder b) const {
  switch (weakFloatType) {
    case WeakFloatType::F32:
      return b.getF32Type();
    case WeakFloatType::F64:
      return b.getF64Type();
  }
}

bool LoweringTypeConverter::isTypeLegal(Type t) const {
  return t.isa<builtin_d::IntegerType, builtin_d::FloatType,
               builtin_d::IndexType, iree_d::ListType>();
}

bool LoweringTypeConverter::areTypesLegal(TypeRange types) const {
  for (Type t : types) {
    if (!isTypeLegal(t)) return false;
  }
  return true;
}
