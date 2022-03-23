// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/PyDM/Transforms/ToIREE/TypeConverter.h"

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

static Type getVariantListType(Builder &builder) {
  return builder.getType<IREE::Input::ListType>(
      builder.getType<IREE::Input::VariantType>());
}

LoweringTypeConverter::LoweringTypeConverter() {
  addConversion([](PYDM::NoneType t) -> Optional<Type> {
    // TODO: This should really be a zero-width opaque value in the VM. Just
    // making it an integer now.
    return mlir::IntegerType::get(t.getContext(), 32);
  });
  addConversion([](PYDM::ExceptionResultType t) -> Optional<Type> {
    return mlir::IntegerType::get(t.getContext(), 32);
  });
  addConversion([](PYDM::ObjectType t) -> Optional<Type> {
    Builder b(t.getContext());
    return getVariantListType(b);
  });

  // Bool.
  addConversion([&](PYDM::BoolType t) -> Optional<Type> {
    return mlir::IntegerType::get(t.getContext(), 1);
  });

  // Integer type hierarchy.
  addConversion([&](PYDM::IntegerType t) -> Optional<Type> {
    Builder b(t.getContext());
    if (t.isWeak()) {
      return getWeakIntegerType(b);
    }
    return b.getIntegerType(t.getBitWidth());
  });

  // Real type hierarchy.
  addConversion([&](PYDM::RealType t) -> Optional<Type> {
    Builder b(t.getContext());
    if (t.isWeak()) {
      return getWeakFloatType(b);
    }
    return t.getFloatType();
  });

  // Tuple, List.
  // TODO: Fork these based on CollectionStorageClass as they can avoid
  // using variant lists.
  addConversion([&](PYDM::ListType t) -> Optional<Type> {
    Builder b(t.getContext());
    return getVariantListType(b);
  });
  addConversion([&](PYDM::TupleType t) -> Optional<Type> {
    Builder b(t.getContext());
    return getVariantListType(b);
  });

  // Variable references.
  addConversion([](PYDM::FreeVarRefType t) -> Optional<Type> {
    // Just an object record.
    Builder b(t.getContext());
    return getVariantListType(b);
  });

  // Explicit conversions for allowed built-in types (avoids default conversion
  // which can mask issues).
  addConversion([](mlir::IndexType t) -> Optional<Type> { return t; });
  addConversion([](mlir::IntegerType t) -> Optional<Type> { return t; });
  addConversion([](mlir::FloatType t) -> Optional<Type> { return t; });
  addConversion([](mlir::IndexType t) -> Optional<Type> { return t; });
  addConversion([](IREE::Input::ListType t) -> Optional<Type> { return t; });
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
  return t.isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType,
               IREE::Input::ListType>();
}

bool LoweringTypeConverter::areTypesLegal(TypeRange types) const {
  for (Type t : types) {
    if (!isTypeLegal(t))
      return false;
  }
  return true;
}
