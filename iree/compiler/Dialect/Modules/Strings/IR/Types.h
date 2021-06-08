// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_STRINGS_IR_TYPES_H_
#define IREE_COMPILER_DIALECT_MODULES_STRINGS_IR_TYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Strings {

class StringType : public Type::TypeBase<StringType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class StringTensorType
    : public Type::TypeBase<StringTensorType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace Strings
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_STRINGS_IR_TYPES_H_
