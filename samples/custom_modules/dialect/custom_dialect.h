// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_DIALECT_H_
#define IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_DIALECT_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Custom {

class CustomDialect : public Dialect {
 public:
  explicit CustomDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "custom"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;
};

class MessageType : public Type::TypeBase<MessageType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace Custom
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/samples/custom_modules/dialect/custom_ops.h.inc"

#endif  // IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_DIALECT_H_
