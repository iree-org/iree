// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_IREEDIALECT_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_IREEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {

#include "iree/compiler/Dialect/Shape/IR/ShapeInterfaces.h.inc"

class ShapeDialect : public Dialect {
 public:
  explicit ShapeDialect(MLIRContext* context);
  // TODO(b/143787186): rename to iree.
  static StringRef getDialectNamespace() { return "shapex"; }

  Type parseType(DialectAsmParser& parser) const override;
  void printType(Type type, DialectAsmPrinter& os) const override;

  Operation* materializeConstant(OpBuilder& builder, Attribute value, Type type,
                                 Location loc) override;

 private:
  /// Register the types of this dialect.
  void registerTypes();
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_IREEDIALECT_H_
