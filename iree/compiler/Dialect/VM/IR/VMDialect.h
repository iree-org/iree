// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMDIALECT_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMDIALECT_H_

#include "iree/compiler/Dialect/VM/IR/VMFuncEncoder.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

#include "iree/compiler/Dialect/VM/IR/VMOpInterface.h.inc"

class VMDialect : public Dialect {
 public:
  explicit VMDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "vm"; }

  /// Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  /// Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  /// Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

 private:
  /// Register the attributes of this dialect.
  void registerAttributes();
  /// Register the types of this dialect.
  void registerTypes();
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_IR_VMDIALECT_H_
