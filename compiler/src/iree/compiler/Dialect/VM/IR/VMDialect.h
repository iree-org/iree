// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMDIALECT_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMDIALECT_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMFuncEncoder.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::IREE::VM {

#include "iree/compiler/Dialect/VM/IR/VMOpInterfaces.h.inc" // IWYU pragma: export

class VMDialect : public Dialect {
public:
  explicit VMDialect(MLIRContext *context);
  ~VMDialect() override;
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

  // Provides a hook for op interface.
  void *getRegisteredInterfaceForOp(mlir::TypeID interface,
                                    mlir::OperationName opName) override;

private:
  /// Register the attributes of this dialect.
  void registerAttributes();
  /// Register the types of this dialect.
  void registerTypes();

  struct VMOpAsmInterface;
  VMOpAsmInterface *fallbackOpAsmInterface;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_IR_VMDIALECT_H_
