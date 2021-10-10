// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_IR_FLOWDIALECT_H_
#define IREE_COMPILER_DIALECT_STREAM_IR_FLOWDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

class StreamDialect : public Dialect {
 public:
  explicit StreamDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "stream"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;

  static bool isDialectOp(Operation *op) {
    return op && op->getDialect() &&
           op->getDialect()->getNamespace() == getDialectNamespace();
  }

 private:
  void registerAttributes();
  void registerTypes();
};

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_STREAM_IR_FLOWDIALECT_H_
