// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_IR_STREAMDIALECT_H_
#define IREE_COMPILER_DIALECT_STREAM_IR_STREAMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler::IREE::Stream {

class StreamDialect : public Dialect {
public:
  explicit StreamDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "stream"; }

  void getCanonicalizationPatterns(RewritePatternSet &results) const override;

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;

private:
  void registerAttributes();
  void registerTypes();
};

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_IR_STREAMDIALECT_H_
