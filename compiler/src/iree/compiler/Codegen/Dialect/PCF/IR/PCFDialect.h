// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFDIALECT_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::PCF {

class PCFDialect : public Dialect {
public:
  explicit PCFDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "pcf"; }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;

private:
  void registerOperations();
  void registerAttributes();
  void registerTypes();
};

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_IR_PCFDIALECT_H_
