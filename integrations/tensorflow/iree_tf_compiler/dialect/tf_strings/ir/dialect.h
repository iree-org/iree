// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_DIALECT_H_
#define INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

#include "iree_tf_compiler/dialect/tf_strings/ir/op_interface.h.inc"

class TFStringsDialect : public Dialect {
 public:
  explicit TFStringsDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "tf_strings"; }

  Type parseType(DialectAsmParser& parser) const override;

  void printType(Type type, DialectAsmPrinter& os) const override;
};

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir

#endif  // INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_DIALECT_H_
