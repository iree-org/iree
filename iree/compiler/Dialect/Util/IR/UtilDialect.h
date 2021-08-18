// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILDIALECT_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

class UtilDialect : public Dialect {
 public:
  explicit UtilDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "util"; }

  /// Parses a type registered to this dialect.
  Type parseType(DialectAsmParser& parser) const override;

  /// Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter& os) const override;

 private:
  /// Register the types of this dialect.
  void registerTypes();
};

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_UTIL_IR_UTILDIALECT_H_
