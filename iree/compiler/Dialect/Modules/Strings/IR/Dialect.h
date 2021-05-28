// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TRANSLATION_MODULES_STRINGS_IR_DIALECT_H_
#define IREE_COMPILER_TRANSLATION_MODULES_STRINGS_IR_DIALECT_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Strings {

class StringsDialect : public Dialect {
 public:
  explicit StringsDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "strings"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;
};

}  // namespace Strings
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_MODULES_STRINGS_IR_DIALECT_H_
