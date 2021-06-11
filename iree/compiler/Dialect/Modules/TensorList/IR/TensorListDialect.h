// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLIST_DIALECT_H_
#define IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLIST_DIALECT_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

class TensorListDialect : public Dialect {
 public:
  explicit TensorListDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tensorlist"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;
};

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLIST_DIALECT_H_
