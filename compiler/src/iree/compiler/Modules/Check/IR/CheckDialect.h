// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_CHECK_IR_CHECK_DIALECT_H_
#define IREE_COMPILER_MODULES_CHECK_IR_CHECK_DIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

class CheckDialect : public Dialect {
 public:
  explicit CheckDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "check"; }
};

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_MODULES_CHECK_IR_CHECK_DIALECT_H_
