// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTDIALECT_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

class LinalgExtDialect : public Dialect {
 public:
  explicit LinalgExtDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "linalg_ext"; }
};

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTDIALECT_H_
