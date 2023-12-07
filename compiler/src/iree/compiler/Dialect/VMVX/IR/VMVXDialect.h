// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VMVX_IR_VMVXDIALECT_H_
#define IREE_COMPILER_DIALECT_VMVX_IR_VMVXDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::VMVX {

class VMVXDialect : public Dialect {
public:
  explicit VMVXDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "vmvx"; }
};

} // namespace mlir::iree_compiler::IREE::VMVX

#endif // IREE_COMPILER_DIALECT_VMVX_IR_VMVXDIALECT_H_
