// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_HAL_INLINE_IR_HALINLINEDIALECT_H_
#define IREE_COMPILER_MODULES_HAL_INLINE_IR_HALINLINEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Inline {

class HALInlineDialect : public Dialect {
public:
  explicit HALInlineDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "hal_inline"; }
};

} // namespace Inline
} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_MODULES_HAL_INLINE_IR_HALINLINEDIALECT_H_
