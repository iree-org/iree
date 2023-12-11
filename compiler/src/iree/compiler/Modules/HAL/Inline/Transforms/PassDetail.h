// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_HAL_INLINE_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_MODULES_HAL_INLINE_TRANSFORMS_PASS_DETAIL_H_

#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Inline {

#define GEN_PASS_CLASSES
#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h.inc"

} // namespace Inline
} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_MODULES_HAL_INLINE_TRANSFORMS_PASS_DETAIL_H_
