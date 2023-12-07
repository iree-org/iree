// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VMVX_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_VMVX_TRANSFORMS_PASS_DETAIL_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VMVX {

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h.inc"

} // namespace mlir::iree_compiler::IREE::VMVX

#endif // IREE_COMPILER_DIALECT_VMVX_TRANSFORMS_PASS_DETAIL_H_
