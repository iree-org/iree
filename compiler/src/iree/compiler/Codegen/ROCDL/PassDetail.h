// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_ROCDL_PASS_DETAIL_H_
#define IREE_COMPILER_CODEGEN_ROCDL_PASS_DETAIL_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_CLASSES
#include "iree/compiler/Codegen/ROCDL/Passes.h.inc"

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_ROCDL_PASS_DETAIL_H_
