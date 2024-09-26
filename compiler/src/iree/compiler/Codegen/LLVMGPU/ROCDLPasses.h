// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLPASSES_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLPASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc" // IWYU pragma: keep
                                                           //
void registerCodegenROCDLPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLPASSES_H_
