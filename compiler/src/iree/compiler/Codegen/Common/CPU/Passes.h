// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the LLVMCPU Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_COMMON_CPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_CPU_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

/// Adds CPU bufferization passes to the pipeline.
void addCPUBufferizePasses(OpPassManager &funcPassManager);

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc" // IWYU pragma: keep

void registerCodegenCommonCPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_CPU_PASSES_H_
