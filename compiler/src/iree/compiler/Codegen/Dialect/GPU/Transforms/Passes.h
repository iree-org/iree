
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::GPU {
#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc" // IWYU pragma: keep
} // namespace mlir::iree_compiler::IREE::GPU

namespace mlir::iree_compiler {
/// Register GPU passes.
void registerIREEGPUPasses();
} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES_H_
