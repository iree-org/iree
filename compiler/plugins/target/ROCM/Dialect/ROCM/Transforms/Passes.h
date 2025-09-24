// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMPILER_PLUGINS_TARGET_ROCM_DIALECT_TRANSFORMS_PASSES_H_
#define COMPILER_PLUGINS_TARGET_ROCM_DIALECT_TRANSFORMS_PASSES_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::ROCM {

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "compiler/plugins/target/ROCM/Dialect/ROCM/Transforms/Passes.h.inc" // IWYU pragma: keep

void registerROCMTargetPasses();

} // namespace mlir::iree_compiler::IREE::ROCM

#endif // COMPILER_PLUGINS_TARGET_ROCM_DIALECT_TRANSFORMS_PASSES_H_
