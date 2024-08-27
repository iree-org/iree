// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the VMVX related Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_VMVX_PASSES_H_
#define IREE_COMPILER_CODEGEN_VMVX_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//------------------------------------------------------------------------------
// VMVX Pass Pipelines
//------------------------------------------------------------------------------

/// Populates the passes to lower to tiled/distributed/bufferized ops,
/// suitable for library call dispatch and lowering to loops.
void addVMVXDefaultPassPipeline(OpPassManager &funcPassManager,
                                bool enableUKernels);

//----------------------------------------------------------------------------//
// VMVX Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Populates passes needed to link HAL executables across VMVX targets.
void buildVMVXLinkingPassPipeline(OpPassManager &variantPassManager);

//----------------------------------------------------------------------------//
// Register VMVX Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/VMVX/Passes.h.inc" // IWYU pragma: keep

void registerCodegenVMVXPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_VMVX_PASSES_H_
