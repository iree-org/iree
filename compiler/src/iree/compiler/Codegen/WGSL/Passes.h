// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the WGSL Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_WGSL_PASSES_H_
#define IREE_COMPILER_CODEGEN_WGSL_PASSES_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

// Removes push constants by replacing hal.interface.constant.loads with
// hal.interface.binding.subspan + flow.dispatch.tensor.load.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createWGSLReplacePushConstantsPass();

//----------------------------------------------------------------------------//
// Register WGSL Passes
//----------------------------------------------------------------------------//

void registerCodegenWGSLPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_WGSL_PASSES_H_
