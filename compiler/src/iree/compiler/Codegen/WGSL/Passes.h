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

//----------------------------------------------------------------------------//
// Register WGSL Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/WGSL/Passes.h.inc" // IWYU pragma: keep

void registerCodegenWGSLPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_WGSL_PASSES_H_
