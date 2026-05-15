// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//----------------------------------------------------------------------------//
// Register Common Transformation Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Transforms/Passes.h.inc" // IWYU pragma: keep

void registerTransformsPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TRANSFORMS_PASSES_H_
