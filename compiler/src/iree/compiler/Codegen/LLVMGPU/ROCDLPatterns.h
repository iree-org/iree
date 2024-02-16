// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLPATTERNS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLPATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

// Adds patterns that distributes vector.contract ops with nested layout
// annotations to amdgpu.mfma ops.
void populateGPUDistributeNestedLayoutContractAMDGPUPatterns(
    Value threadId, RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_ROCDLPATTERNS_H_
