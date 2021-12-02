// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SANDBOX_PASSES_H_
#define IREE_CODEGEN_SANDBOX_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to drive tile + fuse transformations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgFusePass();

/// Creates a pass to drive transformations on Linalg on tensors.
std::unique_ptr<OperationPass<FuncOp>> createLinalgSingleTilingExpertPass();

/// Creates a pass to driver the lowering of vector operations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgVectorLoweringPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace iree_compiler {
void registerSandboxPasses();
}

}  // namespace mlir
#endif  // IREE_CODEGEN_SANDBOX_PASSES_H_
