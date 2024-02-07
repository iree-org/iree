// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_ROCDL_PASSES_H_
#define IREE_COMPILER_CODEGEN_ROCDL_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Creates a pass that calls a dynamic pipeline to progressively lower Linalg
/// with tensor semantics to ROCDL.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createROCDLLowerExecutableTargetPass();

/// Creates a pass to select the lowering strategy for converting to ROCDL.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createROCDLSelectLoweringStrategyPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerCodegenROCDLPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_ROCDL_PASSES_H_
