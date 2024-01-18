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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//------------------------------------------------------------------------------
// VMVX passes
//------------------------------------------------------------------------------

// Lowers high level library calls from named ops and generics. This operates
// at the bufferized linalg level.
std::unique_ptr<Pass> createVMVXLowerLinalgMicrokernelsPass();

/// Materialize the encoding of operations. The layout to use for the encoded
/// operations are VMVX specific.
std::unique_ptr<OperationPass<func::FuncOp>>
createVMVXMaterializeEncodingPass();

/// Pass to select a lowering strategy for a hal.executable.variant operation.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createVMVXSelectLoweringStrategyPass();

/// Pass to lower the module an hal.executable.variant operation to external
/// dialect.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createVMVXLowerExecutableTargetPass();

/// Populates the passes to lower to tiled/distributed/bufferized ops,
/// suitable for library call dispatch and lowering to loops.
void addVMVXDefaultPassPipeline(OpPassManager &passManager,
                                bool enableUKernels);

//----------------------------------------------------------------------------//
// VMVX Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Assigns executable constant ordinals across all VMVX variants.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createVMVXAssignConstantOrdinalsPass();

/// Links VMVX HAL executables within the top-level program module.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createVMVXLinkExecutablesPass();

/// Populates passes needed to link HAL executables across VMVX targets.
void buildVMVXLinkingPassPipeline(OpPassManager &passManager);

//----------------------------------------------------------------------------//
// Register VMVX Passes
//----------------------------------------------------------------------------//

void registerCodegenVMVXPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_VMVX_PASSES_H_
