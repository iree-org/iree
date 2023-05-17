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

namespace mlir {
namespace iree_compiler {
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
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_VMVX_PASSES_H_
