// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVMGPU_PASSES_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLLVMGPU_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass();

/// Convert Linalg ops to Vector.
std::unique_ptr<OperationPass<FuncOp>> createVectorizationPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createTileAndDistributeToThreads();

std::unique_ptr<OperationPass<FuncOp>> createRemoveSingleIterationLoopPass();

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVMGPU_PASSES_H_
