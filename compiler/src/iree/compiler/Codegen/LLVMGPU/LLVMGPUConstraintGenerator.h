// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

/// LLVMGPU constraint emitter callback. Suitable for use as a
/// GPUConstraintEmitter registered via registerGPUPipelineCallbacks.
LogicalResult emitLLVMGPUConstraints(Attribute pipelineAttr,
                                     ArrayRef<Operation *> rootOps);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPUCONSTRAINTGENERATOR_H_
