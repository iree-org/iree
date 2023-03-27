// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

/// Structure to represent target features.
struct TargetInfo {
  // TODO: add finer grain control for other tensorcore types.
  bool hasTF32TensorCore = false;
  bool hasWarpShuffle = false;
};

LogicalResult initGPULaunchConfig(ModuleOp moduleOp);

/// Attempts to find entry point of the given op, afterwards, it finds TargetInfo. 
FailureOr<TargetInfo> getTargetInfoFromAnyOp(Operation *op);

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_
