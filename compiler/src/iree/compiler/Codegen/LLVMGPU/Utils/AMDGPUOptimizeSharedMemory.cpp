// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/AMDGPU/Transforms/Transforms.h"

namespace mlir::iree_compiler {

void optimizeSharedMemoryReadsAndWrites(mlir::FunctionOpInterface funcOp) {
  SmallVector<memref::AllocOp> shmAllocOps;
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (!hasSharedMemoryAddressSpace(allocOp.getType()))
      return;
    shmAllocOps.push_back(allocOp);
  });
  for (auto allocOp : shmAllocOps) {
    if (failed(amdgpu::optimizeSharedMemoryReadsAndWrites(funcOp,
                                                          allocOp.getMemref())))
      return;
  }
}

} // namespace mlir::iree_compiler
