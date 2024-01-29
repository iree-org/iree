// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
