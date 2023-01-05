// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECT_STRATEGIES_GPU_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Low-level reusable builder APIs, these should follow MLIR-style builders.
//===----------------------------------------------------------------------===//

static constexpr int64_t kCudaWarpSize = 32;
static constexpr int64_t kCudaMaxNumThreads = 1024;

/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildMapToBlockAndThreads(ImplicitLocOpBuilder &b, Value funcH,
                                ArrayRef<int64_t> blockSize);

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildDistributeVectors(ImplicitLocOpBuilder &b, Value variantH,
                             Value funcH, int64_t warpSize = kCudaWarpSize);

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//
/// Return success if the IR matches what the GPU reduction strategy can handle.
/// If it is success it will append the transform dialect after the entry point
/// module.
LogicalResult matchAndSetGPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECT_STRATEGIES_GPU_H_
