// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_COMMON_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_COMMON_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct AbstractReductionStrategy;

//===----------------------------------------------------------------------===//
// General helpers.
//===----------------------------------------------------------------------===//
static constexpr int64_t kCudaWarpSize = 32;
static constexpr int64_t kCudaMaxNumThreads = 1024;
static constexpr int64_t kCudaMaxVectorLoadBitWidth = 128;

/// Return max(1, (value * 32) / bitWidth).
int64_t scaleUpByBitWidth(int64_t value, int64_t bitWidth);

/// Adjust the number of warps to use to benefit from packing multiple smaller
/// elemental types within a single 128 bit shuffled element.
int64_t adjustNumberOfWarpsForBlockShuffle(int64_t numWarpsToUse,
                                           int64_t bitWidth);

//===----------------------------------------------------------------------===//
// Low-level reusable retargetable builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//
/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildMapToBlockAndThreads(ImplicitLocOpBuilder& b, Value funcH,
                                ArrayRef<int64_t> blockSize);

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildDistributeVectors(ImplicitLocOpBuilder& b, Value variantH,
                             Value funcH, int64_t warpSize = kCudaWarpSize);

/// Take care of the last common steps in a GPU strategy (i.e. vectorize,
/// bufferize, maps to blocks and threads and distribute vectors).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
// TODO: abstract away AbstractReductionStrategy, this is supposed to be
// retargetable.
std::pair<Value, Value> buildCommonTrailingStrategy(
    ImplicitLocOpBuilder& b, Value variantH,
    const AbstractReductionStrategy& strategy);

//===----------------------------------------------------------------------===//
// Mid-level problem-specific strategy builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//
/// Given a handle `elementwiseH` to an elementwise op of rank `rank`, sizes
/// `elementwiseSizes` mapped to `numThreadsXInBlock` threads along dimension
/// x. Build a schedule that maps the most minor dimension to a scf.foreach op
/// itself mapped to the `gpu.thread x` dimension.
/// The schedule first performs a split of the largest possible multiple of
/// `numThreadsXInBlock * maxVectorSize` to form a maximally divisible region
/// Assumes the most minor dimension of the op is the last one.
// TODO: More robustness wrt selecting the most minor dimension otherwise
// performance may suffer.
// TODO: Split point should be dynamic and aware of future stride / alignment
// to also guarantee proper vector alignments.
void build1DSplittingStrategyWithOptionalThreadMapping(
    ImplicitLocOpBuilder& b, Value elementwiseH, int64_t rank,
    SmallVector<int64_t> elementwiseSizes, int64_t numThreads,
    int64_t maxVectorSize = 4);

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//
/// Return success if the IR matches what the GPU reduction strategy can
/// handle. If it is success it will append the transform dialect after the
/// entry point module.
LogicalResult matchAndSetReductionStrategy(func::FuncOp entryPoint,
                                           linalg::LinalgOp op);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_COMMON_H_
