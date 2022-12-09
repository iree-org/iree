// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECT_STRATEGIES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Low-level reusable builder APIs, these should follow MLIR-style builders.
//===----------------------------------------------------------------------===//

/// Prints `handles` in order. Prints the whole IR if `handles` is empty.
static void buildPrint(ImplicitLocOpBuilder &b, ValueRange handles = {});

/// Performs the following transformations:
///   1. Tiles `rootH` to scf.foreach_thread to with `tileSizesOrNumThreads`
///      according to whether spec is a TileSizesSpec or a NumThreadsSpec.
///   2. Maps the resulting scf.foreach_thread to threads according to
///      `threadDimMapping`.
///   3. Iterates over `opsHToFuse` in order and fuses into the containing op.
/// Returns a handle to the resulting scf.foreach_thread.
///
/// Fusion operates in batch mode: a single fusion command is issued and a
/// topological sort is automatically computed by the fusion.
/// Since this applies a single fusion, no interleaved canonicalization / cse /
/// enabling transformation occurs and the resulting fusion may not be as good.
///
/// In the future, an iterative mode in which the user is responsible for
/// providing the fusion order and has interleaved canonicalization / cse /
/// enabling transform will be introduced and may result in better fusions.
///
/// If `resultingFusedOpsHandles` is a non-null pointer, the fused operation are
/// appended in order.
///
// TODO: if someone knows how to properly export templates go for it .. sigh.
Value buildTileFuseDistToForeachThreadWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);
Value buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);

/// See buildTileFuseDistWithTileSizes.
Value buildTileFuseDistToForeachThreadWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> numThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);
Value buildTileFuseDistToForeachThreadAndWorgroupCountWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> numThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);

/// Apply patterns and vectorize (for now always applies rank-reduction).
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildVectorize(ImplicitLocOpBuilder &b, Value funcH);

/// Bufferize and drop HAL decriptor from memref ops.
/// Takes a handle variantOp and returns a handle to the same variant op.
Value buildBufferize(ImplicitLocOpBuilder &b, Value variantH,
                     bool targetGpu = false);

/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildMapToBlockAndThreads(ImplicitLocOpBuilder &b, Value funcH,
                                ArrayRef<int64_t> blockSize);

static constexpr unsigned kCudaWarpSize = 32;

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildDistributeVectors(ImplicitLocOpBuilder &b, Value variantH,
                             Value funcH, int64_t warpSize = kCudaWarpSize);

using StrategyBuilderFn = std::function<void(ImplicitLocOpBuilder &, Value)>;

void createTransformRegion(func::FuncOp entryPoint,
                           StrategyBuilderFn buildStrategy);

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//
/// Distribute to blocks using the current IREE lowering config.
// TODO: consider passing a problem-specific struct to control information.
Value createReductionStrategyBlockDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, Value originalFillH,
    Value reductionH, Value optionalFusionRootH,
    ArrayRef<OpFoldResult> tileSizes0Generic, bool hasLeadingEltwise = false,
    bool hasTrailingEltwise = false);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECT_STRATEGIES_H_
