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
// TODO: apply forwarding pattern.
template <typename TilingTransformOp, typename TileOrNumThreadSpec>
static Value buildTileAndFuseAndDistributeImpl(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizesOrNumThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles);

/// Call buildTileAndFuseAndDistributeImpl with ArrayRef<int64_t> tilesSizes.
template <typename TilingTransformOp>
static Value buildTileFuseDistWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);

/// Call buildTileAndFuseAndDistributeImpl with ArrayRef<int64_t> numThreads.
template <typename TilingTransformOp>
static Value buildTileFuseDistWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<int64_t> numThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);

/// Call buildTileAndFuseAndDistributeImpl with a handle to multiple numThreads.
template <typename TilingTransformOp>
static Value buildTileFuseDistWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    Value numThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr);

/// Apply patterns and vectorize (for now always applies rank-reduction).
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value buildVectorizeStrategy(ImplicitLocOpBuilder &b, Value funcH);

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

//===----------------------------------------------------------------------===//
// Target-specific strategies .
// TODO: Move the code below to a target-specific location.
//===----------------------------------------------------------------------===//

/// Return success if the IR matches what the GPU reduction strategy can handle.
/// If it is success it will append the transform dialect after the entry point
/// module.
LogicalResult matchAndSetGPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op);

LogicalResult matchAndSetCPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECT_STRATEGIES_H_
