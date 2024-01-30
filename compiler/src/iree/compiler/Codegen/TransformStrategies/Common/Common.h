// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_COMMON_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_COMMON_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
// Needed until IREE builds its own gpu::GPUBlockMappingAttr / gpu::Blocks
// attributes that are reusable across all targets.
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Base quantities generally useful for all CPU and GPU strategies.
//===----------------------------------------------------------------------===//
inline Attribute blockX(MLIRContext *ctx) {
  return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::MappingId::DimX);
}
inline Attribute blockY(MLIRContext *ctx) {
  return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::MappingId::DimY);
}
inline Attribute blockZ(MLIRContext *ctx) {
  return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::MappingId::DimZ);
}

struct AbstractReductionStrategy;

//===----------------------------------------------------------------------===//
// General helpers.
//===----------------------------------------------------------------------===//

/// Return the greatest value smaller or equal to `val` that is a multiple
/// of `multiple`. Asserts that all quantities are nonnegative. I.e. returns
/// `(val / multiple) * multiple` a.k.a `floordiv(val, multiple) * multiple`.
int64_t previousMultipleOf(int64_t val, int64_t multiple);

/// Return the smallest value greater or equal to `val` that is a multiple of
/// `multiple`. Asserts that all quantities are nonnegative.
/// I.e. returns `((val + multiple - 1) / multiple) * multiple`  a.k.a
///        a.k.a `ceildiv(val, multiple) * multiple`.
int64_t nextMultipleOf(int64_t val, int64_t multiple);

/// Find the highest divisor of `value` that is smaller than `limit`. This is
/// useful to capture any tiling that is guaranteed to keep the IR static.
/// Conservatively return failure when `limit` is greater than 1024 to avoid
/// prohibitively long compile time overheads.
// TODO: approximate with a faster implementation based on a few desirable
// primes.
FailureOr<int64_t> maxDivisorOfValueBelowLimit(int64_t value, int64_t limit);

using StrategyBuilderFn = std::function<void(ImplicitLocOpBuilder &, Value)>;

/// Use `buildStrategy` to build a ModuleOp containing transform dialect IR,
/// right after function `entryPoint`.
/// This embed the transform into the IR and allows applying it either in debug
/// mode or within the IREE pipeline.
void createTransformRegion(mlir::FunctionOpInterface entryPoint,
                           StrategyBuilderFn buildStrategy);

//===----------------------------------------------------------------------===//
// Low-level reusable builder APIs, these should follow MLIR-style builders.
//===----------------------------------------------------------------------===//

/// Build transform IR that prints `handles` in order, or print the whole IR if
/// `handles` is empty.
void buildPrint(ImplicitLocOpBuilder &b, ValueRange handles = {});

using ApplyPatternsOpBodyBuilderFn = std::function<void(OpBuilder &, Location)>;

/// Create an ApplyPatternsOp that performs a set of key canonicalizations and
/// so-called enabling transformations to normalize the IR.
/// In addition to the specified transform, perform the following ones:
///   canonicalization, tiling_canonicalization, licm and cse (in this order).
void buildCanonicalizationAndEnablingTransforms(
    ImplicitLocOpBuilder &b, Value funcH,
    ApplyPatternsOpBodyBuilderFn populatePatternsFn = nullptr);

/// Build transform IR to dynamically selects the first non-empty handle; i.e.
/// if (h1, h2) is:
///   - (non-empty, non-empty), returns (h1, h2)
///   - (empty, non-empty), returns (h2, empty)
///   - (non-empty, empty), returns (h1, empty)
///   - (empty, empty), returns (empty, empty)
/// This is used as a normalization operation that replaces conditionals, either
/// in C++ or in transform IR.
/// This can be thought of as a control-flow -> data-dependent conversion.
std::pair<Value, Value> buildSelectFirstNonEmpty(ImplicitLocOpBuilder &b,
                                                 Value handle1, Value handle2);

/// Result of the combined transform performing tiling, fusion and
/// distribution to parallel constructs.
struct TileToScfForAndFuseResult {
  /// Vector of `scf.for` loops containing the tiled and fused operations.
  SmallVector<Value> forLoops;
  /// Handles to fused operations other than the final consumer operation. May
  /// be empty if fusion was not performed iteratively.
  /// This is currently empty
  // TODO: support returning handles from `fuse_into_containing_op` and remove
  // the restriction above.
  SmallVector<Value> resultingFusedOpsHandles;
  /// Handle to the tiled final consumer operation.
  Value tiledOpH;
};

/// Build transform IR to perform multi-level tile and fuse into an scf.for op.
/// Note: fusion is currently unsupported.
TileToScfForAndFuseResult
buildTileFuseToScfFor(ImplicitLocOpBuilder &b, Value variantH, Value rootH,
                      ValueRange opsHToFuse, ArrayRef<OpFoldResult> tileSizes,
                      bool canonicalize = true);

/// Result of the combined transform performing tiling, fusion and
/// distribution to parallel constructs.
struct TileToForallAndFuseAndDistributeResult {
  /// Outer `scf.forall` loop containing the tiled and fused
  /// operations.
  Value forallH;
  /// Handles to fused operations other than the final consumer operation. May
  /// be empty if fusion was not performed iteratively.
  // TODO: support returning handles from `fuse_into_containing_op` and remove
  // the restriction above.
  SmallVector<Value> resultingFusedOpsHandles;
  /// Handle to the tiled final consumer operation.
  Value tiledOpH;
};

/// Build transform IR to perform the following transformations:
///   1. Tiles `rootH` to scf.forall to with `tileSizesOrNumThreads`
///      according to whether spec is a TileSizesSpec or a NumThreadsSpec.
///   2. Maps the resulting scf.forall to threads according to
///      `threadDimMapping`.
///   3. Iterates over `opsHToFuse` in order and fuses into the containing op.
///
/// Fusion operates in batch mode: a single fusion command is issued and a
/// topological sort is automatically computed by the fusion.
/// Since this applies a single fusion, no interleaved canonicalization / cse
/// / enabling transformation occurs and the resulting fusion may not be as
/// good.
///
/// In the future, an iterative mode in which the user is responsible for
/// providing the fusion order and has interleaved canonicalization / cse /
/// enabling transform will be introduced and may result in better fusions.
///
/// Note: this version cannot be used for the block-level tiling in a dispatch
/// region. `buildTileFuseDistToForallAndWorkgroupCountWithTileSizes` is
/// the modified version that is aware of the `workgroup_count` region.
///
// TODO: if someone knows how to properly export templates go for it .. sigh.
TileToForallAndFuseAndDistributeResult buildTileFuseDistToForallWithTileSizes(
    ImplicitLocOpBuilder &b, Value variantH, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping);

/// Similar to `buildTileFuseDistWithTileSizes` but using `numThreads` instead
/// of `tileSizes`.
TileToForallAndFuseAndDistributeResult buildTileFuseDistToForallWithNumThreads(
    ImplicitLocOpBuilder &b, Value variantH, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> numThreads, ArrayAttr threadDimMapping);

/// Build transform IR to split the reduction into a parallel and combiner part.
/// Then tile the parallel part and map it to `tileSize` threads, each reducing
/// on `vectorSize` elements.
/// Lastly, fuse the newly created fill and elementwise operations into the
/// resulting containing forall op.
/// Return a triple of handles to (forall, fill, combiner)
std::tuple<Value, Value, Value> buildTileReductionUsingScfForeach(
    ImplicitLocOpBuilder &b, Value isolatedParentOpH, Value reductionH,
    int64_t reductionRank, int64_t tileSize, int64_t reductionVectorSize,
    Attribute mappingAttr);

/// Build the transform IR to pad an op `opH`.
// TODO: Better upstream builder.
Value buildPad(ImplicitLocOpBuilder &b, Value opH,
               ArrayRef<Attribute> paddingValues,
               ArrayRef<int64_t> paddingDimensions,
               ArrayRef<int64_t> packingDimensions,
               ArrayRef<SmallVector<int64_t>> transposePaddings = {});

/// Build transform IR that applies rank-reduction patterns and vectorizes.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
/// If `applyCleanups` is true, also apply cleanup patterns.
Value buildVectorize(ImplicitLocOpBuilder &b, Value funcH,
                     bool applyCleanups = false, bool vectorizePadding = false,
                     bool vectorizeNdExtract = false);

/// Build transform IR that applies lowering of masked vector transfer
/// operations and subsequent cleanup patterns (fold-memref-aliases).
/// Takes a handle to a containing op and returns an updated handle to the
/// containing op.
void buildLowerMaskedTransfersAndCleanup(ImplicitLocOpBuilder &b, Value funcH,
                                         bool cleanup = true);

/// Build transform IR that applies vector mask lowering and subsequent cleanup
/// patterns (fold-memref-aliases).
/// Takes a handle to a containing op and returns an updated handle to the
/// containing op.
Value buildLowerVectorMasksAndCleanup(ImplicitLocOpBuilder &b, Value funcH,
                                      bool cleanup = true);

/// Build transform IR to hoist redundant subset operations.
void buildHoisting(ImplicitLocOpBuilder &b, Value funcH);

/// Build transform IR to bufferize and drop HAL descriptor from memref ops.
/// Takes a handle variantOp and returns a handle to the same variant op.
Value buildBufferize(ImplicitLocOpBuilder &b, Value variantH,
                     bool targetGpu = false);

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

/// Build transform IR to match exactly an N-D reduction operation (with
/// optional leading and trailing elementwise) and create a top-level
/// `scf.forall` tiled by `strategy.workgroupTileSizes`.
/// The matched `maybeLeadingH`, `fillH`, `reductionH` and `maybeTrailingH` are
/// fused into the top-level `scf.forall` and handles are returned to
/// the fused versions of these ops, in order, that are all tiled and
/// distributed accordingly. The scf.forall is returned as the last
/// value.
/// The mapping of the `scf.forall` dimensions is tied the first
/// dimensions of `strategy.allBlockAttrs`.
///
/// Note: `buildTileFuseDistToForallAndWorkgroupCountWithTileSizes` is
/// called internally, this version is only for the block-level tiling inside a
/// dispatch region with an attached workgroup_count region.
///
/// Note: the matching is enforced to be exact (i.e. no other compute ops may
/// exist under variantH). This is consistent with application confined within
/// the dispatch region, where we must not miss any op.
///
/// Note: A future version of this op will be able to directly apply on the DAG
/// and form the dispatch region.
std::tuple<Value, Value, Value, Value, Value>
buildReductionStrategyBlockDistribution(ImplicitLocOpBuilder &b, Value variantH,
                                        ArrayRef<int64_t> workgroupTileSizes);

/// Build transform IR that applies memory optimizations.
Value buildMemoryOptimizations(ImplicitLocOpBuilder &b, Value funcH);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_COMMON_H_
