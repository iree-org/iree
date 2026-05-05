// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_MMAUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_MMAUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// HoistableConversionOp tag pairs used by inner_tiled lowerings.
//===----------------------------------------------------------------------===//
//
// Each pair labels a `IREE::Util::HoistableConversionOp` and its inverse so
// `IREE::Util::eliminateHoistableConversions` can cancel them across
// reduction-loop boundaries. The pairing is by string match, so both halves of
// a pair must agree on the spelling — keeping them all here, in one place,
// avoids the interaction-at-a-distance hazard of having paired strings defined
// in separate translation units.
//
// Currently these are emitted from:
//   * `buildDataTiledMMAUnderlyingOperations` below (CPU + GPU DataTiledMMA),
//     and from GPU's `DataTiledScaledMMAAttr::buildUnderlyingOperations`
//     which reuses the helpers below: kDataTiledAcc{Distribute,Reassemble}.
//   * GPU `createMmaOp` for RDNA3 WMMA half-acc widening:
//     kRdna3{Interleave,Deinterleave}Acc.
//   * GPU `createMmaOp` for AMD VDMFMA acc handling:
//     kVDMFMA{Interleave,Deinterleave}Acc.

inline constexpr llvm::StringLiteral kDataTiledAccDistribute =
    "data_tiled_acc_distribute";
inline constexpr llvm::StringLiteral kDataTiledAccReassemble =
    "data_tiled_acc_reassemble";

inline constexpr llvm::StringLiteral kRdna3InterleaveAcc =
    "rdna3_interleave_acc";
inline constexpr llvm::StringLiteral kRdna3DeinterleaveAcc =
    "rdna3_deinterleave_acc";

inline constexpr llvm::StringLiteral kVDMFMAInterleaveAcc =
    "vdmfma_interleave_acc";
inline constexpr llvm::StringLiteral kVDMFMADeinterleaveAcc =
    "vdmfma_deinterleave_acc";

//===----------------------------------------------------------------------===//
// Shared body of DataTiledMMAAttr::buildUnderlyingOperations, plus the
// distribution helpers it relies on. Both helpers are also reused by GPU's
// `DataTiledScaledMMAAttr::buildUnderlyingOperations`, which has a different
// operand layout (extra scales) but the same data-tiled distribution model.
//===----------------------------------------------------------------------===//

/// Increments `indices` row-major under `sizes`. Returns false on overflow.
bool incrementIndices(MutableArrayRef<int64_t> indices,
                      ArrayRef<int64_t> sizes);

/// Distributes a multi-MMA-level tile `value` into per-intrinsic 1-D slices.
/// The input must already be in the swizzle's distributed N-D form (rank
/// equals `swizzle.getExpandedSize()`, with CrossThread dim sizes collapsed
/// to 1). Iterates cross-intrinsic indices in row-major order; each returned
/// value has the swizzle's "internal-only" shape, flattened to 1-D.
SmallVector<Value>
distributeMmaFragmentToIntrinsics(OpBuilder &builder, Location loc, Value value,
                                  const TileSwizzle &swizzle);

/// Callback emitting one architecture-specific MMA intrinsic op. Receives flat
/// 1-D operands sized for one intrinsic invocation. Must return a value with
/// the same type as the input `acc`, or a null Value to signal failure.
using DataTiledMMAIntrinsicEmitter = llvm::function_ref<Value(
    OpBuilder &builder, Location loc, Value lhs, Value rhs, Value acc)>;

/// Common body of `DataTiledMMAAttr::buildUnderlyingOperations`, used by both
/// CPU and GPU. Distributes `inputs` and `outputs[0]` into per-intrinsic flat
/// vectors using the three swizzles, runs an `(mu, nu, ku)` unroll loop
/// calling `emitIntrinsic` for each combination, and reassembles the final
/// accumulator into the original output type. The accumulator distribute and
/// reassemble are wrapped in `IREE::Util::HoistableConversionOp` pairs (tagged
/// kDataTiledAcc{Distribute,Reassemble}) so they can sink/hoist out of
/// reduction loops.
///
/// `inputs` are LHS and RHS in that order. `outputs[0]` is the accumulator.
/// Operand vector ranks may be either the swizzle's full expanded form (GPU)
/// or the 2-D (outer × inner) collapsed form (CPU); the helper reshapes as
/// needed.
LogicalResult buildDataTiledMMAUnderlyingOperations(
    OpBuilder &builder, Location loc, const TileSwizzle &lhsSwizzle,
    const TileSwizzle &rhsSwizzle, const TileSwizzle &accSwizzle,
    int64_t intrinsicsM, int64_t intrinsicsN, int64_t intrinsicsK,
    ValueRange inputs, ValueRange outputs,
    DataTiledMMAIntrinsicEmitter emitIntrinsic,
    SmallVectorImpl<Value> &results);

} // namespace mlir::iree_compiler::IREE::Codegen

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_MMAUTILS_H_
