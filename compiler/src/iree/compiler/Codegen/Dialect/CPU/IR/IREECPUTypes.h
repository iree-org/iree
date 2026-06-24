// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUEnums.h.inc"

namespace mlir::iree_compiler::IREE::CPU {

/// Representation for all the supported tiling levels. All or just a subset of
/// them may be available in a valid configuration.
enum class TilingLevel {
  DistributionTiles = 0,
  CacheParallelTiles = 1,
  CacheReductionTiles = 2,
  VectorCommonParallelTiles = 3,
  VectorReductionTiles = 4,
  VectorInnerParallelTiles = 5,
  MaxNumTileLevels = 6,
  InvalidLevel = 7,
};

struct LoweringConfigLevelInfo {
  IREE::CPU::TilingLevel level;
  SmallVector<int64_t> sizes;
  SmallVector<bool> scalableFlags;
};

} // namespace mlir::iree_compiler::IREE::CPU

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h.inc"

namespace mlir::iree_compiler {
struct CodegenPipelineOptions;
} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::CPU {

/// Callback type for CPU pipeline builders. The callback receives a
/// PipelineAttr and must handle all LoweringPipeline enum values.
/// Returns success if the pipeline was built.
using CPUPipelineBuilder =
    LogicalResult (*)(Attribute pipelineAttr, OpPassManager &pm,
                      const CodegenPipelineOptions *options);

/// Registers the CPU pipeline builder callback. Must be called before
/// any compilation that uses #iree_cpu.pipeline attrs.
void registerCPUPipelineBuilder(CPUPipelineBuilder builder);

/// Returns all the tiling levels as integer values.
SmallVector<int> getTilingLevelsAsInts();

/// Returns the corresponding key string for `level`.
StringRef getTilingLevelName(TilingLevel level);

// Returns the TileSwizzle for the given intrinsic and operand index.
Codegen::TileSwizzle getIntrinsicSwizzle(MMAIntrinsic mma, int operandIdx);

// Returns the TileSwizzle for the given MMA attr and operand index.
Codegen::TileSwizzle getSwizzle(DataTiledMMAAttr mma, int operandIdx);

// Returns the architectural vector register file capacity, in bytes, that the
// inner-tiled MMA cost model may use to fit the union of ACC, LHS and RHS
// tiles. For ISAs with scalable vectors (e.g. SVE/SVE2) the vector length is
// treated as its 128-bit minimum — a deliberate simplification that produces
// good-enough `intrinsics_m`/`intrinsics_n` choices without leaking
// scalability into the cost model.
// Values: AVX/AVX2 = 16 × 32 B, AVX-512 = 32 × 64 B, SVE/SVE2 = 32 × 16 B.
int64_t getRegisterSpaceBytes(MMAIntrinsic intrinsic);

// True if `intr` is one of the `MMA_GENERIC_SCALAR_1x1x1_REG*` cases.
bool isGenericScalar(MMAIntrinsic intr);

// For an `MMA_GENERIC_SCALAR_1x1x1_REG*` intrinsic, returns the register
// budget encoded in the enum case (8 or 16). Asserts otherwise.
int64_t getGenericScalarRegisterBudget(MMAIntrinsic intr);

// Returns the (LHS, RHS, ACC) element types for `intrinsic`, with
// signedness preserved (e.g. `ui8` for vpdpbusd) — used by the cost model
// to match against an encoding's `element_types`. Tile types in IR are
// signless; `DataTiledMMAAttr::getUndistributedTileTypes` strips the
// signedness annotations.
std::tuple<Type, Type, Type> getABCElementTypes(MLIRContext *ctx,
                                                MMAIntrinsic intrinsic);

// Returns the (M, N, K) tile shape for a row-major-tile intrinsic, or
// nullopt otherwise (e.g. SVE).
std::optional<std::tuple<int64_t, int64_t, int64_t>>
getRowMajorTilesMNKShape(MMAIntrinsic intrinsic);

// Idempotently attaches the bitcode for ukernel `name` as
// `hal.executable.objects` on `op`, looking it up first in any
// user-supplied `hal.executable.objects` ancestor, then falling back to
// the global `EmbeddedDataDirectory` (populated at LLVMCPU plugin init).
// No-op if `op` already carries a matching entry. Returns true if a matching
// bitcode was found (and is now attached), false if none exists for `name` —
// callers use the bool to decide whether a ukernel is actually available.
// Used by both `LLVMCPUSelectUKernels` (at kernel-config time, so the bitcode
// is on the op for the rest of codegen) and by
// `UKernelProviderAttr::createAndReplaceWithUkernelOp` (as a backstop for
// tests / future paths that bypass SelectUKernels).
bool attachUKernelBitcodeOnOp(Operation *op, StringRef name);

} // namespace mlir::iree_compiler::IREE::CPU

// clang-format on
#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
