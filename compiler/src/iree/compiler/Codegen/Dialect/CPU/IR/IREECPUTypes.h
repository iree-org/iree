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

/// Callback type for CPU pipeline builders. Returns success if the pipeline
/// was handled.
using CPUPipelineBuilder =
    LogicalResult (*)(Attribute pipelineAttr, OpPassManager &pm,
                      const CodegenPipelineOptions *options);

/// Registers a CPU pipeline builder callback. Called from the LLVMCPU backend
/// at pass registration time. Thread-safe (uses std::call_once).
void registerCPUPipelineBuilder(CPUPipelineBuilder builder);

/// Returns all the tiling levels as integer values.
SmallVector<int> getTilingLevelsAsInts();

/// Returns the corresponding key string for `level`.
StringRef getTilingLevelName(TilingLevel level);

// Returns the TileSwizzle for the given intrinsic and operand index.
Codegen::TileSwizzle getIntrinsicSwizzle(MMAIntrinsic mma, int operandIdx);

// Returns the TileSwizzle for the given MMA attr and operand index.
Codegen::TileSwizzle getSwizzle(DataTiledMMAAttr mma, int operandIdx);

} // namespace mlir::iree_compiler::IREE::CPU

// clang-format on
#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
