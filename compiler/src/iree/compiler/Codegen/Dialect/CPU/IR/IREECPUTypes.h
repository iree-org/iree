// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"

namespace mlir::iree_compiler::IREE::CPU {

/// Representation for all the supported tiling levels. All or just a subset of
/// them may be available in a valid configuration.
enum TilingLevel : unsigned {
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

/// Returns the corresponding key string for `level`.
StringRef getTilingLevelName(TilingLevel level);

} // namespace mlir::iree_compiler::IREE::CPU

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h.inc"
// clang-format on
#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUTYPES_H_
