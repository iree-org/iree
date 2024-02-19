// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Types.h"

namespace mlir::iree_compiler {

/// Struct containing information about a matmul's shape and type.
struct GPUMatmulShapeType {
  int64_t mSize;
  int64_t nSize;
  int64_t kSize;
  Type aType;
  Type bType;
  Type cType;

  GPUMatmulShapeType(int64_t m, int64_t n, int64_t k, Type a, Type b, Type c)
      : mSize(m), nSize(n), kSize(k), aType(a), bType(b), cType(c) {}
};

/// Struct containing seed tile sizes for GPU MMA heuristics deduction logic.
struct GPUMMAHeuristicSeeds {
  // The default number of subgroups to use per workgroup
  int64_t numSubgroupsPerWorkgroup;
  // The default number of tiles along M/N dimension to use per workgroup
  int64_t numMNTilesPerSubgroup;
  // The default number of tiles along K dimension to use per subgroup
  int64_t numKTilesPerSubgroup;
};

struct GPUMMASchedule {
  int64_t mSize;      // Native MMA size along M dimension
  int64_t nSize;      // Native MMA size along N dimension
  int64_t kSize;      // Native MMA size along K dimension
  int64_t mWarpCount; // Number of subgroups along M dimension
  int64_t nWarpCount; // Number of subgroups along N dimension
  int64_t mTileCount; // Number of tiles per subgroup along M dimension
  int64_t nTileCount; // Number of tiles per subgroup along N dimension
  int64_t kTileCount; // Number of tiles along K dimension
};

/// Returns a schedule for using one of the given MMA |intrinsics| to target the
/// input |problem|. Returns std::nullopt if we cannot find such a schedule.
std::optional<GPUMMASchedule>
deduceMMASchedule(const GPUMatmulShapeType &problem,
                  ArrayRef<GPUMatmulShapeType> intrinsics,
                  const GPUMMAHeuristicSeeds &seeds);

} // namespace mlir::iree_compiler
