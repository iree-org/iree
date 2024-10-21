// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Types.h"

namespace mlir::iree_compiler {

/// Struct containing information about a matmul's shape and type.
struct GPUMatmulShapeType {
  SmallVector<int64_t> mSizes;
  SmallVector<int64_t> nSizes;
  SmallVector<int64_t> kSizes;
  Type aType;
  Type bType;
  Type cType;

  GPUMatmulShapeType(int64_t m, int64_t n, int64_t k, Type a, Type b, Type c)
      : mSizes({m}), nSizes({n}), kSizes({k}), aType(a), bType(b), cType(c) {}
  GPUMatmulShapeType(SmallVector<int64_t> m, SmallVector<int64_t> n,
                     SmallVector<int64_t> k, Type a, Type b, Type c)
      : mSizes(m), nSizes(n), kSizes(k), aType(a), bType(b), cType(c) {}
};

/// Struct containing seed tile sizes for GPU MMA heuristics deduction logic.
struct GPUMMAHeuristicSeeds {
  // The best number of subgroups to use per workgroup
  int64_t bestSubgroupCountPerWorkgroup;
  // The best number of total tiles along M*N dimensions per subgroup
  int64_t bestMNTileCountPerSubgroup;
  // The best number of tiles along K dimension per subgroup
  int64_t bestKTileCountPerSubgroup;
  // The best number of elements along K dimension per subgroup. This is
  // equivalent to `bestKTileCountPerSubgroup * bestIntrinsic.kSize`, for
  // some chosen intrinsic `bestIntrinsic`.
  int64_t bestKElementCountPerSubgroup = 0;
};

struct GPUMMASchedule {
  // Index of the chosen intrinsic into the list of given MMA intrinsics
  uint64_t index;
  int64_t mSize; // Native MMA size along M dimension
  int64_t nSize; // Native MMA size along N dimension
  int64_t kSize; // Native MMA size along K dimension

  // Number of subgroups along M dimensions
  SmallVector<int64_t> mSubgroupCounts;
  // Number of subgroups along N dimensions
  SmallVector<int64_t> nSubgroupCounts;
  // Number of tiles per subgroup along M dimensions
  SmallVector<int64_t> mTileCounts;
  // Number of tiles per subgroup along N dimensions
  SmallVector<int64_t> nTileCounts;
  // Number of tiles along K dimensions
  SmallVector<int64_t> kTileCounts;

  GPUMMASchedule(uint64_t i, int64_t m, int64_t n, int64_t k, int64_t mSubgroup,
                 int64_t nSubgroup, int64_t mTile, int64_t nTile, int64_t kTile)
      : index(i), mSize(m), nSize(n), kSize(k), mSubgroupCounts({mSubgroup}),
        nSubgroupCounts({nSubgroup}), mTileCounts({mTile}),
        nTileCounts({nTile}), kTileCounts({kTile}) {}
  GPUMMASchedule(uint64_t i, int64_t m, int64_t n, int64_t k,
                 SmallVector<int64_t> mSubgroup, SmallVector<int64_t> nSubgroup,
                 SmallVector<int64_t> mTile, SmallVector<int64_t> nTile,
                 SmallVector<int64_t> kTile)
      : index(i), mSize(m), nSize(n), kSize(k), mSubgroupCounts(mSubgroup),
        nSubgroupCounts(nSubgroup), mTileCounts(mTile), nTileCounts(nTile),
        kTileCounts(kTile) {}
};

/// Returns a schedule for using one of the given MMA |intrinsics| to target the
/// input |problem|. Returns std::nullopt if we cannot find such a schedule.
FailureOr<GPUMMASchedule>
deduceMMASchedule(const GPUMatmulShapeType &problem,
                  ArrayRef<GPUMatmulShapeType> intrinsics,
                  const GPUMMAHeuristicSeeds &seeds,
                  int64_t sharedMemLimitInBytes, int64_t subgroupSize,
                  bool transposedLhs = false, bool transposedRhs = false,
                  bool canUpcastAcc = false, bool mustBeAligned = true);

/// Returns a schedule for the pvMatmul in attention using one of the given MMA
/// |intrinsics| to target the given attention matmul problems, |qkMatmul|
/// and |pvMatmul|. Returns std::nullopt if we cannot find such a schedule.
FailureOr<GPUMMASchedule> deduceAttentionSchedule(
    const GPUMatmulShapeType &qkMatmul, const GPUMatmulShapeType &pvMatmul,
    ArrayRef<GPUMatmulShapeType> intrinsics,
    const GPUMMAHeuristicSeeds &pvMatmulSeeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, bool transposedQ = false, bool transposedK = true,
    bool transposedV = false, bool canUpcastAcc = false,
    bool mustBeAligned = true);

} // namespace mlir::iree_compiler
