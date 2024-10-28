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
  int64_t mSize; // Native MMA intrinsic size along M dimension for a subgroup.
  int64_t nSize; // Native MMA intrinsic size along N dimension for a subgroup.
  int64_t kSize; // Native MMA intrinsic size along K dimension for a subgroup.

  // Number of subgroups along each M and N dimension.
  SmallVector<int64_t> mSubgroupCounts;
  SmallVector<int64_t> nSubgroupCounts;

  // Tile sizes for each M, N, and K dimension. When there are multiple M, N,
  // or K dimensions, the intrinsic sizes are targeted to the innermost
  // dimension, and the outer dimensions can be thought of as unrolling factors
  // along M, N, or K.
  SmallVector<int64_t> mTileSizes; // M tile sizes per subgroup.
  SmallVector<int64_t> nTileSizes; // N tile sizes per subgroup.
  SmallVector<int64_t> kTileSizes; // K tile sizes.

  // Constructor for multi M, N, K dim schedules.
  GPUMMASchedule(uint64_t i, int64_t mIntrinsicSize, int64_t nIntrinsicSize,
                 int64_t kIntrinsicSize, SmallVector<int64_t> mSubgroupCounts,
                 SmallVector<int64_t> nSubgroupCounts,
                 SmallVector<int64_t> mTileSizes,
                 SmallVector<int64_t> nTileSizes,
                 SmallVector<int64_t> kTileSizes)
      : index(i), mSize(mIntrinsicSize), nSize(nIntrinsicSize),
        kSize(kIntrinsicSize), mSubgroupCounts(mSubgroupCounts),
        nSubgroupCounts(nSubgroupCounts), mTileSizes(mTileSizes),
        nTileSizes(nTileSizes), kTileSizes(kTileSizes) {}

  // Constructor for single M, N, K dim schedules.
  GPUMMASchedule(uint64_t i, int64_t mIntrinsicSize, int64_t nIntrinsicSize,
                 int64_t kIntrinsicSize, int64_t mSubgroup, int64_t nSubgroup,
                 int64_t mTileSize, int64_t nTileSize, int64_t kTileSize)
      : index(i), mSize(mIntrinsicSize), nSize(nIntrinsicSize),
        kSize(kIntrinsicSize), mSubgroupCounts({mSubgroup}),
        nSubgroupCounts({nSubgroup}), mTileSizes({mTileSize}),
        nTileSizes({nTileSize}), kTileSizes({kTileSize}) {}
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUMMASchedule &schedule);

} // namespace mlir::iree_compiler
