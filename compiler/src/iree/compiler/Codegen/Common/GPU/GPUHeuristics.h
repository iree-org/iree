// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler {

enum class GemmSize { NotSet, SmallGemm, MediumGemm, LargeGemm };

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const GemmSize &gemmSize);

/// Struct containing information about a matmul's shape and type.
struct GPUMatmulShapeType {
  SmallVector<int64_t, 2> mSizes;
  SmallVector<int64_t, 2> nSizes;
  SmallVector<int64_t, 2> kSizes;
  SmallVector<int64_t, 2> batchSizes;
  Type aType;
  Type bType;
  Type cType;
  GemmSize gemmSize = GemmSize::NotSet;

  // Number of horizontally fused operations.
  // Horizontal fusion: C1,C2 = fused_matmul(A, B1, B2) where A is shared.
  // Default 1 for regular matmul: C = A @ B.
  int64_t numHorizontallyFusedOps = 1;

  GPUMatmulShapeType(int64_t m, int64_t n, int64_t k, Type a, Type b, Type c,
                     int64_t numHorizontallyFusedOps = 1)
      : mSizes({m}), nSizes({n}), kSizes({k}), batchSizes({}), aType(a),
        bType(b), cType(c), numHorizontallyFusedOps(numHorizontallyFusedOps) {}
  GPUMatmulShapeType(ArrayRef<int64_t> m, ArrayRef<int64_t> n,
                     ArrayRef<int64_t> k, ArrayRef<int64_t> batch, Type a,
                     Type b, Type c, int64_t numHorizontallyFusedOps = 1)
      : mSizes(m), nSizes(n), kSizes(k), batchSizes(batch), aType(a), bType(b),
        cType(c), numHorizontallyFusedOps(numHorizontallyFusedOps) {}
};

/// Struct containing information about a GPU MMA intrinsic type.
struct GPUIntrinsicType : public GPUMatmulShapeType {
  IREE::Codegen::InnerTileDescAttrInterface mmaKind;
  GPUIntrinsicType(int64_t m, int64_t n, int64_t k, Type a, Type b, Type c,
                   IREE::Codegen::InnerTileDescAttrInterface kind)
      : GPUMatmulShapeType(m, n, k, a, b, c), mmaKind(kind) {}
  GPUIntrinsicType(ArrayRef<int64_t> m, ArrayRef<int64_t> n,
                   ArrayRef<int64_t> k, ArrayRef<int64_t> batch, Type a, Type b,
                   Type c, IREE::Codegen::InnerTileDescAttrInterface kind)
      : GPUMatmulShapeType(m, n, k, batch, a, b, c), mmaKind(kind) {}
};

/// Struct containing seed tile sizes for GPU MMA heuristics deduction logic.
struct GPUMMAHeuristicSeeds {
  // The best number of subgroups to use per workgroup
  int64_t bestSubgroupCountPerWorkgroup = 0;
  // The best number of total tiles along M*N dimensions per subgroup
  int64_t bestMNTileCountPerSubgroup = 0;
  // The best number of tiles along K dimension per subgroup
  int64_t bestKTileCountPerSubgroup = 0;
  // The best number of elements along K dimension per subgroup. This is
  // equivalent to `bestKTileCountPerSubgroup * bestIntrinsic.kSize`, for
  // some chosen intrinsic `bestIntrinsic`.
  int64_t bestKElementCountPerSubgroup = 0;
};

struct GPUMMASchedule {
  // The MMA intrinsic kind to use for this schedule.
  IREE::Codegen::InnerTileDescAttrInterface mmaKind;

  // Native MMA intrinsic sizes along M, N and K dimensions for a subgroup.
  SmallVector<int64_t, 2> mSizes;
  SmallVector<int64_t, 2> nSizes;
  SmallVector<int64_t, 2> kSizes;

  // Number of subgroups along each M and N dimension.
  SmallVector<int64_t, 2> mSubgroupCounts;
  SmallVector<int64_t, 2> nSubgroupCounts;

  // Tile sizes for each M, N, and K dimension. When there are multiple M, N,
  // or K dimensions, the intrinsic sizes are targeted to the innermost
  // dimension, and the outer dimensions can be thought of as unrolling factors
  // along M, N, or K.
  SmallVector<int64_t, 2> mTileSizes; // M tile sizes per subgroup.
  SmallVector<int64_t, 2> nTileSizes; // N tile sizes per subgroup.
  SmallVector<int64_t, 2> kTileSizes; // K tile sizes.

  // Constructor for multi M, N, K dim schedules.
  GPUMMASchedule(IREE::Codegen::InnerTileDescAttrInterface kind,
                 ArrayRef<int64_t> mIntrinsicSizes,
                 ArrayRef<int64_t> nIntrinsicSizes,
                 ArrayRef<int64_t> kIntrinsicSizes,
                 ArrayRef<int64_t> mSubgroupCounts,
                 ArrayRef<int64_t> nSubgroupCounts,
                 ArrayRef<int64_t> mTileSizes, ArrayRef<int64_t> nTileSizes,
                 ArrayRef<int64_t> kTileSizes)
      : mmaKind(kind), mSizes(mIntrinsicSizes), nSizes(nIntrinsicSizes),
        kSizes(kIntrinsicSizes), mSubgroupCounts(mSubgroupCounts),
        nSubgroupCounts(nSubgroupCounts), mTileSizes(mTileSizes),
        nTileSizes(nTileSizes), kTileSizes(kTileSizes) {}

  // Constructor for single M, N, K dim schedules.
  GPUMMASchedule(IREE::Codegen::InnerTileDescAttrInterface kind,
                 int64_t mIntrinsicSize, int64_t nIntrinsicSize,
                 int64_t kIntrinsicSize, int64_t mSubgroup, int64_t nSubgroup,
                 int64_t mTileSize, int64_t nTileSize, int64_t kTileSize)
      : mmaKind(kind), mSizes({mIntrinsicSize}), nSizes({nIntrinsicSize}),
        kSizes({kIntrinsicSize}), mSubgroupCounts({mSubgroup}),
        nSubgroupCounts({nSubgroup}), mTileSizes({mTileSize}),
        nTileSizes({nTileSize}), kTileSizes({kTileSize}) {}

  // Helper methods to get the total product of intrinsic sizes.
  int64_t getTotalMSize() const { return llvm::product_of(mSizes); }
  int64_t getTotalNSize() const { return llvm::product_of(nSizes); }
  int64_t getTotalKSize() const { return llvm::product_of(kSizes); }

  // Helper methods to get the total product of tile sizes.
  int64_t getTotalMTileSize() const { return llvm::product_of(mTileSizes); }
  int64_t getTotalNTileSize() const { return llvm::product_of(nTileSizes); }
  int64_t getTotalKTileSize() const { return llvm::product_of(kTileSizes); }

  // Helper methods to get the total product of subgroup counts.
  int64_t getTotalMSubgroupCount() const {
    return llvm::product_of(mSubgroupCounts);
  }
  int64_t getTotalNSubgroupCount() const {
    return llvm::product_of(nSubgroupCounts);
  }

  // Check if all schedule dimensions are single-element.
  bool hasSingleDimensions() const {
    return llvm::all_equal({size_t(1), mSubgroupCounts.size(),
                            nSubgroupCounts.size(), mTileSizes.size(),
                            nTileSizes.size(), kTileSizes.size(), mSizes.size(),
                            nSizes.size(), kSizes.size()});
  }
};

/// Returns a schedule for using one of the given MMA |intrinsics| to target the
/// input |problem|. Returns std::nullopt if we cannot find such a schedule.
FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, std::optional<int64_t> cuCount,
    bool transposedLhs = false, bool transposedRhs = false,
    bool canUpcastAcc = false, bool mustBeAligned = true,
    bool doCPromotion = false, int64_t splitReductionTripCnt = 0);

/// Returns a schedule for the pvMatmul in attention using one of the given MMA
/// |intrinsics| to target the given attention matmul problems, |qkMatmul|
/// and |pvMatmul|. Returns std::nullopt if we cannot find such a schedule.
FailureOr<std::pair<GPUMMASchedule, GPUMMASchedule>> deduceAttentionSchedule(
    const GPUMatmulShapeType &qkMatmul, const GPUMatmulShapeType &pvMatmul,
    ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &pvMatmulSeeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, bool transposedQ = false, bool transposedK = true,
    bool transposedV = false, bool canUpcastAcc = false,
    bool mustBeAligned = true);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUMMASchedule &schedule);

} // namespace mlir::iree_compiler
