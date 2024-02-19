// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Returns the cooperative matrix (M, N, K) sizes that are supported by the
/// target environment and match the given parameters.
static std::optional<CooperativeMatrixSize>
getCooperativeMatrixSize(spirv::ResourceLimitsAttr resourceLimits,
                         const unsigned numSubgroupsPerWorkgroup,
                         const unsigned numMNTilesPerSubgroup, Type aType,
                         Type bType, Type cType, int64_t m, int64_t n,
                         int64_t k) {
  auto properties =
      resourceLimits.getCooperativeMatrixPropertiesKhr()
          .getAsRange<spirv::CooperativeMatrixPropertiesKHRAttr>();
  for (auto property : properties) {
    if (property.getAType() != aType || property.getBType() != bType ||
        property.getCType() != cType || property.getResultType() != cType ||
        property.getScope().getValue() != spirv::Scope::Subgroup) {
      continue; // Cannot use this cooperative matrix configuration
    }

    const unsigned matmulM = property.getMSize();
    const unsigned matmulN = property.getNSize();
    const unsigned matmulK = property.getKSize();
    if (m % matmulM != 0 || n % matmulN != 0 || k % matmulK != 0)
      continue;

    uint64_t nTotalTileCount = n / matmulN;
    uint64_t mTotalTileCount = m / matmulM;

    uint64_t remainingWarps = numSubgroupsPerWorkgroup;
    uint64_t remainingTiles = numMNTilesPerSubgroup;
    // Assign more warps to the M dimension (used later) to balance thread
    // counts along X and Y dimensions.
    uint64_t warpSqrt = 1ull << (divideCeil(llvm::Log2_64(remainingWarps), 2));
    uint64_t tileSqrt = 1ull << (llvm::Log2_64(remainingTiles) / 2);

    int64_t mWarpCount = 0, nWarpCount = 0;
    int64_t mTileCount = 0, nTileCount = 0;

    // See if the square root can divide mTotalTileCount. If so it means we can
    // distribute to both dimensions evenly. Otherwise, try to distribute to N
    // and then M.
    if (mTotalTileCount > (warpSqrt * tileSqrt) &&
        mTotalTileCount % (warpSqrt * tileSqrt) == 0) {
      mWarpCount = warpSqrt;
      mTileCount = tileSqrt;

      remainingWarps /= warpSqrt;
      remainingTiles /= tileSqrt;

      APInt nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCount),
                                         APInt(64, remainingWarps));
      nWarpCount = nGCD.getSExtValue();
      nTotalTileCount /= nWarpCount;
      remainingWarps /= nWarpCount;

      nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCount),
                                   APInt(64, remainingTiles));
      nTileCount = nGCD.getSExtValue();
    } else {
      APInt nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCount),
                                         APInt(64, remainingWarps));
      nWarpCount = nGCD.getSExtValue();
      nTotalTileCount /= nWarpCount;
      remainingWarps /= nWarpCount;

      nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCount),
                                   APInt(64, remainingTiles));
      nTileCount = nGCD.getSExtValue();
      remainingTiles /= nTileCount;

      APInt mGCD = GreatestCommonDivisor(APInt(64, mTotalTileCount),
                                         APInt(64, remainingWarps));
      mWarpCount = mGCD.getSExtValue();
      mTotalTileCount /= mWarpCount;
      remainingWarps /= mWarpCount;

      mGCD = GreatestCommonDivisor(APInt(64, mTotalTileCount),
                                   APInt(64, remainingTiles));
      mTileCount = mGCD.getSExtValue();
    }

    const uint64_t kTotalTileCount = k / matmulK;
    APInt kGCD = GreatestCommonDivisor(APInt(64, kTotalTileCount),
                                       APInt(64, numTilesPerSubgroupDimK));
    int64_t kTileCount = kGCD.getSExtValue();

    LLVM_DEBUG({
      llvm::dbgs() << "chosen cooperative matrix configuration:\n";
      llvm::dbgs() << "  (M, N, K) size = (" << matmulM << ", " << matmulN
                   << ", " << matmulK << ")\n";
      llvm::dbgs() << "  (M, N) subgroup count = (" << mWarpCount << ", "
                   << nWarpCount << ")\n";
      llvm::dbgs() << "  (M, N, K) tile count per subgroup = (" << mTileCount
                   << ", " << nTileCount << ", " << kTileCount << ")\n";
    });
    return CooperativeMatrixSize{matmulM,    matmulN,    matmulK,
                                 mWarpCount, nWarpCount, mTileCount,
                                 nTileCount, kTileCount};
  }
  return std::nullopt;
}
