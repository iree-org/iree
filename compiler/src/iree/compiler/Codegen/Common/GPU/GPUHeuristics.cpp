// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-gpu-heuristics"

using llvm::APIntOps::GreatestCommonDivisor;

namespace mlir::iree_compiler {

std::optional<GPUMMASchedule>
deduceMMASchedule(const GPUMatmulShapeType &problem,
                  ArrayRef<GPUMatmulShapeType> intrinsics,
                  const GPUMMAHeuristicSeeds &seeds, bool canUpcastAcc) {
  for (auto [index, intrinsic] : llvm::enumerate(intrinsics)) {
    if (problem.aType != intrinsic.aType || problem.bType != intrinsic.bType) {
      continue; // Cannot use this intrinsic for mismatched types
    }
    if (problem.cType != intrinsic.cType) {
      auto isSameKind =
          isa<FloatType>(problem.cType) == isa<FloatType>(intrinsic.cType) ||
          isa<IntegerType>(problem.cType) == isa<IntegerType>(intrinsic.cType);
      auto isUpcast = problem.cType.getIntOrFloatBitWidth() <
                      intrinsic.cType.getIntOrFloatBitWidth();
      if (!(canUpcastAcc && isSameKind && isUpcast)) {
        continue; // Cannot use this intrinsic if not upcasting
      }
    }

    if (problem.mSize % intrinsic.mSize != 0 ||
        problem.nSize % intrinsic.nSize != 0 ||
        problem.kSize % intrinsic.kSize != 0) {
      continue; // Cannot use this intrinsic for misaligned cases
    }

    int64_t mTotalTileCount = problem.mSize / intrinsic.mSize;
    int64_t nTotalTileCount = problem.nSize / intrinsic.nSize;

    int64_t remainingWarps = seeds.bestSubgroupCountPerWorkgroup;
    int64_t remainingTiles = seeds.bestMNTileCountPerSubgroup;
    // Assign more warps to the M dimension (used later) to balance thread
    // counts along X and Y dimensions.
    int64_t warpSqrt = 1ull
                       << (llvm::divideCeil(llvm::Log2_64(remainingWarps), 2));
    int64_t tileSqrt = 1ull << (llvm::Log2_64(remainingTiles) / 2);

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

    const uint64_t kTotalTileCount = problem.kSize / intrinsic.kSize;
    APInt kGCD = GreatestCommonDivisor(
        APInt(64, kTotalTileCount), APInt(64, seeds.bestKTileCountPerSubgroup));
    int64_t kTileCount = kGCD.getSExtValue();

    LLVM_DEBUG({
      llvm::dbgs() << "chosen MMA schedule:\n";
      llvm::dbgs() << "  intrinsic (M, N, K) = (" << intrinsic.mSize << ", "
                   << intrinsic.nSize << ", " << intrinsic.kSize << ")\n";
      llvm::dbgs() << "  subgroup count (M, N) = (" << mWarpCount << ", "
                   << nWarpCount << ")\n";
      llvm::dbgs() << "  subgroup tile count (M, N, K) = (" << mTileCount
                   << ", " << nTileCount << ", " << kTileCount << ")\n";
    });
    return GPUMMASchedule{index,           intrinsic.mSize, intrinsic.nSize,
                          intrinsic.kSize, mWarpCount,      nWarpCount,
                          mTileCount,      nTileCount,      kTileCount};
  }
  return std::nullopt;
}

} // namespace mlir::iree_compiler
