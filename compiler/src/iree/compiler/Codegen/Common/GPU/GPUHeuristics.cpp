// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"

#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-gpu-heuristics"

using llvm::APIntOps::GreatestCommonDivisor;

namespace mlir::iree_compiler {

static int64_t calculateSharedMemoryUsedInBytes(const GPUMMASchedule &schedule,
                                                int64_t lhsBitwidth,
                                                int64_t rhsBitwidth) {
  int64_t tileM = schedule.mSize * schedule.mTileCount * schedule.mWarpCount;
  int64_t tileN = schedule.nSize * schedule.nTileCount * schedule.nWarpCount;
  int64_t tileK = schedule.kSize * schedule.kTileCount;
  return (tileM * tileK * lhsBitwidth + tileN * tileK * rhsBitwidth) / 8;
}

bool isValidSchedule(const GPUMatmulShapeType &problem,
                     const GPUMMASchedule &schedule, const bool mustBeAligned) {
  auto alignedMSize =
      mustBeAligned
          ? problem.mSize
          : llvm::divideCeil(problem.mSize, schedule.mSize) * schedule.mSize;
  auto alignedNSize =
      mustBeAligned
          ? problem.nSize
          : llvm::divideCeil(problem.nSize, schedule.nSize) * schedule.nSize;
  auto alignedKSize =
      mustBeAligned
          ? problem.kSize
          : llvm::divideCeil(problem.kSize, schedule.kSize) * schedule.kSize;
  bool isValidM = (alignedMSize % (schedule.mSize * schedule.mTileCount *
                                   schedule.mWarpCount)) == 0;
  bool isValidN = (alignedNSize % (schedule.nSize * schedule.nTileCount *
                                   schedule.nWarpCount)) == 0;
  bool isValidK = (alignedKSize % (schedule.kSize * schedule.kTileCount)) == 0;
  return isValidN && isValidM && isValidK;
}

FailureOr<GPUMMASchedule>
fitScheduleInSharedMemory(const GPUMatmulShapeType &problem,
                          ArrayRef<GPUMatmulShapeType> intrinsics,
                          GPUMMASchedule schedule,
                          int64_t sharedMemLimitInBytes, bool mustBeAligned) {
  int64_t lhsBitwidth =
      intrinsics[schedule.index].aType.getIntOrFloatBitWidth();
  int64_t rhsBitwidth =
      intrinsics[schedule.index].bType.getIntOrFloatBitWidth();

  while (!isValidSchedule(problem, schedule, mustBeAligned) ||
         calculateSharedMemoryUsedInBytes(schedule, lhsBitwidth, rhsBitwidth) >
             sharedMemLimitInBytes) {
    LLVM_DEBUG({
      llvm::dbgs() << "Shrinking schedule\n";
      llvm::dbgs() << "mSize: " << schedule.mSize << "\n";
      llvm::dbgs() << "nSize: " << schedule.nSize << "\n";
      llvm::dbgs() << "kSize: " << schedule.kSize << "\n";
      llvm::dbgs() << "mTileCount: " << schedule.mTileCount << "\n";
      llvm::dbgs() << "nTileCount: " << schedule.nTileCount << "\n";
      llvm::dbgs() << "kTileCount: " << schedule.kTileCount << "\n";
      llvm::dbgs() << "mWarpCount: " << schedule.mWarpCount << "\n";
      llvm::dbgs() << "nWarpCount: " << schedule.nWarpCount << "\n";
    });

    auto decrementIfPossible = [](int64_t &c) -> LogicalResult {
      if (c <= 1) {
        return failure();
      }
      --c;
      return success();
    };

    // Attempt to shrink the schedule along one of the dimensions.
    // TODO: A better solution should probably factor problem.mSize /
    // (mWarpCount * mTileCount * mSize) and then pop off the smallest factors
    // one at a time, preferably trying to keep the tile "generally square."
    if (succeeded(decrementIfPossible(schedule.mTileCount))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.nTileCount))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.kTileCount))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.mWarpCount))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.nWarpCount))) {
      continue;
    }

    // If no dimension can be shrunk, give up.
    return failure();
  }
  return schedule;
}

FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUMatmulShapeType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    bool canUpcastAcc, bool mustBeAligned) {
  for (auto [index, intrinsic] : llvm::enumerate(intrinsics)) {
    if (problem.aType != intrinsic.aType || problem.bType != intrinsic.bType) {
      continue; // Cannot use this intrinsic for mismatched types
    }
    if (problem.cType != intrinsic.cType) {
      auto isFpCase =
          isa<FloatType>(problem.cType) && isa<FloatType>(intrinsic.cType);
      auto isUpcast = problem.cType.getIntOrFloatBitWidth() <
                      intrinsic.cType.getIntOrFloatBitWidth();
      if (!(canUpcastAcc && isFpCase && isUpcast)) {
        continue; // Cannot use this intrinsic if not upcasting
      }
    }

    if (mustBeAligned && (problem.mSize % intrinsic.mSize != 0 ||
                          problem.nSize % intrinsic.nSize != 0 ||
                          problem.kSize % intrinsic.kSize != 0)) {
      continue; // Cannot use this intrinsic for misaligned cases
    }

    // Cannot use the intrinsic when the tile size is greater than problem size.
    // Because tiling is a no-op, and we can't infer tiling sizes from IR.
    if (!mustBeAligned &&
        (problem.mSize < intrinsic.mSize || problem.nSize < intrinsic.nSize ||
         problem.kSize < intrinsic.kSize)) {
      continue;
    }

    int64_t mTotalTileCount = llvm::divideCeil(problem.mSize, intrinsic.mSize);
    int64_t nTotalTileCount = llvm::divideCeil(problem.nSize, intrinsic.nSize);

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

    const uint64_t kTotalTileCount =
        llvm::divideCeil(problem.kSize, intrinsic.kSize);
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
    return fitScheduleInSharedMemory(
        problem, intrinsics,
        GPUMMASchedule{index, intrinsic.mSize, intrinsic.nSize, intrinsic.kSize,
                       mWarpCount, nWarpCount, mTileCount, nTileCount,
                       kTileCount},
        sharedMemLimitInBytes, mustBeAligned);
  }
  return failure();
}

} // namespace mlir::iree_compiler
