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

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const GPUMMASchedule &schedule) {
  os << "mSize: " << schedule.mSize << ", ";
  os << "nSize: " << schedule.nSize << ", ";
  os << "kSize: " << schedule.kSize << ", ";
  os << "mTileCount: " << schedule.mTileCount << ", ";
  os << "nTileCount: " << schedule.nTileCount << ", ";
  os << "kTileCount: " << schedule.kTileCount << ", ";
  os << "mWarpCount: " << schedule.mWarpCount << ", ";
  os << "nWarpCount: " << schedule.nWarpCount;
  return os;
}

static int64_t calculateSharedMemoryUsedInBytes(const GPUMMASchedule &schedule,
                                                int64_t lhsBitwidth,
                                                int64_t rhsBitwidth) {
  int64_t tileM = schedule.mSize * schedule.mTileCount * schedule.mWarpCount;
  int64_t tileN = schedule.nSize * schedule.nTileCount * schedule.nWarpCount;
  int64_t tileK = schedule.kSize * schedule.kTileCount;
  return (tileM * tileK * lhsBitwidth + tileN * tileK * rhsBitwidth) / 8;
}

static bool isScheduleAligned(const GPUMatmulShapeType &problem,
                              const GPUMMASchedule &schedule,
                              bool mustBeAligned) {
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
  return isValidM && isValidN && isValidK;
}

static bool isValidMMASchedule(const GPUMatmulShapeType &problem,
                               const GPUMMASchedule &schedule,
                               bool mustBeAligned, int64_t subgroupSize,
                               bool transposedLhs, bool transposedRhs) {
  bool isAligned = isScheduleAligned(problem, schedule, mustBeAligned);

  // Constraint to ensure wgTileSize is distributable by wgSize.
  // such that we can distribute to it's corresponding vector.transfer_read.
  const int64_t kMaxVectorLoadBitWidth = 128;
  int64_t elemsPerThread =
      kMaxVectorLoadBitWidth / problem.bType.getIntOrFloatBitWidth();
  int64_t wgThreads = schedule.mWarpCount * schedule.nWarpCount * subgroupSize;

  int64_t mWgSize = schedule.mSize * schedule.mTileCount * schedule.mWarpCount;
  int64_t nWgSize = schedule.nSize * schedule.nTileCount * schedule.nWarpCount;
  int64_t kWgSize = schedule.kSize * schedule.kTileCount;
  int64_t innerLhsDimSize = transposedLhs ? mWgSize : kWgSize;
  int64_t innerRhsDimSize = transposedRhs ? kWgSize : nWgSize;

  bool isDistributableLhs =
      (innerLhsDimSize / elemsPerThread) % wgThreads == 0 ||
      wgThreads % (innerLhsDimSize / elemsPerThread) == 0;
  bool isDistributableRhs =
      (innerRhsDimSize / elemsPerThread) % wgThreads == 0 ||
      wgThreads % (innerRhsDimSize / elemsPerThread) == 0;

  return isAligned && isDistributableLhs && isDistributableRhs;
}

static FailureOr<GPUMMASchedule> fitScheduleInSharedMemory(
    GPUMatmulShapeType intrinsic, GPUMMASchedule schedule,
    llvm::function_ref<bool(const GPUMMASchedule &schedule)> isScheduleValid) {

  while (!isScheduleValid(schedule)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Chosen schedule is invalid:\n";
      llvm::dbgs() << schedule << "\n";
      llvm::dbgs() << "Shrinking schedule...\n";
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

  LLVM_DEBUG({
    llvm::dbgs() << "Chosen schedule is valid:\n";
    llvm::dbgs() << schedule << "\n";
  });

  return schedule;
}

static LogicalResult canTargetIntrinsic(const GPUMatmulShapeType &problem,
                                        const GPUMatmulShapeType &intrinsic,
                                        bool canUpcastAcc, bool mustBeAligned) {
  if (problem.aType != intrinsic.aType || problem.bType != intrinsic.bType) {
    return failure(); // Cannot use this intrinsic for mismatched types
  }
  if (problem.cType != intrinsic.cType) {
    bool isFpCase =
        isa<FloatType>(problem.cType) && isa<FloatType>(intrinsic.cType);
    bool isUpcast = problem.cType.getIntOrFloatBitWidth() <
                    intrinsic.cType.getIntOrFloatBitWidth();
    if (!(canUpcastAcc && isFpCase && isUpcast)) {
      return failure(); // Cannot use this intrinsic if not upcasting.
    }
  }

  if (mustBeAligned && (problem.mSize % intrinsic.mSize != 0 ||
                        problem.nSize % intrinsic.nSize != 0 ||
                        problem.kSize % intrinsic.kSize != 0)) {
    return failure(); // Cannot use this intrinsic for misaligned cases.
  }

  // Cannot use the intrinsic when the tile size is greater than problem size.
  // Because tiling is a no-op, and we can't infer tiling sizes from IR.
  if (!mustBeAligned &&
      (problem.mSize < intrinsic.mSize || problem.nSize < intrinsic.nSize ||
       problem.kSize < intrinsic.kSize)) {
    return failure();
  }

  return success();
}

/// Choose an optimal mma schedule with the heuristic that minimized the total
/// amount of data read from global memory, per workgroup, respecting the
/// heuristic seeds.
static GPUMMASchedule getOptimalMMASchedule(const GPUMatmulShapeType &problem,
                                            const GPUMatmulShapeType &intrinsic,
                                            const GPUMMAHeuristicSeeds &seeds,
                                            uint64_t intrinsicIndex) {
  int64_t mTotalTileCount = llvm::divideCeil(problem.mSize, intrinsic.mSize);
  int64_t nTotalTileCount = llvm::divideCeil(problem.nSize, intrinsic.nSize);

  int64_t remainingWarps = seeds.bestSubgroupCountPerWorkgroup;
  int64_t remainingTiles = seeds.bestMNTileCountPerSubgroup;
  // Assign more warps to the M dimension (used later) to balance thread
  // counts along X and Y dimensions.
  int64_t warpSqrt =
      1ull << (llvm::divideCeil(llvm::Log2_64(remainingWarps), 2));
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
  int64_t bestKTileCountPerSubgroup =
      seeds.bestKElementCountPerSubgroup
          ? llvm::divideCeil(seeds.bestKElementCountPerSubgroup,
                             intrinsic.kSize)
          : seeds.bestKTileCountPerSubgroup;
  APInt kGCD = GreatestCommonDivisor(APInt(64, kTotalTileCount),
                                     APInt(64, bestKTileCountPerSubgroup));
  int64_t kTileCount = kGCD.getSExtValue();

  return GPUMMASchedule{intrinsicIndex,  intrinsic.mSize, intrinsic.nSize,
                        intrinsic.kSize, mWarpCount,      nWarpCount,
                        mTileCount,      nTileCount,      kTileCount};
}

FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUMatmulShapeType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, bool transposedLhs, bool transposedRhs,
    bool canUpcastAcc, bool mustBeAligned) {
  for (auto [index, intrinsic] : llvm::enumerate(intrinsics)) {
    if (failed(canTargetIntrinsic(problem, intrinsic, canUpcastAcc,
                                  mustBeAligned))) {
      continue;
    }

    GPUMMASchedule schedule =
        getOptimalMMASchedule(problem, intrinsic, seeds, index);

    LLVM_DEBUG({
      llvm::dbgs() << "chosen MMA schedule:\n";
      llvm::dbgs() << "  " << schedule << "\n";
    });

    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      int64_t lhsBitwidth =
          intrinsics[schedule.index].aType.getIntOrFloatBitWidth();
      int64_t rhsBitwidth =
          intrinsics[schedule.index].bType.getIntOrFloatBitWidth();
      bool isAligned =
          isValidMMASchedule(problem, schedule, mustBeAligned, subgroupSize,
                             transposedLhs, transposedRhs);
      int64_t sharedMemoryUsed =
          calculateSharedMemoryUsedInBytes(schedule, lhsBitwidth, rhsBitwidth);

      LLVM_DEBUG({
        llvm::dbgs() << "Available Shared Memory: ";
        llvm::dbgs() << sharedMemLimitInBytes << " bytes\n";
        llvm::dbgs() << "Predicted Shared Memory Used by Schedule: ";
        llvm::dbgs() << sharedMemoryUsed << " bytes\n";
      });

      return isAligned && sharedMemoryUsed <= sharedMemLimitInBytes;
    };

    return fitScheduleInSharedMemory(intrinsic, schedule, isValidSchedule);
  }
  return failure();
}

FailureOr<GPUMMASchedule> deduceAttentionSchedule(
    const GPUMatmulShapeType &qkMatmul, const GPUMatmulShapeType &pvMatmul,
    ArrayRef<GPUMatmulShapeType> intrinsics,
    const GPUMMAHeuristicSeeds &pvMatmulSeeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, bool transposedQ, bool transposedK, bool transposedV,
    bool canUpcastAcc, bool mustBeAligned) {

  for (auto [index, intrinsic] : llvm::enumerate(intrinsics)) {
    if (failed(canTargetIntrinsic(qkMatmul, intrinsic, canUpcastAcc,
                                  mustBeAligned))) {
      continue;
    }

    if (failed(canTargetIntrinsic(pvMatmul, intrinsic, canUpcastAcc,
                                  mustBeAligned))) {
      continue;
    }

    GPUMMASchedule schedule =
        getOptimalMMASchedule(pvMatmul, intrinsic, pvMatmulSeeds, index);

    LLVM_DEBUG({
      llvm::dbgs() << "chosen MMA schedule:\n";
      llvm::dbgs() << "  " << schedule << "\n";
    });

    int64_t intrinsicK = intrinsic.kSize;
    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      // Create a mma schedule for qkMatmul in attention.
      // qkMatmul.M = pvMatmul.M
      // qkMatmul.N = pvMatmul.K
      // qkMatmul.K = problem.K
      GPUMMASchedule qkSchedule{schedule.index,
                                schedule.mSize,
                                schedule.kSize,
                                intrinsicK,
                                /*mWarpCount=*/schedule.mWarpCount,
                                /*nWarpCount=*/1,
                                schedule.mTileCount,
                                schedule.kTileCount,
                                qkMatmul.kSize / intrinsicK};

      bool isQKAligned =
          isValidMMASchedule(qkMatmul, qkSchedule, mustBeAligned, subgroupSize,
                             transposedQ, transposedK);
      // P doesn't need to be distributable by wgSize, since it's not loaded
      // from global memory, but for now, it is okay to ensure it is, as in
      // practice, it should be distributable by wgSize.
      bool isPVAligned = isValidMMASchedule(pvMatmul, schedule, mustBeAligned,
                                            subgroupSize, false, transposedV);

      int64_t lhsBitwidth =
          intrinsics[schedule.index].aType.getIntOrFloatBitWidth();
      int64_t rhsBitwidth =
          intrinsics[schedule.index].bType.getIntOrFloatBitWidth();
      int64_t sharedMemoryUsed =
          calculateSharedMemoryUsedInBytes(qkSchedule, lhsBitwidth,
                                           rhsBitwidth) +
          calculateSharedMemoryUsedInBytes(schedule, lhsBitwidth, rhsBitwidth);

      LLVM_DEBUG({
        llvm::dbgs() << "Available Shared Memory: ";
        llvm::dbgs() << sharedMemLimitInBytes << " bytes\n";
        llvm::dbgs() << "Predicted Shared Memory Used by Schedule: ";
        llvm::dbgs() << sharedMemoryUsed << " bytes\n";
      });

      return isQKAligned && isPVAligned &&
             sharedMemoryUsed <= sharedMemLimitInBytes;
    };

    return fitScheduleInSharedMemory(intrinsic, schedule, isValidSchedule);
  }

  return failure();
}

} // namespace mlir::iree_compiler
