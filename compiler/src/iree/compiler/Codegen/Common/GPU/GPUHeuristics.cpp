// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"

#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-gpu-heuristics"

using llvm::APIntOps::GreatestCommonDivisor;

namespace mlir::iree_compiler {

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  os << "[";
  llvm::interleaveComma(vector, os);
  os << "]";
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUMMASchedule &schedule) {
  os << "mSizes: " << schedule.mSize << ", ";
  os << "nSizes: " << schedule.nSize << ", ";
  os << "kSizes: " << schedule.kSize << ", ";
  os << "mTileSizes: " << schedule.mTileSizes << ", ";
  os << "nTileSizes: " << schedule.nTileSizes << ", ";
  os << "kTileSizes: " << schedule.kTileSizes << ", ";
  os << "mSubgroupCounts: " << schedule.mSubgroupCounts << ", ";
  os << "nSubgroupCounts: " << schedule.nSubgroupCounts;
  return os;
}

// Shortened helper to compute the product of `values`.
static int64_t prod(ArrayRef<int64_t> values) {
  return ShapedType::getNumElements(values);
}

static int64_t calculateSharedMemoryUsedInBytes(const GPUMMASchedule &schedule,
                                                int64_t lhsBitwidth,
                                                int64_t rhsBitwidth) {

  int64_t tileM = schedule.mSize * prod(schedule.mTileSizes) *
                  prod(schedule.mSubgroupCounts);
  int64_t tileN = schedule.nSize * prod(schedule.nTileSizes) *
                  prod(schedule.nSubgroupCounts);
  int64_t tileK = schedule.kSize * prod(schedule.kTileSizes);
  return (tileM * tileK * lhsBitwidth + tileN * tileK * rhsBitwidth) / 8;
}

/// Check that a GPUMMASchedule fits alignment restrictions. To be aligned,
/// the problem must be evenly divisible by the number of elements in the
/// schedule for each dimension. If `mustBeAligned` is false, then the innermost
/// problem dimension is allowed to be unaligned .
static bool isScheduleAligned(const GPUMatmulShapeType &problem,
                              const GPUMMASchedule &schedule,
                              bool mustBeAligned) {
  SmallVector<int64_t> alignedMSizes(problem.mSizes);
  alignedMSizes.back() =
      mustBeAligned ? problem.mSizes.back()
                    : llvm::divideCeil(problem.mSizes.back(), schedule.mSize) *
                          schedule.mSize;
  SmallVector<int64_t> alignedNSizes(problem.nSizes);
  alignedNSizes.back() =
      mustBeAligned ? problem.nSizes.back()
                    : llvm::divideCeil(problem.nSizes.back(), schedule.nSize) *
                          schedule.nSize;
  SmallVector<int64_t> alignedKSizes(problem.kSizes);
  alignedKSizes.back() =
      mustBeAligned ? problem.kSizes.back()
                    : llvm::divideCeil(problem.kSizes.back(), schedule.kSize) *
                          schedule.kSize;
  // Returns the number of elements in the schedule for each dimension.
  auto getScheduleSizes =
      [&](int64_t size, SmallVector<int64_t> tileCount,
          std::optional<SmallVector<int64_t>> subgroupCount) {
        SmallVector<int64_t> sizes = llvm::map_to_vector(
            llvm::seq<int64_t>(tileCount.size()), [&](int64_t i) {
              return subgroupCount ? tileCount[i] * subgroupCount.value()[i]
                                   : tileCount[i];
            });
        sizes.back() *= size;
        return sizes;
      };
  // Checks whether the elements of `a` are evenly divisible by the
  // corresponding elements of `b`.
  auto areAligned = [](SmallVector<int64_t> a, SmallVector<int64_t> b) {
    for (auto [aVal, bVal] : llvm::zip_equal(a, b)) {
      if (aVal % bVal != 0) {
        return false;
      }
    }
    return true;
  };
  bool isValidM = areAligned(
      alignedMSizes, getScheduleSizes(schedule.mSize, schedule.mTileSizes,
                                      schedule.mSubgroupCounts));
  bool isValidN = areAligned(
      alignedNSizes, getScheduleSizes(schedule.nSize, schedule.nTileSizes,
                                      schedule.nSubgroupCounts));
  bool isValidK = areAligned(
      alignedKSizes,
      getScheduleSizes(schedule.kSize, schedule.kTileSizes, std::nullopt));
  return isValidM && isValidN && isValidK;
}

/// Returns whether or not a GPUMMASchedule is valid for the given problem.
/// This checks that:
///  - The problem is aligned to the schedule
///  - the number of threads in the schedule workgroup can be distributed
///    to a corresponding vector.transfer read in VectorDistribute.
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
  int64_t wgThreads = subgroupSize * prod(schedule.mSubgroupCounts) *
                      prod(schedule.nSubgroupCounts);
  int64_t mWgSize = schedule.mSize * prod(schedule.mTileSizes) *
                    prod(schedule.mSubgroupCounts);
  int64_t nWgSize = schedule.nSize * prod(schedule.nTileSizes) *
                    prod(schedule.nSubgroupCounts);
  int64_t kWgSize = schedule.kSize * prod(schedule.kTileSizes);
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

/// Tries to fit the schedule into shared memory by decrementing the size of the
/// schedule dimensions from outermost to innermost until a valid schedule is
/// found. The schedule sizes are reduced in the order of mTileSizes,
/// nTileSizes, kTileSizes, mSubgroupCounts, nSubgroupCounts.
static FailureOr<GPUMMASchedule> fitScheduleInSharedMemory(
    GPUMatmulShapeType intrinsic, GPUMMASchedule schedule,
    llvm::function_ref<bool(const GPUMMASchedule &schedule)> isScheduleValid) {

  while (!isScheduleValid(schedule)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Chosen schedule is invalid:\n";
      llvm::dbgs() << schedule << "\n";
      llvm::dbgs() << "Shrinking schedule...\n";
    });

    auto decrementIfPossible =
        [](SmallVector<int64_t> &sizes) -> LogicalResult {
      for (int64_t &size : sizes) {
        if (size <= 1)
          continue;
        --size;
        return success();
      }
      return failure();
    };

    // Attempt to shrink the schedule along one of the dimensions.
    // TODO: A better solution should probably factor problem.mSize /
    // (mSubgroupCount * mTileCount * mSize) and then pop off the smallest
    // factors one at a time, preferably trying to keep the tile "generally
    // square."
    if (succeeded(decrementIfPossible(schedule.mTileSizes))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.nTileSizes))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.kTileSizes))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.mSubgroupCounts))) {
      continue;
    }
    if (succeeded(decrementIfPossible(schedule.nSubgroupCounts))) {
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
  assert(intrinsic.mSizes.size() == 1 && intrinsic.nSizes.size() == 1 &&
         intrinsic.kSizes.size() == 1 &&
         "expected intrinsic to have a single M, N, and K dimension.");
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

  if (mustBeAligned && (problem.mSizes.back() % intrinsic.mSizes[0] != 0 ||
                        problem.nSizes.back() % intrinsic.nSizes[0] != 0 ||
                        problem.kSizes.back() % intrinsic.kSizes[0] != 0)) {
    return failure(); // Cannot use this intrinsic for misaligned cases.
  }

  // Cannot use the intrinsic when the tile size is greater than problem size.
  // Because tiling is a no-op, and we can't infer tiling sizes from IR.
  if (!mustBeAligned && (problem.mSizes.back() < intrinsic.mSizes[0] ||
                         problem.nSizes.back() < intrinsic.nSizes[0] ||
                         problem.kSizes.back() < intrinsic.kSizes[0])) {
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
  assert(intrinsic.mSizes.size() == 1 && intrinsic.nSizes.size() == 1 &&
         intrinsic.kSizes.size() == 1 &&
         "expected intrinsic to have a single M, N, and K dimension.");
  // mTotalTileCounts and nTotalTileCounts represent the total number of
  // intrinsics along the M or N dimensions needed to fill the problem size.
  // For example, if the problem is {M:[4, 16], N:[2, 32], K[3, 128]} for a
  // 16x16x16 intrinsic, then:
  //  - mTotalTileCounts would be 4 * (16/16) = 4
  //  - nTotalTileCounts would be 2 * (32/16) = 4
  SmallVector<int64_t> mTotalTileCounts = problem.mSizes;
  SmallVector<int64_t> nTotalTileCounts = problem.nSizes;
  mTotalTileCounts.back() =
      llvm::divideCeil(problem.mSizes.back(), intrinsic.mSizes[0]);
  nTotalTileCounts.back() =
      llvm::divideCeil(problem.nSizes.back(), intrinsic.nSizes[0]);

  int64_t remainingSubgroups = seeds.bestSubgroupCountPerWorkgroup;
  int64_t remainingTiles = seeds.bestMNTileCountPerSubgroup;
  // Assign more subgroups to the M dimension (used later) to balance thread
  // counts along X and Y dimensions.
  int mDim = problem.mSizes.size() - 1;
  int nDim = problem.nSizes.size() - 1;
  SmallVector<int64_t> mTileSizes(problem.mSizes.size(), 0),
      nTileSizes(problem.nSizes.size(), 0),
      mSubgroupCounts(problem.mSizes.size(), 0),
      nSubgroupCounts(problem.nSizes.size(), 0);
  // Start at the innermost nDim and mDim, and try to distribute evenly to M and
  // N for each pair of M and N dims. Otherwise, distribute to N and then M.
  while (mDim >= 0 || nDim >= 0) {
    int64_t subgroupSqrt =
        1ull << (llvm::divideCeil(llvm::Log2_64(remainingSubgroups), 2));
    int64_t tileSqrt = 1ull << (llvm::Log2_64(remainingTiles) / 2);

    // See if the square root can divide mTotalTileCount. If so it means we can
    // distribute to both dimensions evenly to minimize the number of global
    // loads. Otherwise, try to distribute to N and then M.
    if (mDim >= 0 && nDim >= 0 &&
        mTotalTileCounts[mDim] > (subgroupSqrt * tileSqrt) &&
        mTotalTileCounts[mDim] % (subgroupSqrt * tileSqrt) == 0) {
      mSubgroupCounts[mDim] = subgroupSqrt;
      mTileSizes[mDim] = tileSqrt;

      remainingSubgroups /= subgroupSqrt;
      remainingTiles /= tileSqrt;

      APInt nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCounts[nDim]),
                                         APInt(64, remainingSubgroups));
      nSubgroupCounts[nDim] = nGCD.getSExtValue();
      nTotalTileCounts[nDim] /= nSubgroupCounts[nDim];
      remainingSubgroups /= nSubgroupCounts[nDim];

      nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCounts[nDim]),
                                   APInt(64, remainingTiles));
      nTileSizes[nDim] = nGCD.getSExtValue();
      remainingTiles /= nTileSizes[nDim];
    } else {
      if (nDim >= 0) {
        APInt nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCounts[nDim]),
                                           APInt(64, remainingSubgroups));
        nSubgroupCounts[nDim] = nGCD.getSExtValue();
        nTotalTileCounts[nDim] /= nSubgroupCounts[nDim];
        remainingSubgroups /= nSubgroupCounts[nDim];

        nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCounts[nDim]),
                                     APInt(64, remainingTiles));
        nTileSizes[nDim] = nGCD.getSExtValue();
        remainingTiles /= nTileSizes[nDim];
      }

      if (mDim >= 0) {
        APInt mGCD = GreatestCommonDivisor(APInt(64, mTotalTileCounts[mDim]),
                                           APInt(64, remainingSubgroups));
        mSubgroupCounts[mDim] = mGCD.getSExtValue();
        mTotalTileCounts[mDim] /= mSubgroupCounts[mDim];
        remainingSubgroups /= mSubgroupCounts[mDim];

        mGCD = GreatestCommonDivisor(APInt(64, mTotalTileCounts[mDim]),
                                     APInt(64, remainingTiles));
        mTileSizes[mDim] = mGCD.getSExtValue();
        remainingTiles /= mTileSizes[mDim];
      }
    }
    --mDim;
    --nDim;
  }

  // kTotalTileCounts is similar to m/nTotalTileCounts, representing the total
  // number of intrinsics along the K dimensions needed to fill the problem.
  // For the problem described above {M:[4, 16], N:[2, 32], K[3, 128]} with a
  // 16x16x16 intrinsic, then:
  //  - kTotalTileCounts would be 3 * (128/16) = 24
  SmallVector<int64_t> kTotalTileCounts = problem.kSizes;
  kTotalTileCounts.back() =
      llvm::divideCeil(problem.kSizes.back(), intrinsic.kSizes[0]);
  // Compute the ideal number of intrinsics along K per subgroup based on the
  // seed.
  int64_t bestKTileCountPerSubgroup =
      seeds.bestKElementCountPerSubgroup
          ? llvm::divideCeil(seeds.bestKElementCountPerSubgroup,
                             intrinsic.kSizes[0])
          : seeds.bestKTileCountPerSubgroup;
  SmallVector<int64_t> kTileSizes(problem.kSizes.size(), 0);
  // Start at the innermost K dim, and tile each dim to try to satisfy the ideal
  // K intrinsic count per subgroup with the overall product of K tile counts.
  int kDim = problem.kSizes.size() - 1;
  while (kDim >= 0) {
    APInt kGCD = GreatestCommonDivisor(APInt(64, kTotalTileCounts[kDim]),
                                       APInt(64, bestKTileCountPerSubgroup));
    kTileSizes[kDim] = kGCD.getSExtValue();
    bestKTileCountPerSubgroup /= kTileSizes[kDim];
    --kDim;
  }

  return GPUMMASchedule{
      intrinsicIndex,      intrinsic.mSizes[0], intrinsic.nSizes[0],
      intrinsic.kSizes[0], mSubgroupCounts,     nSubgroupCounts,
      mTileSizes,          nTileSizes,          kTileSizes};
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
  assert(pvMatmul.mSizes.size() == 1 && pvMatmul.nSizes.size() == 1 &&
         pvMatmul.kSizes.size() == 1 && qkMatmul.mSizes.size() == 1 &&
         qkMatmul.nSizes.size() == 1 && qkMatmul.kSizes.size() == 1 &&
         "unimplemented: multi M/N/K attention schedule");
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

    int64_t intrinsicK = intrinsic.kSizes[0];
    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      // Create a mma schedule for qkMatmul in attention.
      // qkMatmul.M = pvMatmul.M
      // qkMatmul.N = pvMatmul.K
      // qkMatmul.K = problem.K
      GPUMMASchedule qkSchedule{schedule.index,
                                schedule.mSize,
                                schedule.kSize,
                                intrinsicK,
                                /*mSubgroupCount=*/schedule.mSubgroupCounts[0],
                                /*nSubgroupCount=*/1,
                                schedule.mTileSizes[0],
                                schedule.kTileSizes[0],
                                qkMatmul.kSizes[0] / intrinsicK};

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
