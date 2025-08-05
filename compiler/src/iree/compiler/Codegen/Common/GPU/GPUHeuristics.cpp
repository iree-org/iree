// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-gpu-heuristics"

using llvm::APIntOps::GreatestCommonDivisor;

namespace mlir::iree_compiler {

using IREE::GPU::getSingleSubgroupLayout;

// Threshold used to determine whether a matmul dimension is 'very skinny'.
constexpr int64_t kVerySkinnyDimThreshold = 4;

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const llvm::SmallVectorImpl<T> &vector) {
  return os << llvm::interleaved_array(vector);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUMMASchedule &schedule) {
  os << "mmaKind " << schedule.mmaKind << ", ";
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

static int64_t calculateOperandsSharedMemoryUsedInBytes(
    const GPUMMASchedule &schedule, int64_t lhsBitwidth, int64_t rhsBitwidth) {

  int64_t tileM = schedule.mSize * prod(schedule.mTileSizes) *
                  prod(schedule.mSubgroupCounts);
  int64_t tileN = schedule.nSize * prod(schedule.nTileSizes) *
                  prod(schedule.nSubgroupCounts);
  int64_t tileK = schedule.kSize * prod(schedule.kTileSizes);
  return (tileM * tileK * lhsBitwidth + tileN * tileK * rhsBitwidth) / 8;
}

static int64_t
calculateResultSharedMemoryUsedInBytes(const GPUMMASchedule &schedule,
                                       int64_t resultBitwidth) {

  int64_t tileM = schedule.mSize * prod(schedule.mTileSizes) *
                  prod(schedule.mSubgroupCounts);
  int64_t tileN = schedule.nSize * prod(schedule.nTileSizes) *
                  prod(schedule.nSubgroupCounts);
  return (tileM * tileN * resultBitwidth) / 8;
}

/// Check that a GPUMMASchedule fits alignment restrictions. To be aligned,
/// the problem must be evenly divisible by the number of elements in the
/// schedule for each dimension. If `mustBeAligned` is false, then the innermost
/// problem dimension is allowed to be unaligned .
static bool isScheduleAligned(const GPUMatmulShapeType &problem,
                              const GPUMMASchedule &schedule,
                              bool mustBeAligned) {
  SmallVector<int64_t, 2> alignedMSizes(problem.mSizes);
  alignedMSizes.back() =
      mustBeAligned ? problem.mSizes.back()
                    : llvm::divideCeil(problem.mSizes.back(), schedule.mSize) *
                          schedule.mSize;
  SmallVector<int64_t, 2> alignedNSizes(problem.nSizes);
  alignedNSizes.back() =
      mustBeAligned ? problem.nSizes.back()
                    : llvm::divideCeil(problem.nSizes.back(), schedule.nSize) *
                          schedule.nSize;
  SmallVector<int64_t, 2> alignedKSizes(problem.kSizes);
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
  auto areAligned = [](SmallVector<int64_t, 2> a, SmallVector<int64_t, 2> b) {
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
    GPUMMASchedule schedule,
    llvm::function_ref<bool(const GPUMMASchedule &schedule)> isScheduleValid) {

  while (!isScheduleValid(schedule)) {
    LDBG() << "Chosen schedule is invalid:\n"
           << schedule << "\nShrinking schedule...";

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

  LDBG() << "Chosen schedule is valid:\n" << schedule;

  return schedule;
}

static LogicalResult canTargetIntrinsic(const GPUMatmulShapeType &problem,
                                        const GPUMatmulShapeType &intrinsic,
                                        int64_t preferredSubgroupSize,
                                        bool canUpcastAcc, bool mustBeAligned) {
  assert(intrinsic.mSizes.size() == 1 && intrinsic.nSizes.size() == 1 &&
         intrinsic.kSizes.size() <= 2 &&
         "expected intrinsic to have a single M, N, and K <= 2 dimensions.");
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

  if (mustBeAligned) {
    if ((problem.mSizes.back() % intrinsic.mSizes[0] != 0 ||
         problem.nSizes.back() % intrinsic.nSizes[0] != 0 ||
         problem.kSizes.back() % intrinsic.kSizes[0] != 0)) {
      return failure();
    }
    return success();
  }

  // Send very skinny, {2-4}xNxK and Mx{2-4}xK, matmuls to the vector reduction
  // pipeline, similar to matvec.
  // TODO: Figure out what the precise cutoff is, this may be machine dependent.
  // In situation when alignment isn't required, we disallow intrinsics to be
  // picked if the tile size is too small. For example, this will force a matmul
  // with a tiny dimension to not use MFMA instructions because of the
  // additional overhead that comes with it. However, 4 is only an approximation
  // to boundary between matvec and matmul. The actual heuristic can be
  // established after we sweep the different tile sizes for a problem config.
  // Once a precise threshold is established, replace 4 with the threshold and
  // remove this todo.
  if (llvm::all_equal({problem.mSizes.size(), problem.nSizes.size(),
                       problem.kSizes.size(), size_t{1}}) &&
      problem.batchSizes.empty()) {
    int64_t mSize = problem.mSizes.back();
    int64_t nSize = problem.nSizes.back();
    if ((mSize <= kVerySkinnyDimThreshold && (nSize > preferredSubgroupSize)) ||
        (nSize <= kVerySkinnyDimThreshold && (mSize > preferredSubgroupSize))) {
      return failure();
    }
  }
  return success();
}

static SmallVector<int64_t>
getBestKTileSizes(const GPUMatmulShapeType &problem,
                  const GPUIntrinsicType &intrinsic,
                  const GPUMMAHeuristicSeeds &seeds) {
  // kTotalTileCounts is similar to m/nTotalTileCounts, representing the total
  // number of intrinsics along the K dimensions needed to fill the problem.
  // For the problem described above {M:[4, 16], N:[2, 32], K[3, 128]} with a
  // 16x16x16 intrinsic, then:
  //  - kTotalTileCounts would be 3 * (128/16) = 24
  SmallVector<int64_t, 2> kTotalTileCounts = problem.kSizes;
  for (auto [kTotalTileCount, intrinsicKSize] : llvm::zip_equal(
           MutableArrayRef{kTotalTileCounts}.take_back(intrinsic.kSizes.size()),
           intrinsic.kSizes)) {
    kTotalTileCount = llvm::divideCeil(kTotalTileCount, intrinsicKSize);
  }

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

  return kTileSizes;
}

/// Choose an optimal mma schedule with the heuristic that minimized the total
/// amount of data read from global memory, per workgroup, respecting the
/// heuristic seeds.
static GPUMMASchedule getOptimalMMASchedule(const GPUMatmulShapeType &problem,
                                            const GPUIntrinsicType &intrinsic,
                                            const GPUMMAHeuristicSeeds &seeds) {
  assert(intrinsic.mSizes.size() == 1 && intrinsic.nSizes.size() == 1 &&
         intrinsic.kSizes.size() <= 2 &&
         "expected intrinsic to have a single M, N, and K <= 2 dimensions.");
  // mTotalTileCounts and nTotalTileCounts represent the total number of
  // intrinsics along the M or N dimensions needed to fill the problem size.
  // For example, if the problem is {M:[4, 16], N:[2, 32], K[3, 128]} for a
  // 16x16x16 intrinsic, then:
  //  - mTotalTileCounts would be 4 * (16/16) = 4
  //  - nTotalTileCounts would be 2 * (32/16) = 4
  SmallVector<int64_t, 2> mTotalTileCounts = problem.mSizes;
  SmallVector<int64_t, 2> nTotalTileCounts = problem.nSizes;
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
  LDBG() << "Starting MMA schedule distribution";
  while (mDim >= 0 || nDim >= 0) {
    LDBG() << "Current iteration: mDim: " << mDim << ", nDim: " << nDim
           << ", remainingSubgroups: " << remainingSubgroups
           << ", remainingTiles: " << remainingTiles
           << ", mTileSizes: " << mTileSizes << ", nTileSizes: " << nTileSizes;
    int64_t subgroupSqrt =
        1ull << (llvm::divideCeil(llvm::Log2_64(remainingSubgroups), 2));
    int64_t tileSqrt = 1ull << (llvm::Log2_64(remainingTiles) / 2);

    // See if the square root can divide mTotalTileCount. If so it means we can
    // distribute to both dimensions evenly to minimize the number of global
    // loads. Otherwise, try to distribute to N and then M.
    if (mDim >= 0 && nDim >= 0 &&
        mTotalTileCounts[mDim] > (subgroupSqrt * tileSqrt) &&
        mTotalTileCounts[mDim] % (subgroupSqrt * tileSqrt) == 0) {
      LDBG() << "Distributing evenly to M and N dimensions.";
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
        LDBG() << "Distributing to N dimension first.";
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
        LDBG() << "Distributing to M dimension next.";
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

  SmallVector<int64_t> kTileSizes =
      getBestKTileSizes(problem, intrinsic, seeds);

  return GPUMMASchedule{
      intrinsic.mmaKind,   intrinsic.mSizes[0], intrinsic.nSizes[0],
      intrinsic.kSizes[0], mSubgroupCounts,     nSubgroupCounts,
      mTileSizes,          nTileSizes,          kTileSizes};
}

/// Compare the MMA intrinsics by following precedence rules:
///   1) k-alignment. We prefer intrinsics that can evenly divide the K
///   dimension of the problem.
///   2) M/N-alignment. We prefer intrinsics that can evenly divide the M
///   and N dimensions of the problem.
///   3) Intrinsic with larger gemm size.
///   4) Intrinsic with larger K size.
///
/// This function acts as a comparison function object for std::sort, which
/// returns true if the lhs is ordered before rhs.
bool compareIntrinsics(const GPUMatmulShapeType &problem,
                       const GPUMatmulShapeType &lhs,
                       const GPUMatmulShapeType &rhs) {
  // Prefer K-aligned intrinsics.
  int lhsKAligned = problem.kSizes.back() % lhs.kSizes.back() == 0 ? 1 : 0;
  int rhsKAligned = problem.kSizes.back() % rhs.kSizes.back() == 0 ? 1 : 0;
  if (lhsKAligned != rhsKAligned) {
    return lhsKAligned > rhsKAligned;
  }

  // If K alignment is the same, prefer the intrinsic that aligns M and N.
  int lhsMNAligned = (problem.mSizes.back() % lhs.mSizes.back() == 0 &&
                      problem.nSizes.back() % lhs.nSizes.back() == 0)
                         ? 1
                         : 0;
  int rhsMNAligned = (problem.mSizes.back() % rhs.mSizes.back() == 0 &&
                      problem.nSizes.back() % rhs.nSizes.back() == 0)
                         ? 1
                         : 0;
  if (lhsMNAligned != rhsMNAligned) {
    return lhsMNAligned > rhsMNAligned;
  }

  auto intrinsicArea = [&](const GPUMatmulShapeType &intrinsic) {
    return (ShapedType::getNumElements(intrinsic.mSizes) +
            ShapedType::getNumElements(intrinsic.nSizes)) *
           ShapedType::getNumElements(intrinsic.kSizes);
  };
  int64_t lhsArea = intrinsicArea(lhs);
  int64_t rhsArea = intrinsicArea(rhs);
  if (lhsArea != rhsArea) {
    return lhsArea > rhsArea;
  }

  // Finally if everything else is the same, prefer large K size.
  return ShapedType::getNumElements(lhs.kSizes) >
         ShapedType::getNumElements(rhs.kSizes);
}

static SmallVector<GPUIntrinsicType>
sortMMAIntrinsics(GPUMatmulShapeType problem,
                  ArrayRef<GPUIntrinsicType> intrinsics) {
  SmallVector<GPUIntrinsicType> sortedIntrinsics;
  llvm::sort(sortedIntrinsics,
             [&](const GPUMatmulShapeType &lhs, const GPUMatmulShapeType &rhs) {
               return compareIntrinsics(problem, lhs, rhs);
             });
  return sortedIntrinsics;
}

static int64_t adjustSeedsForWgpCount(const GPUMatmulShapeType &problem,
                                      const GPUIntrinsicType &intrinsic,
                                      std::optional<int64_t> wgpCount,
                                      int64_t bestSubgroupCountPerWorkgroup,
                                      int64_t bestMNTileCountPerSubgroup) {
  if (!wgpCount.has_value()) {
    LDBG() << "WGP count is not available,"
           << "Skipping adjustment of seeds for workgroup count.";
    return bestMNTileCountPerSubgroup;
  }

  int64_t mSize = ShapedType::getNumElements(problem.mSizes);
  int64_t nSize = ShapedType::getNumElements(problem.nSizes);
  int64_t kSize = ShapedType::getNumElements(problem.kSizes);
  float arithmeticIntensity =
      (2.0f * mSize * nSize * kSize) /
      static_cast<float>(mSize * nSize + nSize * kSize + mSize * kSize);

  // TODO(jerryyin): compute arithmetic intensity bound based on the information
  // from the target chip.
  if (arithmeticIntensity <= 10.0f) {
    LDBG() << "Arithmetic intensity is too low, " << arithmeticIntensity
           << ", skipping adjustment of seeds for workgroup count.";
    return bestMNTileCountPerSubgroup;
  }
  auto computeWorkgroupCount = [&] {
    // Compute the number of workgroups needed to cover the problem size.
    // This number tends to be lower than actual workgroup count, since:
    // 1) It assumes tile and subgroup seeds are all allocated.
    // 2) It assumes shared memory usage does not exceed hardware limits.
    int64_t mnTileSizePerSubgroup =
        bestMNTileCountPerSubgroup * intrinsic.mSizes[0] * intrinsic.nSizes[0];
    int64_t workgroupSize =
        mnTileSizePerSubgroup * bestSubgroupCountPerWorkgroup;
    return mSize * nSize / workgroupSize;
  };
  int64_t numWorkgroups = computeWorkgroupCount();
  LDBG() << "Estimated number of workgroups: " << numWorkgroups
         << ", WGP count: " << wgpCount;

  while (numWorkgroups < wgpCount) {
    if (bestMNTileCountPerSubgroup <= 1) {
      LDBG() << "Cannot decrease tile size further, "
                "bestMNTileCountPerSubgroup is already 1.";
      break;
    }
    bestMNTileCountPerSubgroup /= 2;
    LDBG() << "Decreasing bestMNTileCountPerSubgroup to "
           << bestMNTileCountPerSubgroup;
    numWorkgroups = computeWorkgroupCount();
  }
  return bestMNTileCountPerSubgroup;
}

FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, std::optional<int64_t> wgpCount, bool transposedLhs,
    bool transposedRhs, bool canUpcastAcc, bool mustBeAligned,
    bool doCPromotion) {

  sortMMAIntrinsics(problem, intrinsics);

  for (const GPUIntrinsicType &intrinsic : intrinsics) {
    if (failed(canTargetIntrinsic(problem, intrinsic, subgroupSize,
                                  canUpcastAcc, mustBeAligned))) {
      continue;
    }

    // Note: don't amend the original seeds, as deduceMMASchedule can be called
    // more than once in a row, and we want to keep the original seeds intact
    // for the next call.
    GPUMMAHeuristicSeeds localSeeds = seeds;
    localSeeds.bestMNTileCountPerSubgroup = adjustSeedsForWgpCount(
        problem, intrinsic, wgpCount, seeds.bestSubgroupCountPerWorkgroup,
        seeds.bestMNTileCountPerSubgroup);
    GPUMMASchedule schedule =
        getOptimalMMASchedule(problem, intrinsic, localSeeds);

    LDBG() << "Chosen MMA schedule:\n" << schedule;

    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      int64_t lhsBitwidth = intrinsic.aType.getIntOrFloatBitWidth();
      int64_t rhsBitwidth = intrinsic.bType.getIntOrFloatBitWidth();
      int64_t resultBitwidth = intrinsic.cType.getIntOrFloatBitWidth();
      bool isAligned =
          isValidMMASchedule(problem, schedule, mustBeAligned, subgroupSize,
                             transposedLhs, transposedRhs);
      int64_t sharedMemoryUsed = calculateOperandsSharedMemoryUsedInBytes(
          schedule, lhsBitwidth, rhsBitwidth);
      if (doCPromotion) {
        sharedMemoryUsed +=
            calculateResultSharedMemoryUsedInBytes(schedule, resultBitwidth);
      }

      LDBG() << "Available Shared Memory: " << sharedMemLimitInBytes << " bytes"
             << "Predicted Shared Memory Used by Schedule: " << sharedMemoryUsed
             << " bytes";
      return isAligned && sharedMemoryUsed <= sharedMemLimitInBytes;
    };
    return fitScheduleInSharedMemory(schedule, isValidSchedule);
  }
  return failure();
}

/// Choose an optimal attention PV schedule with the heuristic that minimized
/// the total amount of data read from global memory, per workgroup, respecting
/// the heuristic seeds.
static GPUMMASchedule
getOptimalAttentionPVSchedule(const GPUMatmulShapeType &problem,
                              const GPUIntrinsicType &intrinsic,
                              const GPUMMAHeuristicSeeds &seeds) {
  assert(intrinsic.mSizes.size() == 1 && intrinsic.nSizes.size() == 1 &&
         intrinsic.kSizes.size() == 1 &&
         "expected intrinsic to have a single M, N, and K dimension.");
  // mTotalTileCounts and nTotalTileCounts represent the total number of
  // intrinsics along the M or N dimensions needed to fill the problem size.
  // For example, if the problem is {M:[4, 16], N:[2, 32], K[3, 128]} for a
  // 16x16x16 intrinsic, then:
  //  - mTotalTileCounts would be 4 * (16/16) = 4
  //  - nTotalTileCounts would be 2 * (32/16) = 4
  SmallVector<int64_t, 2> mTotalTileCounts = problem.mSizes;
  SmallVector<int64_t, 2> nTotalTileCounts = problem.nSizes;
  mTotalTileCounts.back() =
      llvm::divideCeil(problem.mSizes.back(), intrinsic.mSizes[0]);
  nTotalTileCounts.back() =
      llvm::divideCeil(problem.nSizes.back(), intrinsic.nSizes[0]);

  int64_t remainingSubgroups = seeds.bestSubgroupCountPerWorkgroup;
  int64_t remainingTiles = seeds.bestMNTileCountPerSubgroup;
  SmallVector<int64_t> mTileSizes(problem.mSizes.size(), 0),
      nTileSizes(problem.nSizes.size(), 0),
      mSubgroupCounts(problem.mSizes.size(), 0),
      nSubgroupCounts(problem.nSizes.size(), 0);

  // For Attention, we use a simple heuristic based on other Flash Attention
  // implementations, there are better heuristics to use, but we use something
  // that consistently works, is simple, and is used every other implementation.
  //
  // For Attention, we can assume that the N dimension is constant and is
  // completely unrolled. This means that we distribute all available tiles to
  // N first, and then the remaining tiles to M.
  //
  // We do not distribute subgroups on N. This is because distributing
  // subgroups on N leaves room to distribute subgroups on K1 and how that
  // effects the softmax computation hasn't been experimented with yet.
  //
  // Distribute tile sizes on N as much as we can as it's completly unrolled and
  // then distribute remaining tiles and subgroups on M.
  for (int nDim = problem.nSizes.size() - 1; nDim >= 0; --nDim) {
    // Do not distribute N on subgroups.
    nSubgroupCounts[nDim] = 1;

    APInt nGCD = GreatestCommonDivisor(APInt(64, nTotalTileCounts[nDim]),
                                       APInt(64, remainingTiles));
    nTileSizes[nDim] = nGCD.getSExtValue();
    remainingTiles /= nTileSizes[nDim];
  }
  for (int mDim = problem.mSizes.size() - 1; mDim >= 0; --mDim) {
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

  SmallVector<int64_t> kTileSizes =
      getBestKTileSizes(problem, intrinsic, seeds);

  return GPUMMASchedule{
      intrinsic.mmaKind,   intrinsic.mSizes[0], intrinsic.nSizes[0],
      intrinsic.kSizes[0], mSubgroupCounts,     nSubgroupCounts,
      mTileSizes,          nTileSizes,          kTileSizes};
}

struct ChainedMMAIntrinsics {
  GPUIntrinsicType intrinsicA;
  GPUIntrinsicType intrinsicB;
  bool canReuseAOutputForB;
};

FailureOr<std::pair<GPUMMASchedule, GPUMMASchedule>> deduceAttentionSchedule(
    const GPUMatmulShapeType &qkMatmul, const GPUMatmulShapeType &pvMatmul,
    ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &pvMatmulSeeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, bool transposedQ, bool transposedK, bool transposedV,
    bool canUpcastAcc, bool mustBeAligned) {
  assert(pvMatmul.mSizes.size() == 1 && pvMatmul.nSizes.size() == 1 &&
         pvMatmul.kSizes.size() == 1 && qkMatmul.mSizes.size() == 1 &&
         qkMatmul.nSizes.size() == 1 && qkMatmul.kSizes.size() == 1 &&
         "unimplemented: multi M/N/K attention schedule");

  std::vector<ChainedMMAIntrinsics> intrinsicPairs;

  for (const GPUIntrinsicType &intrinsicA : intrinsics) {
    for (const GPUIntrinsicType &intrinsicB : intrinsics) {
      if (failed(canTargetIntrinsic(qkMatmul, intrinsicA, subgroupSize,
                                    canUpcastAcc, mustBeAligned))) {
        continue;
      }

      if (failed(canTargetIntrinsic(pvMatmul, intrinsicB, subgroupSize,
                                    canUpcastAcc, mustBeAligned))) {
        continue;
      }
      // Check if we can reuse the output of intrinsicA for lhs/rhs of
      // intrinsicB.
      auto matchLayout =
          [](IREE::GPU::MMASingleSubgroupLayout layoutA,
             IREE::GPU::MMASingleSubgroupLayout layoutB) -> bool {
        return (layoutA.element == layoutB.element) &&
               (layoutA.thread == layoutB.thread) &&
               (layoutA.tstrides == layoutB.tstrides);
      };
      bool canReuseAOutForBLhs =
          matchLayout(getSingleSubgroupLayout(intrinsicA.mmaKind,
                                              IREE::GPU::MMAFragment::Acc),
                      getSingleSubgroupLayout(intrinsicB.mmaKind,
                                              IREE::GPU::MMAFragment::Lhs));
      bool canReuseAOutForBRhs =
          matchLayout(getSingleSubgroupLayout(intrinsicA.mmaKind,
                                              IREE::GPU::MMAFragment::Acc),
                      getSingleSubgroupLayout(intrinsicB.mmaKind,
                                              IREE::GPU::MMAFragment::Rhs));
      intrinsicPairs.push_back(
          {intrinsicA, intrinsicB, canReuseAOutForBLhs || canReuseAOutForBRhs});
    }
  }

  llvm::sort(intrinsicPairs, [&](const ChainedMMAIntrinsics &lhs,
                                 const ChainedMMAIntrinsics &rhs) {
    if (lhs.canReuseAOutputForB && !rhs.canReuseAOutputForB) {
      return true;
    }
    if (!lhs.canReuseAOutputForB && rhs.canReuseAOutputForB) {
      return false;
    }

    if (lhs.intrinsicA.mmaKind != rhs.intrinsicA.mmaKind) {
      return compareIntrinsics(qkMatmul, lhs.intrinsicA, rhs.intrinsicA);
    }
    return compareIntrinsics(pvMatmul, lhs.intrinsicB, rhs.intrinsicB);
  });

  for (ChainedMMAIntrinsics intrinsics : intrinsicPairs) {
    // Structured bindings cannot be captured in C++ < 20.
    GPUIntrinsicType intrinsicA = intrinsics.intrinsicA;
    GPUIntrinsicType intrinsicB = intrinsics.intrinsicB;
    bool canReuseAOutput = intrinsics.canReuseAOutputForB;

    GPUMMASchedule schedule =
        getOptimalAttentionPVSchedule(pvMatmul, intrinsicB, pvMatmulSeeds);

    LDBG() << "Chosen MMA schedule:\n" << schedule;
    int64_t intrinsicAK = intrinsicA.kSizes[0];
    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      // Create a mma schedule for qkMatmul in attention.
      // qkMatmul.M = pvMatmul.M
      // qkMatmul.N = pvMatmul.K
      // qkMatmul.K = problem.K
      GPUMMASchedule qkSchedule{intrinsicA.mmaKind,
                                schedule.mSize,
                                schedule.kSize,
                                intrinsicAK,
                                /*mSubgroupCount=*/schedule.mSubgroupCounts[0],
                                /*nSubgroupCount=*/1,
                                schedule.mTileSizes[0],
                                schedule.kTileSizes[0],
                                qkMatmul.kSizes[0] / intrinsicAK};

      bool isQKAligned =
          isValidMMASchedule(qkMatmul, qkSchedule, mustBeAligned, subgroupSize,
                             transposedQ, transposedK);
      // P doesn't need to be distributable by wgSize, since it's not loaded
      // from global memory, but for now, it is okay to ensure it is, as in
      // practice, it should be distributable by wgSize.
      bool isPVAligned = isValidMMASchedule(pvMatmul, schedule, mustBeAligned,
                                            subgroupSize, false, transposedV);

      int64_t lhsABitwidth = intrinsicA.aType.getIntOrFloatBitWidth();
      int64_t rhsABitwidth = intrinsicA.bType.getIntOrFloatBitWidth();
      int64_t rhsBBitwidth = intrinsicB.bType.getIntOrFloatBitWidth();
      // We don't need to use shared memory for lhsB if we can reuse A intrinsic
      // output.
      int64_t lhsBBitwidth =
          canReuseAOutput ? 0 : intrinsicB.aType.getIntOrFloatBitWidth();

      int64_t sharedMemoryUsed = calculateOperandsSharedMemoryUsedInBytes(
                                     qkSchedule, lhsABitwidth, rhsABitwidth) +
                                 calculateOperandsSharedMemoryUsedInBytes(
                                     schedule, lhsBBitwidth, rhsBBitwidth);

      LDBG() << "Available Shared Memory: " << sharedMemLimitInBytes << " bytes"
             << "Predicted Shared Memory Used by Schedule: " << sharedMemoryUsed
             << " bytes";

      return isQKAligned && isPVAligned &&
             sharedMemoryUsed <= sharedMemLimitInBytes;
    };

    FailureOr<GPUMMASchedule> pvSchedule =
        fitScheduleInSharedMemory(schedule, isValidSchedule);
    if (failed(pvSchedule)) {
      return failure();
    }

    // Create a mma schedule for qkMatmul in attention.
    // qkMatmul.M = pvMatmul.M
    // qkMatmul.N = pvMatmul.K
    // qkMatmul.K = problem.K
    GPUMMASchedule qkSchedule{intrinsicA.mmaKind,
                              pvSchedule->mSize,
                              pvSchedule->kSize,
                              intrinsicAK,
                              /*mSubgroupCount=*/schedule.mSubgroupCounts[0],
                              /*nSubgroupCount=*/1,
                              pvSchedule->mTileSizes[0],
                              pvSchedule->kTileSizes[0],
                              qkMatmul.kSizes[0] / intrinsicAK};

    return std::pair(qkSchedule, pvSchedule.value());
  }

  return failure();
}

} // namespace mlir::iree_compiler
