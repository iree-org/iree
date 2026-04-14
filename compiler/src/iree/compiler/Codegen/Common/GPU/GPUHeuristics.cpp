// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

#include <cstdint>

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Remarks.h"

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
  os << "mSizes: " << schedule.mSizes << ", ";
  os << "nSizes: " << schedule.nSizes << ", ";
  os << "kSizes: " << schedule.kSizes << ", ";
  os << "mTileSizes: " << schedule.mTileSizes << ", ";
  os << "nTileSizes: " << schedule.nTileSizes << ", ";
  os << "kTileSizes: " << schedule.kTileSizes << ", ";
  os << "mSubgroupCounts: " << schedule.mSubgroupCounts << ", ";
  os << "nSubgroupCounts: " << schedule.nSubgroupCounts;
  if (!schedule.workgroupBatchSizes.empty()) {
    os << ", workgroupBatchSizes: " << schedule.workgroupBatchSizes;
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GemmSizeKind &gemmSize) {
  switch (gemmSize) {
  case GemmSizeKind::SmallGemm:
    return os << "SmallGemm";
  case GemmSizeKind::MediumGemm:
    return os << "MediumGemm";
  case GemmSizeKind::LargeGemm:
    return os << "LargeGemm";
  case GemmSizeKind::VeryLargeGemm:
    return os << "VeryLargeGemm";
  default:
    assert(false && "Unhandled gemm size");
    return os << "NotSet";
  }
}

static int64_t calculateOperandsSharedMemoryUsedInBytes(
    const GPUMMASchedule &schedule, int64_t lhsBitwidth, int64_t rhsBitwidth,
    int64_t lhsScaleBitwidth = 0, int64_t rhsScaleBitwidth = 0,
    int64_t numRhs = 1, bool useDirectLoad = false,
    int64_t prefetchNumStages = 0) {
  int64_t tileM = schedule.getTotalMSize() * schedule.getTotalMTileSize() *
                  schedule.getTotalMSubgroupCount();
  int64_t tileN = schedule.getTotalNSize() * schedule.getTotalNTileSize() *
                  schedule.getTotalNSubgroupCount();

  // For scaled matmul, the K dimension is split into Ko (outer) and Kb (block),
  // where elements in a Kb block share the same scale. For lhs and rhs we
  // account for both Ko and Kb, while for scale operands, only Ko. For regular
  // matmul, scale bitwidth is 0 so the scale terms below have no effect.
  int64_t tileK = schedule.getTotalKSize() * schedule.getTotalKTileSize();
  int64_t tileKb = schedule.kSizes.back() * schedule.kTileSizes.back();
  int64_t tileKo = tileK / tileKb;

  int64_t lhsSharedMemoryUsed = tileM * tileK * lhsBitwidth;
  int64_t rhsSharedMemoryUsed = numRhs * tileN * tileK * rhsBitwidth;
  int64_t aScaleSharedMemoryUsed = tileM * tileKo * lhsScaleBitwidth;
  int64_t bScaleSharedMemoryUsed = numRhs * tileN * tileKo * rhsScaleBitwidth;

  int64_t totalBits = lhsSharedMemoryUsed + rhsSharedMemoryUsed +
                      aScaleSharedMemoryUsed + bScaleSharedMemoryUsed;

  // In direct load mode, ROCDLPrefetchSharedMemoryPass multi-buffers shared
  // memory allocations, where the number of buffers equals prefetchNumStages.
  if (useDirectLoad && prefetchNumStages > 0) {
    totalBits *= prefetchNumStages;
  }

  return totalBits / 8;
}

static int64_t
calculateResultSharedMemoryUsedInBytes(const GPUMMASchedule &schedule,
                                       int64_t resultBitwidth,
                                       int64_t numRes = 1) {
  int64_t tileM = schedule.getTotalMSize() * schedule.getTotalMTileSize() *
                  schedule.getTotalMSubgroupCount();
  int64_t tileN = schedule.getTotalNSize() * schedule.getTotalNTileSize() *
                  schedule.getTotalNSubgroupCount();
  return (numRes * tileM * tileN * resultBitwidth) / 8;
}

/// Check that a GPUMMASchedule fits alignment restrictions. To be aligned,
/// the problem must be evenly divisible by the number of elements in the
/// schedule for each dimension. If `mustBeAligned` is false, then the problem
/// is allowed to be unaligned and the function simply returns true.
static bool isScheduleAligned(const GPUMatmulShapeType &problem,
                              const GPUMMASchedule &schedule,
                              bool mustBeAligned) {
  // If alignment is not required, skip checks and return true.
  if (!mustBeAligned) {
    return true;
  }
  // Returns the number of elements in the schedule for each dimension.
  auto getScheduleSizes = [&](ArrayRef<int64_t> intrinsicSizes,
                              ArrayRef<int64_t> tileCount,
                              std::optional<ArrayRef<int64_t>> subgroupCount) {
    SmallVector<int64_t> sizes = llvm::map_to_vector(
        llvm::seq<int64_t>(tileCount.size()), [&](int64_t i) {
          return subgroupCount ? tileCount[i] * subgroupCount.value()[i]
                               : tileCount[i];
        });
    // Multiply by intrinsic sizes, applying to the inner dimensions, as
    // the outer dimensions are unrolling factors. For example, if tileCount
    // = [a, b, c, d] and intrinsicSizes = [x, y], the result is [a, b, c*x,
    // d*y].
    assert(intrinsicSizes.size() <= sizes.size() &&
           "intrinsic sizes should not exceed tile count sizes");
    for (auto [intrinsicSize, size] :
         llvm::zip(llvm::reverse(intrinsicSizes), llvm::reverse(sizes))) {
      size *= intrinsicSize;
    }
    return sizes;
  };
  // Checks whether the elements of `a` are evenly divisible by the
  // corresponding elements of `b`.
  auto areAligned = [](ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
    for (auto [aVal, bVal] : llvm::zip_equal(a, b)) {
      if (aVal % bVal != 0) {
        return false;
      }
    }
    return true;
  };
  bool isValidM = areAligned(
      problem.mSizes, getScheduleSizes(schedule.mSizes, schedule.mTileSizes,
                                       schedule.mSubgroupCounts));
  bool isValidN = areAligned(
      problem.nSizes, getScheduleSizes(schedule.nSizes, schedule.nTileSizes,
                                       schedule.nSubgroupCounts));
  bool isValidK = areAligned(
      problem.kSizes,
      getScheduleSizes(schedule.kSizes, schedule.kTileSizes, std::nullopt));
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
  int64_t wgThreads = subgroupSize * schedule.getTotalMSubgroupCount() *
                      schedule.getTotalNSubgroupCount();
  int64_t mWgSize = schedule.getTotalMSize() * schedule.getTotalMTileSize() *
                    schedule.getTotalMSubgroupCount();
  int64_t nWgSize = schedule.getTotalNSize() * schedule.getTotalNTileSize() *
                    schedule.getTotalNSubgroupCount();
  int64_t kWgSize = schedule.getTotalKSize() * schedule.getTotalKTileSize();
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
        [](MutableArrayRef<int64_t> sizes) -> LogicalResult {
      for (int64_t &size : sizes) {
        if (size <= 1) {
          continue;
        }
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
         "expected intrinsic to have a single M, N, and K <= 2 dimensions");
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

  // Block intrinsics require the problem to have batch dimensions, and the
  // innermost problem batch dimension must be perfectly divisible by the
  // intrinsic batch size for simplicity and avoid overpadding.
  if (!intrinsic.batchSizes.empty()) {
    if (problem.batchSizes.empty() ||
        problem.batchSizes.back() % intrinsic.batchSizes.back() != 0) {
      return failure();
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
  const int64_t mSize = llvm::product_of(problem.mSizes);
  const int64_t nSize = llvm::product_of(problem.nSizes);
  // For block intrinsics, tighten the skinny threshold per dimension to the
  // minimum of the default threshold and half the intrinsic size in that
  // dimension, since smaller block intrinsics are themselves skinny.
  const int64_t mSkinnyThreshold =
      intrinsic.batchSizes.empty()
          ? kVerySkinnyDimThreshold
          : std::min(kVerySkinnyDimThreshold, intrinsic.mSizes[0] / 2);
  const int64_t nSkinnyThreshold =
      intrinsic.batchSizes.empty()
          ? kVerySkinnyDimThreshold
          : std::min(kVerySkinnyDimThreshold, intrinsic.nSizes[0] / 2);
  if ((mSize <= mSkinnyThreshold && (nSize > preferredSubgroupSize)) ||
      (nSize <= nSkinnyThreshold && (mSize > preferredSubgroupSize))) {
    return failure();
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

  assert(intrinsic.kSizes.size() <= 2 &&
         "expected intrinsic to have at most two K dimensions");

  // In the case of two K dimensions, we need to divide both seed values by the
  // last K dim prior to calculating the K tile count.
  int64_t bestKTileCountPerSubgroup = seeds.bestKTileCountPerSubgroup;
  int64_t bestKElementCountPerSubgroup = seeds.bestKElementCountPerSubgroup;
  if (intrinsic.kSizes.size() > 1) {
    bestKTileCountPerSubgroup =
        llvm::divideCeil(bestKTileCountPerSubgroup, intrinsic.kSizes[1]);
    bestKElementCountPerSubgroup =
        llvm::divideCeil(bestKElementCountPerSubgroup, intrinsic.kSizes[1]);
  }
  // Compute the ideal number of intrinsics along K per subgroup based on the
  // seed.
  bestKTileCountPerSubgroup =
      bestKElementCountPerSubgroup
          ? llvm::divideCeil(bestKElementCountPerSubgroup, intrinsic.kSizes[0])
          : bestKTileCountPerSubgroup;
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

/// Distributes tilesToDistribute to totalTiles using their GCD. Both
/// totalTiles and tilesToDistribute are updated to reflect the remaining
/// tiles to distribute. The return value is the number of tiles distributed.
static int64_t distributeTilesUsingGCD(int64_t &totalTiles,
                                       int64_t &tilesToDistribute) {
  APInt gcd = GreatestCommonDivisor(APInt(64, tilesToDistribute),
                                    APInt(64, totalTiles));
  int64_t distributeTileCount = gcd.getSExtValue();
  totalTiles /= distributeTileCount;
  tilesToDistribute /= distributeTileCount;

  return distributeTileCount;
}

/// Like distributeTilesUsingGCD but uses min instead of GCD. This handles
/// non-power-of-2 tile counts where GCD fails (e.g., prime tile counts).
static int64_t distributeTilesUsingMin(int64_t &totalTiles,
                                       int64_t &tilesToDistribute) {
  int64_t distributeTileCount = std::min(tilesToDistribute, totalTiles);
  totalTiles = llvm::divideCeil(totalTiles, distributeTileCount);
  tilesToDistribute /= distributeTileCount;
  return distributeTileCount;
}

/// Distributes the square root of the subgroup and tile counts to both M and N
/// dimensions. The first argument servers as a flag to indicate whether the
/// distribution is for the M or N dimension. Both total tiles and remaining
/// tiles are updated to reflect the remaining tiles to distribute. Note: This
/// function should only be used for primary distribution as it assigns the sqrt
/// directly to the dimension.
static void distributeSqrtForDim(
    bool isMDim, int64_t subgroupSqrt, int64_t tileSqrt,
    int64_t &mTotalTileToDistribute, int64_t &nTotalTileToDistribute,
    int64_t &mSubgroupDistributed, int64_t &nSubgroupDistributed,
    int64_t &mTileSizeDistributed, int64_t &nTileSizeDistributed,
    int64_t &remainingSubgroups, int64_t &remainingTiles) {
  if (isMDim) {
    mSubgroupDistributed = subgroupSqrt;
    mTileSizeDistributed = tileSqrt;
    mTotalTileToDistribute /= (subgroupSqrt * tileSqrt);
  } else {
    nSubgroupDistributed = subgroupSqrt;
    nTileSizeDistributed = tileSqrt;
    nTotalTileToDistribute /= (subgroupSqrt * tileSqrt);
  }

  remainingSubgroups /= subgroupSqrt;
  remainingTiles /= tileSqrt;
}

/// Distributes tiles and subgroups to both M and N dimensions using their GCD.
/// The first argument servers as a flag to indicate whether the distribution is
/// for the M or N dimension. Both total tiles and remaining tiles are updated
/// to reflect the remaining tiles to distribute.
static void distributeGCDForDim(bool isMDim, int64_t &mTotalTileToDistribute,
                                int64_t &nTotalTileToDistribute,
                                int64_t &mSubgroupDistributed,
                                int64_t &nSubgroupDistributed,
                                int64_t &mTileSizeDistributed,
                                int64_t &nTileSizeDistributed,
                                int64_t &remainingSubgroups,
                                int64_t &remainingTiles) {
  int64_t &totalTilesToDistribute =
      isMDim ? mTotalTileToDistribute : nTotalTileToDistribute;
  int64_t &subgroupDistributed =
      isMDim ? mSubgroupDistributed : nSubgroupDistributed;
  int64_t &tileDistributed =
      isMDim ? mTileSizeDistributed : nTileSizeDistributed;

  subgroupDistributed =
      distributeTilesUsingGCD(totalTilesToDistribute, remainingSubgroups);
  tileDistributed =
      distributeTilesUsingGCD(totalTilesToDistribute, remainingTiles);
}

/// Choose an optimal mma schedule with the heuristic that minimized the total
/// amount of data read from global memory, per workgroup, respecting the
/// heuristic seeds.
static GPUMMASchedule getOptimalMMASchedule(const GPUMatmulShapeType &problem,
                                            const GPUIntrinsicType &intrinsic,
                                            const GPUMMAHeuristicSeeds &seeds) {
  assert(intrinsic.mSizes.size() == 1 && intrinsic.nSizes.size() == 1 &&
         intrinsic.kSizes.size() <= 2 &&
         "expected intrinsic to have a single M, N, and K <= 2 dimensions");
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
  int64_t mTotalTileToDistribute = llvm::product_of(mTotalTileCounts);
  int64_t nTotalTileToDistribute = llvm::product_of(nTotalTileCounts);

  int64_t remainingSubgroups = seeds.bestSubgroupCountPerWorkgroup;
  int64_t remainingTiles = seeds.bestMNTileCountPerSubgroup;

  // Initial collapsed subgroup counts and tile sizes. Distribute to collapsed M
  // and N dimensions to avoid starving either dimension. Once the collapsed
  // distribution is determined, it will be distributed to individual dimensions
  // of M and N.
  int64_t mSubgroupDistributed = 1;
  int64_t nSubgroupDistributed = 1;
  int64_t mTileSizeDistributed = 1;
  int64_t nTileSizeDistributed = 1;

  LDBG() << "Starting MMA schedule distribution";
  LDBG() << "mTotalTileCounts: " << mTotalTileCounts
         << ", nTotalTileCounts: " << nTotalTileCounts
         << ", remainingSubgroups: " << remainingSubgroups
         << ", remainingTiles: " << remainingTiles;

  // This aims to be generous on subgroup splitting, produce the smallest
  // power-of-two that is >= sqrt(remainingSubgroups)
  int64_t subgroupSqrt =
      1ull << (llvm::divideCeil(llvm::Log2_64(remainingSubgroups), 2));
  // This aims to be conservative on tile splitting, produce the largest
  // power-of-two that is <= sqrt(remainingTiles)
  int64_t tileSqrt = 1ull << (llvm::Log2_64(remainingTiles) / 2);
  int64_t splitFactor = subgroupSqrt * tileSqrt;

  LDBG() << "splitFactor: " << splitFactor << ", subgroupSqrt: " << subgroupSqrt
         << ", tileSqrt: " << tileSqrt;

  // See if the square root can divide total tile count. If so it means we can
  // distribute to a dimensions evenly to minimize the number of global
  // loads. Or else fall back to GCD distribution.
  bool canMDistributeEvenly = mTotalTileToDistribute > splitFactor &&
                              mTotalTileToDistribute % splitFactor == 0;
  bool canNDistributeEvenly = nTotalTileToDistribute > splitFactor &&
                              nTotalTileToDistribute % splitFactor == 0;
  if (canMDistributeEvenly) {
    LDBG() << "Distributing seed evenly to M dim";
    distributeSqrtForDim(true, subgroupSqrt, tileSqrt, mTotalTileToDistribute,
                         nTotalTileToDistribute, mSubgroupDistributed,
                         nSubgroupDistributed, mTileSizeDistributed,
                         nTileSizeDistributed, remainingSubgroups,
                         remainingTiles);
    distributeGCDForDim(false, mTotalTileToDistribute, nTotalTileToDistribute,
                        mSubgroupDistributed, nSubgroupDistributed,
                        mTileSizeDistributed, nTileSizeDistributed,
                        remainingSubgroups, remainingTiles);
  } else if (canNDistributeEvenly) {
    LDBG() << "Distributing seed evenly to N dim";
    distributeSqrtForDim(false, subgroupSqrt, tileSqrt, mTotalTileToDistribute,
                         nTotalTileToDistribute, mSubgroupDistributed,
                         nSubgroupDistributed, mTileSizeDistributed,
                         nTileSizeDistributed, remainingSubgroups,
                         remainingTiles);
    distributeGCDForDim(true, mTotalTileToDistribute, nTotalTileToDistribute,
                        mSubgroupDistributed, nSubgroupDistributed,
                        mTileSizeDistributed, nTileSizeDistributed,
                        remainingSubgroups, remainingTiles);
  } else {
    LDBG() << "Distributing seed using GCD";
    distributeGCDForDim(false, mTotalTileToDistribute, nTotalTileToDistribute,
                        mSubgroupDistributed, nSubgroupDistributed,
                        mTileSizeDistributed, nTileSizeDistributed,
                        remainingSubgroups, remainingTiles);
    distributeGCDForDim(true, mTotalTileToDistribute, nTotalTileToDistribute,
                        mSubgroupDistributed, nSubgroupDistributed,
                        mTileSizeDistributed, nTileSizeDistributed,
                        remainingSubgroups, remainingTiles);
  }

  // Determine whether to use min-based distribution for per-dimension tile
  // assignment. Min-based is needed in two cases:
  //
  // 1) Degenerate schedule: GCD produced all-1 distributions because the tile
  //    counts are small odd numbers (e.g., 3x3 filter dims) with GCD=1 against
  //    power-of-2 seeds. Retry the collapsed distribution with min, then use
  //    min for per-dimension assignment.
  //
  // 2) Imbalanced problems: One dimension has 4x+ more tiles than the other
  //    (e.g., 149 vs 8 tiles). GCD fails for non-power-of-2 counts, so use
  //    min to redirect remaining tiles to the dominant dimension. Only apply
  //    when both dimensions have enough tiles (>= 8) to avoid hurting small
  //    shapes.
  bool isDegenerate = mSubgroupDistributed == 1 && nSubgroupDistributed == 1 &&
                      mTileSizeDistributed == 1 && nTileSizeDistributed == 1 &&
                      (remainingSubgroups > 1 || remainingTiles > 1);
  bool mMultiDim = problem.mSizes.size() > 1;
  bool nMultiDim = problem.nSizes.size() > 1;
  bool hasMultiDim = mMultiDim || nMultiDim;
  if (isDegenerate && hasMultiDim) {
    LDBG() << "Degenerate GCD schedule, using min-based distribution";
    remainingSubgroups = seeds.bestSubgroupCountPerWorkgroup;
    remainingTiles = seeds.bestMNTileCountPerSubgroup;
    // For single-dim M or N, only distribute if tiles >= 2x budget to avoid
    // over-distributing (e.g., 8 subgroups for 9 tiles). Multi-dim can
    // benefit even with fewer tiles since work spreads across sub-dims.
    auto distributeMin = [](int64_t &totalTiles, int64_t &distributed,
                            int64_t &budget, bool multiDim) {
      if (multiDim || totalTiles >= 2 * budget) {
        distributed *= distributeTilesUsingMin(totalTiles, budget);
      }
    };
    distributeMin(mTotalTileToDistribute, mSubgroupDistributed,
                  remainingSubgroups, mMultiDim);
    distributeMin(mTotalTileToDistribute, mTileSizeDistributed, remainingTiles,
                  mMultiDim);
    distributeMin(nTotalTileToDistribute, nSubgroupDistributed,
                  remainingSubgroups, nMultiDim);
    distributeMin(nTotalTileToDistribute, nTileSizeDistributed, remainingTiles,
                  nMultiDim);
  }

  // For heavily imbalanced problems (4:1+ tile ratio with both dims >= 8),
  // redirect remaining tiles to the dominant dimension.
  constexpr int64_t kMinTileCountThreshold = 8;
  int64_t minMNTileCount =
      std::min(mTotalTileCounts.back(), nTotalTileCounts.back());
  bool imbalancedM = minMNTileCount >= kMinTileCountThreshold &&
                     mTotalTileCounts.back() >= 4 * nTotalTileCounts.back();
  bool imbalancedN = minMNTileCount >= kMinTileCountThreshold &&
                     nTotalTileCounts.back() >= 4 * mTotalTileCounts.back();
  auto redirectRemainingTiles = [&](bool condition, int64_t totalTiles,
                                    int64_t &tileSizeDistributed) {
    if (!condition || remainingTiles <= 1) {
      return;
    }
    int64_t newTile = std::min(remainingTiles, totalTiles);
    if (newTile > tileSizeDistributed) {
      remainingTiles /= (newTile / tileSizeDistributed);
      tileSizeDistributed = newTile;
    }
  };
  redirectRemainingTiles(imbalancedM, mTotalTileToDistribute,
                         mTileSizeDistributed);
  redirectRemainingTiles(imbalancedN, nTotalTileToDistribute,
                         nTileSizeDistributed);

  LDBG() << "Leftover factors: subgroups: " << remainingSubgroups
         << ", tiles: " << remainingTiles;
  LDBG() << "Collapsed subgroup counts: M: " << mSubgroupDistributed
         << ", N: " << nSubgroupDistributed;
  LDBG() << "Collapsed tile sizes: M: " << mTileSizeDistributed
         << ", N: " << nTileSizeDistributed;

  // Distribute collapsed counts to per-dimension M and N (inner -> outer).
  // Use min-based distribution when GCD fails for that dimension: either
  // from a degenerate multi-dim schedule or an imbalanced problem.
  bool useMinForM = (isDegenerate && mMultiDim) || imbalancedM;
  bool useMinForN = (isDegenerate && nMultiDim) || imbalancedN;
  auto distributeToDims = [](MutableArrayRef<int64_t> tileCounts,
                             MutableArrayRef<int64_t> subgroupCounts,
                             MutableArrayRef<int64_t> tileSizes,
                             int64_t &subgroupBudget, int64_t &tileBudget,
                             bool useMin) {
    auto distribute = [useMin](int64_t &tiles, int64_t &budget) -> int64_t {
      if (!useMin) {
        return distributeTilesUsingGCD(tiles, budget);
      }
      // Skip min-based distribution when tiles/budget < 2 (e.g., 9/8 = 1) to
      // avoid over-distributing.
      if (tiles > budget && tiles / budget < 2) {
        return 1;
      }
      return distributeTilesUsingMin(tiles, budget);
    };
    for (size_t e = tileCounts.size(), i = e - 1; i < e; --i) {
      subgroupCounts[i] = distribute(tileCounts[i], subgroupBudget);
      tileSizes[i] = distribute(tileCounts[i], tileBudget);
    }
  };

  SmallVector<int64_t> mSubgroupCounts(problem.mSizes.size(), 0),
      nSubgroupCounts(problem.nSizes.size(), 0),
      mTileSizes(problem.mSizes.size(), 0),
      nTileSizes(problem.nSizes.size(), 0);

  distributeToDims(mTotalTileCounts, mSubgroupCounts, mTileSizes,
                   mSubgroupDistributed, mTileSizeDistributed, useMinForM);
  distributeToDims(nTotalTileCounts, nSubgroupCounts, nTileSizes,
                   nSubgroupDistributed, nTileSizeDistributed, useMinForN);

  SmallVector<int64_t> kTileSizes =
      getBestKTileSizes(problem, intrinsic, seeds);

  return GPUMMASchedule{intrinsic.mmaKind, intrinsic.mSizes, intrinsic.nSizes,
                        intrinsic.kSizes,  mSubgroupCounts,  nSubgroupCounts,
                        mTileSizes,        nTileSizes,       kTileSizes};
}

/// Compute the M*N utilization of an intrinsic for a given problem, measuring
/// the fraction of the intrinsic's M*N tile occupied by real data vs padding.
/// Returns a value in (0.0, 1.0]. A value < 1.0 indicates padding is needed.
static double computeMNUtilization(const GPUMatmulShapeType &problem,
                                   const GPUIntrinsicType &intrinsic) {
  double mUtil = std::min(problem.mSizes.back(), intrinsic.mSizes.back()) /
                 static_cast<double>(intrinsic.mSizes.back());
  double nUtil = std::min(problem.nSizes.back(), intrinsic.nSizes.back()) /
                 static_cast<double>(intrinsic.nSizes.back());
  return mUtil * nUtil;
}

/// Compare the MMA intrinsics by following precedence rules:
///   1) M*N utilization. When one intrinsic has significantly better (>= 2x)
///   M*N utilization, prefer it regardless of other rules. This avoids
///   choosing an intrinsic that wastes most of its compute on M/N padding
///   (e.g., 32x32 = 6.25% util vs 16x16 = 25% for an 8x8 problem).
///   2) K-alignment. We prefer intrinsics that can evenly divide the K
///   dimension of the problem.
///   3) M/N-alignment. We prefer intrinsics that can evenly divide the M
///   and N dimensions of the problem.
///   4) Intrinsic with larger gemm input size.
///   5) Intrinsic with larger K size.
///
/// This function acts as a comparison function object for std::sort, which
/// returns true if the lhs is ordered before rhs.
static bool compareIntrinsics(const GPUMatmulShapeType &problem,
                              const GPUIntrinsicType &lhs,
                              const GPUIntrinsicType &rhs,
                              bool preferHighComputeIntrinsic = false) {
  // When both M and N need padding, prefer the intrinsic with better M*N
  // utilization. This targets grouped convolutions where per-group channels
  // are small (e.g., 8x8 problem: 16x16 at 25% util >> 32x32 at 6.25%).
  // Only applies when both dims are smaller than the larger intrinsic's tile;
  // when only one dim is small, the larger intrinsic still helps on the big
  // dim.
  double lhsUtil = computeMNUtilization(problem, lhs);
  double rhsUtil = computeMNUtilization(problem, rhs);
  bool bothDimsNeedPadding =
      problem.mSizes.back() < std::max(lhs.mSizes.back(), rhs.mSizes.back()) &&
      problem.nSizes.back() < std::max(lhs.nSizes.back(), rhs.nSizes.back());
  bool hasUtilAdvantage =
      std::max(lhsUtil, rhsUtil) >= 2.0 * std::min(lhsUtil, rhsUtil);
  if (bothDimsNeedPadding && hasUtilAdvantage) {
    return lhsUtil > rhsUtil;
  }

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

  auto intrinsicCompute = [&](const GPUIntrinsicType &intrinsic) {
    return ShapedType::getNumElements(intrinsic.mSizes) *
           ShapedType::getNumElements(intrinsic.nSizes) *
           ShapedType::getNumElements(intrinsic.kSizes);
  };
  auto intrinsicArea = [&](const GPUIntrinsicType &intrinsic) {
    return (ShapedType::getNumElements(intrinsic.mSizes) +
            ShapedType::getNumElements(intrinsic.nSizes)) *
           ShapedType::getNumElements(intrinsic.kSizes);
  };

  // When both dims need padding with equal utilization, prefer smaller M*N
  // tile (less waste per instruction). For same-M*N intrinsics differing
  // only in K (e.g., 16x16x16 vs 16x16x32), prefer smaller K at moderate
  // utilization (>= 10%); at very low utilization, let later rules pick
  // larger K for better throughput.
  if (bothDimsNeedPadding && lhsUtil == rhsUtil) {
    int64_t lhsMN = ShapedType::getNumElements(lhs.mSizes) *
                    ShapedType::getNumElements(lhs.nSizes);
    int64_t rhsMN = ShapedType::getNumElements(rhs.mSizes) *
                    ShapedType::getNumElements(rhs.nSizes);
    if (lhsMN != rhsMN) {
      return lhsMN < rhsMN;
    }
    // lhsUtil == rhsUtil here, so checking one suffices.
    if (lhsUtil >= 0.10) {
      int64_t lhsCompute = intrinsicCompute(lhs);
      int64_t rhsCompute = intrinsicCompute(rhs);
      if (lhsCompute != rhsCompute) {
        return lhsCompute < rhsCompute;
      }
    }
  }

  // For compute-bound GEMMs, maximize compute throughput first, then
  // minimize operand VGPR pressure among equal-compute intrinsics.
  // E.g., 32x32x16 (compute=16384, area=1024) beats 32x32x8
  // (compute=8192, area=512) because throughput matters more. Among
  // 16x16x32 and 32x32x16 (both area=1024), prefer smaller K (16 vs 32)
  // for less operand staging pressure.
  if (problem.gemmSize == GemmSizeKind::VeryLargeGemm ||
      (problem.gemmSize == GemmSizeKind::LargeGemm &&
       preferHighComputeIntrinsic)) {
    int64_t lhsCompute = intrinsicCompute(lhs);
    int64_t rhsCompute = intrinsicCompute(rhs);
    if (lhsCompute != rhsCompute) {
      return lhsCompute > rhsCompute;
    }

    int64_t lhsArea = intrinsicArea(lhs);
    int64_t rhsArea = intrinsicArea(rhs);
    if (lhsArea != rhsArea) {
      return lhsArea < rhsArea;
    }

    return ShapedType::getNumElements(lhs.kSizes) <
           ShapedType::getNumElements(rhs.kSizes);
  }

  // For memory-bound GEMMs, prefer larger area to amortize memory latency.
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
                  ArrayRef<GPUIntrinsicType> intrinsics,
                  bool preferHighComputeIntrinsic = false) {
  SmallVector<GPUIntrinsicType> sortedIntrinsics(intrinsics);
  llvm::stable_sort(sortedIntrinsics, [&](const GPUIntrinsicType &lhs,
                                          const GPUIntrinsicType &rhs) {
    return compareIntrinsics(problem, lhs, rhs, preferHighComputeIntrinsic);
  });
  return sortedIntrinsics;
}

static int64_t computeEstimatedWorkgroupCount(const GPUMMAHeuristicSeeds &seeds,
                                              const GPUMatmulShapeType &problem,
                                              const GPUIntrinsicType &intrinsic,
                                              int64_t splitReductionTripCnt) {
  int64_t mSize = ShapedType::getNumElements(problem.mSizes);
  int64_t nSize = ShapedType::getNumElements(problem.nSizes);
  int64_t mnTileSizePerSubgroup = seeds.bestMNTileCountPerSubgroup *
                                  intrinsic.mSizes[0] * intrinsic.nSizes[0];
  int64_t workgroupSize =
      mnTileSizePerSubgroup * seeds.bestSubgroupCountPerWorkgroup;
  assert(workgroupSize > 0 && "workgroup size must be positive");
  int64_t numWorkgroups = mSize * nSize / workgroupSize;
  if (splitReductionTripCnt > 1) {
    numWorkgroups *= splitReductionTripCnt;
  }
  return numWorkgroups;
}

/// Adjust M*N tile-count (bestMNTileCountPerSubgroup) seeds based on target
/// hardware and problem characteristics. Four independent adjustments, applied
/// in order:
/// 1. Baseline (all targets): reduces bestMNTileCountPerSubgroup until the
///    estimated workgroup count fills all CUs.
/// 2. Tile-count boost (when boostMNTileCountPerSubgroup is set): for GEMMs
///    with balanced K, boosts tile count to the architecture-specific target.
/// 3. Utilization guard (when minUtilizationThreshold is set): halves tile
///    count until GPU utilization meets the threshold.
/// 4. VGPR pressure cap: limits MN tile count based on per-thread output
///    register pressure from the selected intrinsic, preventing spilling.
static void adjustSeedsForTarget(GPUMMAHeuristicSeeds &seeds,
                                 const GPUMatmulShapeType &problem,
                                 const GPUIntrinsicType &intrinsic,
                                 IREE::GPU::TargetAttr target,
                                 int64_t splitReductionTripCnt) {
  IREE::GPU::TargetChipAttr chip = target ? target.getChip() : nullptr;
  int64_t wgpCount = chip ? chip.getWgpCount() : 0;
  if (wgpCount == 0) {
    LDBG() << "WGP count unavailable, skipping seed adjustment.";
    return;
  }

  if (!problem.gemmSize || problem.gemmSize == GemmSizeKind::SmallGemm) {
    LDBG() << "Arithmetic intensity is too low, "
           << "skipping adjustment of seeds for workgroup count.";
    return;
  }

  // Baseline for all architectures: reduce MNT until workgroups fill CUs.
  int64_t numWorkgroups = computeEstimatedWorkgroupCount(
      seeds, problem, intrinsic, splitReductionTripCnt);
  LDBG() << "Estimated number of workgroups: " << numWorkgroups
         << ", WGP count: " << wgpCount;

  while (numWorkgroups < wgpCount) {
    if (seeds.bestMNTileCountPerSubgroup <= 1) {
      LDBG() << "Cannot decrease tile size further, "
                "bestMNTileCountPerSubgroup is already 1.";
      break;
    }
    seeds.bestMNTileCountPerSubgroup /= 2;
    LDBG() << "Decreasing bestMNTileCountPerSubgroup to "
           << seeds.bestMNTileCountPerSubgroup;
    numWorkgroups = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                                   splitReductionTripCnt);
  }

  // For GEMMs with balanced K dimensions (K <= max(M, N)), boost MNT to the
  // architecture-specific target to improve per-workgroup compute density
  // (more output elements per workgroup). The workload benefits from wider M*N
  // tiles rather than deeper K unrolling.
  if (seeds.boostMNTileCountPerSubgroup) {
    int64_t boostMNT = *seeds.boostMNTileCountPerSubgroup;
    int64_t mSize = ShapedType::getNumElements(problem.mSizes);
    int64_t nSize = ShapedType::getNumElements(problem.nSizes);
    int64_t kSize = ShapedType::getNumElements(problem.kSizes);
    int64_t boostedWGSize = boostMNT * intrinsic.mSizes[0] *
                            intrinsic.nSizes[0] *
                            seeds.bestSubgroupCountPerWorkgroup;
    bool kDominated = kSize > std::max(mSize, nSize);
    bool enoughOutput = mSize * nSize >= 2 * wgpCount * boostedWGSize;
    if (!kDominated && enoughOutput) {
      seeds.bestMNTileCountPerSubgroup =
          std::max(seeds.bestMNTileCountPerSubgroup, boostMNT);
      LDBG() << "Boosting MNT to " << seeds.bestMNTileCountPerSubgroup
             << " for balanced large gemm";
      // Halve subgroup count to offset the MNT boost, keeping the total
      // workgroup resource footprint (threads, LDS) in check for occupancy.
      seeds.bestSubgroupCountPerWorkgroup /= 2;
      LDBG() << "Halving subgroup count to "
             << seeds.bestSubgroupCountPerWorkgroup << " to offset MNT boost";
    }
  }

  // When a utilization threshold is set and workgroup count barely exceeds a
  // wave boundary, the last wave has most CUs idle. For example, 260
  // workgroups on 256 CUs gives 2 waves but only 50.8% utilization. Halve
  // MNT until utilization meets the threshold.
  if (seeds.minUtilizationThreshold) {
    double threshold = *seeds.minUtilizationThreshold;
    numWorkgroups = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                                   splitReductionTripCnt);
    auto computeUtilization = [&]() -> double {
      int64_t waves = llvm::divideCeil(numWorkgroups, wgpCount);
      if (waves == 0) {
        return 0.0;
      }
      return static_cast<double>(numWorkgroups) / (waves * wgpCount);
    };

    while (computeUtilization() < threshold) {
      if (seeds.bestMNTileCountPerSubgroup <= 1) {
        break;
      }
      seeds.bestMNTileCountPerSubgroup /= 2;
      numWorkgroups = computeEstimatedWorkgroupCount(seeds, problem, intrinsic,
                                                     splitReductionTripCnt);
      LDBG() << "Low utilization, decreasing MNT to "
             << seeds.bestMNTileCountPerSubgroup;
    }
  }

  // Cap per-subgroup MN tile count based on output VGPR pressure from the
  // selected intrinsic. Only applies when the MNT boost (step 2) is
  // configured, since the boost can push MN tile counts high enough to
  // cause spilling with large-output intrinsics (32x32). Capping at 128
  // output VGPRs per thread (8 MN tiles for 32x32, 32 for 16x16) prevents
  // spilling while preserving the boost for intrinsics that can handle
  // higher tile counts.
  if (seeds.boostMNTileCountPerSubgroup) {
    constexpr int64_t kMaxOutputVGPRsPerThread = 128;
    int64_t subgroupSize = target.getPreferredSubgroupSize();
    int64_t outputVGPRsPerTile =
        (intrinsic.mSizes[0] * intrinsic.nSizes[0]) / subgroupSize;
    int64_t maxMNTiles = kMaxOutputVGPRsPerThread / outputVGPRsPerTile;
    if (seeds.bestMNTileCountPerSubgroup > maxMNTiles) {
      LDBG() << "VGPR cap: reducing bestMNTileCountPerSubgroup from "
             << seeds.bestMNTileCountPerSubgroup << " to " << maxMNTiles
             << " (intrinsic " << intrinsic.mSizes[0] << "x"
             << intrinsic.nSizes[0] << ")";
      seeds.bestMNTileCountPerSubgroup = maxMNTiles;
    }
  }
}

FailureOr<GPUMMASchedule> deduceMMASchedule(
    const GPUMatmulShapeType &problem, ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &seeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, IREE::GPU::TargetAttr target, Location loc,
    bool transposedLhs, bool transposedRhs, bool canUpcastAcc,
    bool useDirectLoad, int64_t prefetchNumStages, bool mustBeAligned,
    bool doCPromotion, int64_t splitReductionTripCnt) {

  bool preferHighComputeIntrinsic =
      !doCPromotion && seeds.boostMNTileCountPerSubgroup.has_value();
  SmallVector<GPUIntrinsicType> sortedIntrinsics =
      sortMMAIntrinsics(problem, intrinsics, preferHighComputeIntrinsic);

  // Compute product of M and N problem sizes to decide if block intrinsics
  // should be considered. If both M and N products exceed the threshold, skip
  // block intrinsics as they are unlikely to be beneficial.
  bool allowBlockIntrinsics =
      llvm::product_of(problem.mSizes) <= 2 * kVerySkinnyDimThreshold ||
      llvm::product_of(problem.nSizes) <= 2 * kVerySkinnyDimThreshold;

  for (const GPUIntrinsicType &intrinsic : sortedIntrinsics) {
    // Skip block intrinsics if both M and N products are a fit for regular
    // intrinsics.
    bool isBlockIntrinsic = !intrinsic.batchSizes.empty();
    if (isBlockIntrinsic && !allowBlockIntrinsics) {
      continue;
    }
    if (failed(canTargetIntrinsic(problem, intrinsic, subgroupSize,
                                  canUpcastAcc, mustBeAligned))) {
      continue;
    }

    // Note: don't amend the original seeds, as deduceMMASchedule can be called
    // more than once in a row, and we want to keep the original seeds intact
    // for the next call.
    GPUMMAHeuristicSeeds localSeeds = seeds;
    adjustSeedsForTarget(localSeeds, problem, intrinsic, target,
                         splitReductionTripCnt);
    GPUMMASchedule schedule =
        getOptimalMMASchedule(problem, intrinsic, localSeeds);

    // Compute batch tile sizes. For block intrinsics the intrinsic itself
    // defines the batch tile size. Otherwise, when both M and N need padding
    // (problem size < intrinsic size), tile the static innermost batch dim up
    // to 4 to give each workgroup more useful work and amortize dispatch
    // overhead.
    SmallVector<int64_t, 2> wgBatchSizes(problem.batchSizes.size(), 1);
    if (!problem.batchSizes.empty()) {
      if (!intrinsic.batchSizes.empty()) {
        wgBatchSizes.back() = intrinsic.batchSizes.back();
        schedule.batchSizes.push_back(intrinsic.batchSizes.back());
      } else {
        int64_t innerBatch = problem.batchSizes.back();
        bool needsMNPadding =
            problem.mSizes.back() < schedule.getTotalMSize() &&
            problem.nSizes.back() < schedule.getTotalNSize();
        if (needsMNPadding && innerBatch % 2 == 0) {
          wgBatchSizes.back() = (innerBatch % 4 == 0) ? 4 : 2;
        }
      }
    }
    schedule.workgroupBatchSizes = wgBatchSizes;
    int64_t totalBatchTile = schedule.getTotalWorkgroupBatchSize();

    LDBG() << "Chosen MMA schedule:\n" << schedule;

    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      int64_t lhsBitwidth = problem.aType.getIntOrFloatBitWidth();
      int64_t rhsBitwidth = problem.bType.getIntOrFloatBitWidth();
      int64_t resultBitwidth = problem.cType.getIntOrFloatBitWidth();
      int64_t lhsScaleBitwidth =
          problem.aScaleType ? problem.aScaleType.getIntOrFloatBitWidth() : 0;
      int64_t rhsScaleBitwidth =
          problem.bScaleType ? problem.bScaleType.getIntOrFloatBitWidth() : 0;
      bool isAligned =
          isValidMMASchedule(problem, schedule, mustBeAligned, subgroupSize,
                             transposedLhs, transposedRhs);
      int64_t sharedMemoryUsed = calculateOperandsSharedMemoryUsedInBytes(
          schedule, lhsBitwidth, rhsBitwidth, lhsScaleBitwidth,
          rhsScaleBitwidth, problem.numHorizontallyFusedOps, useDirectLoad,
          prefetchNumStages);
      // Add accumulator/result memory when it uses shared memory (LDS):
      // - Result needs padding in shared memory, OR
      // - matmul_accumulate loads accumulator from global memory via shared mem
      // For zero-initialized GEMMs without C promotion, the accumulator stays
      // in registers and doesn't need shared memory.
      if (doCPromotion) {
        sharedMemoryUsed += calculateResultSharedMemoryUsedInBytes(
            schedule, resultBitwidth, problem.numHorizontallyFusedOps);
      }

      // Batch tiling multiplies the promoted operand sizes: each batch slice
      // uses separate shared memory, so total usage scales linearly.
      sharedMemoryUsed *= totalBatchTile;

      LDBG() << "Available Shared Memory: " << sharedMemLimitInBytes << " bytes"
             << "Predicted Shared Memory Used by Schedule: " << sharedMemoryUsed
             << " bytes (batch tile factor: " << totalBatchTile << ")";

      bool isValid = isAligned && sharedMemoryUsed <= sharedMemLimitInBytes;
      if (isValid) {
        // Only emit remark for the shared memory usage of the valid schedule.
        remark::analysis(loc, remark::RemarkOpts::name("SharedMemoryUsage")
                                  .category("deduceMMASchedule"))
            << std::to_string(sharedMemoryUsed);
      }
      return isValid;
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
         intrinsic.kSizes.size() <= 2 &&
         "expected intrinsic to have a single M, N, and K <= 2 dimensions");
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
  // Distribute tile sizes on N as much as we can as it's completely unrolled
  // and then distribute remaining tiles and subgroups on M.
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

static bool matchLayout(IREE::GPU::MMASingleSubgroupLayout layoutA,
                        IREE::GPU::MMASingleSubgroupLayout layoutB) {
  return (layoutA.element == layoutB.element) &&
         (layoutA.thread == layoutB.thread) &&
         (layoutA.tstrides == layoutB.tstrides);
};

FailureOr<std::pair<GPUMMASchedule, GPUMMASchedule>> deduceAttentionSchedule(
    const GPUMatmulShapeType &qkMatmul, const GPUMatmulShapeType &pvMatmul,
    ArrayRef<GPUIntrinsicType> intrinsics,
    const GPUMMAHeuristicSeeds &pvMatmulSeeds, int64_t sharedMemLimitInBytes,
    int64_t subgroupSize, bool transposedQ, bool transposedK, bool transposedV,
    bool canUpcastAcc, bool mustBeAligned) {
  SmallVector<uint64_t> qkViableIntrinsicIndices;
  SmallVector<uint64_t> pvViableIntrinsicIndices;
  for (const auto &[index, intrinsic] : llvm::enumerate(intrinsics)) {
    if (!failed(canTargetIntrinsic(qkMatmul, intrinsic, subgroupSize,
                                   canUpcastAcc, mustBeAligned))) {
      qkViableIntrinsicIndices.push_back(index);
    }
    if (!failed(canTargetIntrinsic(pvMatmul, intrinsic, subgroupSize,
                                   canUpcastAcc, mustBeAligned))) {
      pvViableIntrinsicIndices.push_back(index);
    }
  }

  std::vector<ChainedMMAIntrinsics> intrinsicPairs;
  for (unsigned qkIndex : qkViableIntrinsicIndices) {
    for (unsigned pvIndex : pvViableIntrinsicIndices) {
      const GPUIntrinsicType &intrinsicA = intrinsics[qkIndex];
      const GPUIntrinsicType &intrinsicB = intrinsics[pvIndex];
      if (!matchLayout(getSingleSubgroupLayout(intrinsicA.mmaKind,
                                               IREE::GPU::kMMAOperandAcc),
                       getSingleSubgroupLayout(intrinsicB.mmaKind,
                                               IREE::GPU::kMMAOperandAcc))) {
        continue;
      }

      // Check if we can reuse the output of intrinsicA for lhs/rhs of
      // intrinsicB.
      bool canReuseAOutForBLhs =
          matchLayout(getSingleSubgroupLayout(intrinsicA.mmaKind,
                                              IREE::GPU::kMMAOperandAcc),
                      getSingleSubgroupLayout(intrinsicB.mmaKind,
                                              IREE::GPU::kMMAOperandLhs));
      bool canReuseAOutForBRhs =
          matchLayout(getSingleSubgroupLayout(intrinsicA.mmaKind,
                                              IREE::GPU::kMMAOperandAcc),
                      getSingleSubgroupLayout(intrinsicB.mmaKind,
                                              IREE::GPU::kMMAOperandRhs));
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
    int64_t intrinsicAM = intrinsicA.mSizes[0];
    int64_t intrinsicAN = intrinsicA.nSizes[0];
    int64_t intrinsicAK = intrinsicA.kSizes[0];
    auto isValidSchedule = [&](const GPUMMASchedule &schedule) -> bool {
      // Create a mma schedule for qkMatmul in attention.
      // qkMatmul.M = pvMatmul.M
      // qkMatmul.N = pvMatmul.K
      // qkMatmul.K = problem.K
      SmallVector<int64_t, 2> qkKSizes = qkMatmul.kSizes;
      qkKSizes.back() = qkMatmul.kSizes.back() / intrinsicAK;
      GPUMMASchedule qkSchedule{
          intrinsicA.mmaKind,
          intrinsicAM,
          intrinsicAN,
          intrinsicAK,
          /*mSubgroupCount=*/schedule.mSubgroupCounts,
          /*nSubgroupCount=*/SmallVector<int64_t>(qkMatmul.nSizes.size(), 1),
          schedule.mTileSizes,
          schedule.kTileSizes,
          qkKSizes};

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
    SmallVector<int64_t, 2> qkKSizes = qkMatmul.kSizes;
    qkKSizes.back() = qkMatmul.kSizes.back() / intrinsicAK;
    GPUMMASchedule qkSchedule{
        intrinsicA.mmaKind,
        pvSchedule->mSizes,
        pvSchedule->kSizes,
        {intrinsicAK},
        /*mSubgroupCount=*/pvSchedule->mSubgroupCounts,
        /*nSubgroupCount=*/SmallVector<int64_t>(qkMatmul.nSizes.size(), 1),
        pvSchedule->mTileSizes,
        pvSchedule->kTileSizes,
        qkKSizes};

    return std::pair(qkSchedule, pvSchedule.value());
  }

  return failure();
}

} // namespace mlir::iree_compiler
