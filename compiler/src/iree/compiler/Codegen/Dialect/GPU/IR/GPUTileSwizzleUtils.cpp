// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#define DEBUG_TYPE "gpu-tile-swizzle-utils"

namespace mlir::iree_compiler::IREE::GPU {

using ::mlir::iree_compiler::IREE::Codegen::TileSwizzle;
using Kind = TileSwizzle::Dim::Kind;

// Returns the index of the first destination dimension corresponding to the
// given source dimension `srcIdx`.
static int64_t expandedDimIdx(const TileSwizzle::ExpandShapeType &expandShape,
                              int srcIdx) {
  int dstIdx = 0;
  for (int i = 0; i < srcIdx; ++i) {
    dstIdx += expandShape[i].size();
  }
  return dstIdx;
}

// Pushes `dim` to the front of `swizzle.expandShape[srcIdx]`, and updates
// `swizzle.permutation` to make the new dimension outer-most among the dims in
// `swizzle.expandShape[srcIdx]`.
//
// This can be used to unroll a kernel with kind = CrossIntrinsic,
// or to expand a kernel to multiple subgroups with kind = CrossThread.
//
// Example:
//    Input swizzle = { expandShape = [[16], [4]], permutation = [1, 0] }
//    Input srcIdx = 1
//    Input dim.size = 4
// -> Output swizzle = { expandShape = [[16], [4, 4]], permutation = [1, 2, 0] }
//
static void expand(TileSwizzle &swizzle, int srcIdx, TileSwizzle::Dim dim) {
  int dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx);
  // The new unrolling dimension is inserted at the start of the expandShape
  // dimensions group corresponding to srcIdx.
  swizzle.expandShape[srcIdx].insert(swizzle.expandShape[srcIdx].begin(), dim);
  // Since we are not interleaving here, generating side-by-side copies of the
  // original layout, the new unrolling dimension is the new outermost
  // dimension. Existing entries get shifted to make room for it.
  for (auto &p : swizzle.permutation) {
    p += (p >= dstIdx);
  }
  swizzle.permutation.insert(swizzle.permutation.begin(), dstIdx);
}

// Interleaves the layout in `swizzle` by mutating `swizzle.permutation` to
// move permutation[0], the outer-most dimension (which the unroll() function
// created to be the unrolling dimension), to the inner dimension given by
// `expandedIdx`.
//
// Example:
//    Input swizzle = { expandShape = [[16], [4, 4]], permutation = [1, 2, 0] }
//    Input srcIdx = 1
//    Input expandedIdx = 1
// -> Output swizzle = { expandShape = [[16], [4, 4]], permutation = [2, 0, 1] }
//
static void interleave(TileSwizzle &swizzle, int srcIdx, int expandedIdx) {
  int dstIdx = expandedDimIdx(swizzle.expandShape, srcIdx) + expandedIdx;
  SmallVector<int64_t> outPermutation(swizzle.permutation.size());
  // The leading dimension, permutation[0], gets moved inwards to the
  // position that we just computed, dstIdx.
  outPermutation[dstIdx] = swizzle.permutation[0];
  // Outer dimensions get shifted outwards to fill the gap.
  for (int i = 0; i < dstIdx; ++i) {
    outPermutation[i] = swizzle.permutation[i + 1];
  }
  // Inner dimensions don't change.
  for (int i = dstIdx + 1; i < outPermutation.size(); ++i) {
    outPermutation[i] = swizzle.permutation[i];
  }
  swizzle.permutation = outPermutation;
}

static TileSwizzle getIntrinsicSwizzleBeforeMovingCrossThreadOutermost(
    IREE::GPU::MMAIntrinsic intrinsic, IREE::GPU::MMAFragment fragment) {
  auto layout = IREE::GPU::getSingleSubgroupLayout(intrinsic, fragment);

  // MMASingleSubgroupLayout has non-transposed RHS.
  // TileSwizzle has transposed RHS.
  if (fragment == IREE::GPU::MMAFragment::Rhs) {
    std::swap(layout.outer[0], layout.outer[1]);
    std::swap(layout.thread[0], layout.thread[1]);
    std::swap(layout.tstrides[0], layout.tstrides[1]);
    std::swap(layout.element[0], layout.element[1]);
  }

  TileSwizzle swizzle;
  // There are two source dimensions, corresponding to the arrays in `layout`
  // all having size 2. Let's just guard that assumption with one assert here.
  assert(layout.thread.size() == 2);
  swizzle.expandShape.resize(2);
  // Expand the shape from inner-most to outer-most dimension, so that we can
  // simply use the `expand` helper function, which creates new outer dims.
  // `layout.element` dims are inner-most, so we add them first.
  for (auto [i, e] : llvm::enumerate(layout.element)) {
    if (e != 1) {
      expand(swizzle, i, {Kind::Internal, e});
    }
  }
  // Next come `layout.thread` dims.
  for (auto [i, t] : llvm::enumerate(layout.thread)) {
    if (t != 1) {
      expand(swizzle, i, {Kind::CrossThread, t});
    }
  }
  // `layout.thread` dims are special in that they come with `layout.tstrides`
  // which may call for a swap in `swizzle.permutation`. We only need to worry
  // about that when both `layout.thread` sizes are greater than 1, so we didn't
  // skip them above. Note that this condition also implies that we don't need
  // to worry about `layout.tstrides == 0` which only happens with
  // `layout.thread == 1`.
  if (layout.thread[0] != 1 && layout.thread[1] != 1 &&
      layout.tstrides[0] > layout.tstrides[1]) {
    std::swap(swizzle.permutation[0], swizzle.permutation[1]);
  }
  // Finally come `layout.outer` dims, added last so they are outer-most.
  for (auto [i, o] : llvm::enumerate(layout.outer)) {
    if (o != 1) {
      expand(swizzle, i, {Kind::Internal, o});
    }
  }
  return swizzle;
}

static int getInnermostNonInternalDimIdx(
    const TileSwizzle::ExpandShapeDimVectorType &shape) {
  for (int idx = shape.size() - 1; idx >= 0; --idx) {
    if (shape[idx].kind != Kind::Internal) {
      return idx;
    }
  }
  assert(false && "all dimensions are internal!");
  return 0;
}

/// Moves all `Kind::CrossThread` dims of the Acc layout to the outermost
/// within their expand shape reassociation groups. This only moves the cross
/// thread dims of the Acc layout because we want to fuse the unset_encoding
/// ops with the data tiled matmul. In order to do this, the sliced dimensions
/// (CrossThread) for each thread need to be outermost in the final write out.
///
/// This transformation is for the Acc layout, but the Lhs and Rhs layouts need
/// to be transformed too, because the layouts need to match the Acc for their
/// respective M and N tile dimensions.
///
/// Example (CrossThread dims are denoted with surrounding {} braces):
///   Input:
///     Lhs layout:
///       expandShape = [[8, {16}], [2, {4}, 4]],
///       permutation = [0, 3, 1, 2, 4]
///     Rhs layout:
///       expandShape = [[{4}, 2, {16}], [2, {4}, 4]],
///       permutation = [0, 1, 4, 2, 3, 5]
///     Acc layout:
///       expandShape = [[8, {4}, 4], [{4}, 2, {16}]],
///       permutation = [3, 0, 4, 1, 5, 2]
///   Output:
///     Lhs layout:
///       expandShape = [[{4}, 8, {4}], [2, {4}, 4]],
///       permutation = [1, 4, 0, 2, 3, 5]
///     Rhs layout:
///       expandShape = [[{4}, {16}, 2], [2, {4}, 4]],
///       permutation = [0, 2, 4, 1, 3, 5]
///     Acc layout:
///       expandShape = [[{4}, 8, 4], [{4}, {16}, 2]],
///       permutation = [3, 1, 5, 0, 4, 2]
static TileSwizzle moveCrossThreadOutermost(TileSwizzle swizzle,
                                            TileSwizzle accSwizzle,
                                            MMAFragment fragment) {
  assert(accSwizzle.expandShape.size() == 2);
  assert(swizzle.expandShape.size() == 2);
  TileSwizzle::ExpandShapeType expandShape = swizzle.expandShape;
  TileSwizzle::ExpandShapeType accExpandShape = accSwizzle.expandShape;
  // We will construct the permutation on the flattened `expandShape` dims that
  // will move the cross thread dims to outermost, and store the resulting
  // permutation in `crossThreadToOuterPerm`. This will be used later to adjust
  // the swizzle.permutation properly.
  SmallVector<int64_t> crossThreadToOuterPerm;
  int groupStartIdx = 0;
  for (int accGroupIdx = 0; accGroupIdx < accExpandShape.size();
       ++accGroupIdx) {
    // Index for corresponding `swizzle` group is 0 for Lhs and Rhs fragments,
    // since the 1 index of Rhs and Lhs are K dimensions.
    int groupIdx = fragment == MMAFragment::Acc ? accGroupIdx : 0;
    // Skip N group for Lhs.
    if (fragment == MMAFragment::Lhs && accGroupIdx == 1) {
      continue;
    }
    // Skip M group for Rhs.
    if (fragment == MMAFragment::Rhs && accGroupIdx == 0) {
      continue;
    }
    TileSwizzle::ExpandShapeDimVectorType group = expandShape[groupIdx];
    TileSwizzle::ExpandShapeDimVectorType accGroup =
        accExpandShape[accGroupIdx];
    // The expanded shape of the `accSwizzle` group may not necessarily match
    // the expanded shape of the `swizzle` group, so we may need to expand the
    // dimensions of `group` further so that the is a corresponding dimension
    // in `group` for every CrossThread dimension in `accGroup`.
    //
    // For example:
    //   Lhs swizzle.expandShape = [[8, {16}], [2, {4}, 4]],
    //   Acc swizzle.expandShape = [[8, {4}, 4], [{4}, 2, {16}]],
    //
    // For group 0 the Lhs swizzle.expandShape has shape [8, {16}], while the
    // Acc swizzle.expandShape has shape [8, {4}, 4]. Since, the CrossThread dim
    // of the Acc swizzle group does not appear in the Lhs swizzle group, we
    // need to expand the Lhs swizzle group so we can permute the groups in the
    // same way. In this case the Lhs group's {16} dim will be expanded, and the
    // new Lhs swizzle group will be [8, {4}, {4}].
    if (accGroup.size() != group.size()) {
      SmallVector<int64_t> accGroupShape =
          llvm::map_to_vector(accGroup, [](TileSwizzle::Dim d) {
            return static_cast<int64_t>(d.size);
          });
      SmallVector<int64_t> groupShape =
          llvm::map_to_vector(group, [](TileSwizzle::Dim d) {
            return static_cast<int64_t>(d.size);
          });
      std::optional<SmallVector<ReassociationIndices>> groupReassociation =
          getReassociationIndicesForCollapse(accGroupShape, groupShape);
      // For the current MFMA layouts, there should always be a reassociation
      // found, since the ACC layout is always an expanded form of the combined
      // LHS and RHS layouts.
      assert(groupReassociation.has_value() &&
             "expected to find reassociation");
      TileSwizzle::ExpandShapeDimVectorType expandedGroup;
      for (auto [i, reassociation] : llvm::enumerate(*groupReassociation)) {
        for (int64_t reInd : reassociation) {
          expandedGroup.push_back(
              TileSwizzle::Dim(group[i].kind, accGroup[reInd].size));
        }
        int expandedPermIdx;
        for (auto [permIdx, permDim] : llvm::enumerate(swizzle.permutation)) {
          if (permDim > i) {
            permDim += reassociation.size() - 1;
          }
          if (permDim == i) {
            expandedPermIdx = permIdx;
          }
        }
        for (int j = 0, e = reassociation.size() - 1; j < e; ++j) {
          swizzle.permutation.insert(
              swizzle.permutation.begin() + expandedPermIdx + j + 1, i + j + 1);
        }
      }
      swizzle.expandShape[groupIdx] = expandedGroup;
    }

    // At this point, the `group` and `accGroup` will have the same shape, so
    // we can compute a permutation for `accGroup` that would move the Acc
    // CrossThread dims outermost, and then use that exact permutation for the
    // `group`. Compute the localized permutation within the acc reassociation
    // group, and apply to the expandShape dims within the `group`.
    SmallVector<int64_t> crossThreadInds;
    SmallVector<int64_t> otherInds;
    for (int64_t idx = 0; idx < accGroup.size(); ++idx) {
      TileSwizzle::Dim dim = accGroup[idx];
      if (dim.kind == TileSwizzle::Dim::Kind::CrossThread) {
        crossThreadInds.push_back(idx);
      } else {
        otherInds.push_back(idx);
      }
    }
    SmallVector<int64_t> groupPerm(crossThreadInds);
    groupPerm.append(otherInds);
    applyPermutationToVector(swizzle.expandShape[groupIdx], groupPerm);

    // Append the group permutation to the global `crossThreadToOuterPerm`.
    // `groupPerm` contains the local permutation within the expand shape
    // reassociation group, so we need to convert to the global permutation
    // indices when adding to the global crossThreadToOuterPerm.
    for (int64_t idx : groupPerm) {
      crossThreadToOuterPerm.push_back(idx + groupStartIdx);
    }
    groupStartIdx += expandShape[groupIdx].size();
  }

  // The matching groups bewteen `accSwizzle` and `swizzle` have now been
  // permuted. For Lhs and Rhs fragments, we need to fill in the rest of the
  // permutation from the skipped groups that don't appear in the `accSwizzle`.
  if (fragment != MMAFragment::Acc) {
    for (int64_t i = swizzle.expandShape.front().size();
         i < swizzle.permutation.size(); ++i) {
      crossThreadToOuterPerm.push_back(i);
    }
  }

  // At this point, the expandShape dims have been permuted within their groups,
  // but we still need to adjust the swizzle.permutation to preserve the result
  // shape of the swizzle. We have the following permutations:
  //  - perm(originalSrc -> crossThreadOuterSrc)
  //  - perm(originalSrc -> result)
  // And we want `perm(crossThreadOuterSrc -> result)`, so we need to take
  // `inverse(perm(originalSrc -> crossThreadOuterSrc))`, and then apply
  // `perm(originalSrc -> result)`.
  SmallVector<int64_t> perm = invertPermutationVector(crossThreadToOuterPerm);
  applyPermutationToVector(perm, swizzle.permutation);
  swizzle.permutation = perm;
  return swizzle;
}

/// Return the full swizzle without any reordering of CrossThread dims. The
/// result of this function should be passed to moveCrossThreadOutermost to
/// get the final swizzle.
static TileSwizzle
getSwizzleBeforeMovingCrossThreadOutermost(IREE::GPU::DataTiledMMAAttr mma,
                                           IREE::GPU::MMAFragment fragment) {
  auto swizzle = getIntrinsicSwizzleBeforeMovingCrossThreadOutermost(
      mma.getIntrinsic(), fragment);
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs:
    // A-matrix (LHS). Source dimensions are M (index 0) and K (index 1).
    // Unroll on K with interleaving, then on M.
    if (mma.getIntrinsicsK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getIntrinsicsK()});
      int interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getIntrinsicsM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getIntrinsicsM()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
    break;
  case IREE::GPU::MMAFragment::Rhs:
    // B-matrix (RHS). Since the pack ops already took care of transposing B,
    // source dimensions are N (index 0) and K (index 1).
    // Unroll on K with interleaving, then on N.
    if (mma.getIntrinsicsK() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getIntrinsicsK()});
      int interleavingIdx =
          getInnermostNonInternalDimIdx(swizzle.expandShape[1]);
      interleave(swizzle, 1, interleavingIdx);
    }
    if (mma.getIntrinsicsN() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getIntrinsicsN()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    break;
  case IREE::GPU::MMAFragment::Acc:
    // C-matrix (accumulator). Source dimensions are M (index 0) and N (index
    // 1). Unroll on N, then on M.
    if (mma.getIntrinsicsN() > 1) {
      expand(swizzle, 1, {Kind::CrossIntrinsic, mma.getIntrinsicsN()});
    }
    if (mma.getIntrinsicsM() > 1) {
      expand(swizzle, 0, {Kind::CrossIntrinsic, mma.getIntrinsicsM()});
    }
    if (mma.getSubgroupsN() > 1) {
      expand(swizzle, 1, {Kind::CrossThread, mma.getSubgroupsN()});
    }
    if (mma.getSubgroupsM() > 1) {
      expand(swizzle, 0, {Kind::CrossThread, mma.getSubgroupsM()});
    }
    break;
  }
  return swizzle;
}

TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                       IREE::GPU::MMAFragment fragment) {
  TileSwizzle swizzle =
      getSwizzleBeforeMovingCrossThreadOutermost(mma, fragment);
  // We want to move the CrossThread dims to be outermost in the source layout
  // for the result. We need the transformations for the Lhs and Rhs to match
  // with the Acc transformation, so we need to know what the acc swizzle is
  // when moving CrossThread dims, even when the fragment is Lhs or Rhs.
  TileSwizzle accSwizzle = swizzle;
  if (fragment != IREE::GPU::MMAFragment::Acc) {
    accSwizzle = getSwizzleBeforeMovingCrossThreadOutermost(
        mma, IREE::GPU::MMAFragment::Acc);
  }
  LLVM_DEBUG(llvm::dbgs() << fragment
                          << " swizzle before moving CrossThread dims: "
                          << swizzle << "\n");
  TileSwizzle crossThreadOuterSwizzle =
      moveCrossThreadOutermost(swizzle, accSwizzle, fragment);
  LLVM_DEBUG(llvm::dbgs() << fragment
                          << " swizzle after moving CrossThread dims: "
                          << crossThreadOuterSwizzle << "\n\n");
  return crossThreadOuterSwizzle;
}

/// Remove the expanded dimensions for this index and update the permutation by
/// erasing the removed dimensions' indices and adjusting existing larger
/// indices accordingly.
static void remove(TileSwizzle &swizzle, size_t idx) {
  assert(idx < swizzle.expandShape.size() && "idx out of bounds");
  const size_t startIdx = std::accumulate(
      std::begin(swizzle.expandShape), std::begin(swizzle.expandShape) + idx, 0,
      [](size_t idx, const TileSwizzle::ExpandShapeDimVectorType &dims)
          -> size_t { return idx + dims.size(); });
  const size_t endIdx = startIdx + swizzle.expandShape[idx].size();
  swizzle.expandShape.erase(swizzle.expandShape.begin() + idx);
  SmallVector<int64_t> newPermutation;
  for (const int64_t &p : swizzle.permutation) {
    if (p < startIdx) {
      newPermutation.push_back(p);
    } else if (p >= endIdx) {
      newPermutation.push_back(p - (endIdx - startIdx));
    }
  }
  swizzle.permutation = newPermutation;
}

FailureOr<TileSwizzle> getEncodingSwizzle(IREE::Encoding::EncodingAttr encoding,
                                          IREE::GPU::DataTiledMMAAttr mma,
                                          IREE::GPU::MMAFragment fragment) {
  TileSwizzle swizzle = getSwizzle(mma, fragment);
  FailureOr<linalg::ContractionDimensions> cDims =
      Encoding::getEncodingContractionDims(encoding);
  if (failed(cDims)) {
    return failure();
  }
  // The following expects M, N, K, and Batch sizes of at most 1 for now.
  // TODO: Extend this to multiple M/N/K/Batch dims.
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() == 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  std::optional<unsigned> mDim =
      cDims->m.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->m[0]);
  std::optional<unsigned> nDim =
      cDims->n.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims->k[0]);
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs:
    // A-matrix (LHS). Source dimensions are M (index 0) and K (index 1).
    // Dimensions are removed from last to first to ensure correctness.
    if (!kDim.has_value()) {
      remove(swizzle, 1);
    }
    if (!cDims->m.empty() && !mDim.has_value()) {
      remove(swizzle, 0);
    }
    break;
  case IREE::GPU::MMAFragment::Rhs:
    // B-matrix (RHS). Since the pack ops already took care of transposing B,
    // source dimensions are N (index 0) and K (index 1).
    // Dimensions are removed from last to first to ensure correctness.
    if (!kDim.has_value()) {
      remove(swizzle, 1);
    }
    if (!cDims->n.empty() && !nDim.has_value()) {
      remove(swizzle, 0);
    }
    break;
  case IREE::GPU::MMAFragment::Acc:
    // C-matrix (accumulator). Source dimensions are M (index 0) and N (index
    // 1).
    // Dimensions are removed from last to first to ensure correctness.
    if (!cDims->n.empty() && !nDim.has_value()) {
      remove(swizzle, 1);
    }
    if (!cDims->m.empty() && !mDim.has_value()) {
      remove(swizzle, 0);
    }
    break;
  }
  return swizzle;
}

TileSwizzle getIntrinsicSwizzle(IREE::GPU::MMAIntrinsic intrinsic,
                                IREE::GPU::MMAFragment fragment) {
  auto swizzle =
      getIntrinsicSwizzleBeforeMovingCrossThreadOutermost(intrinsic, fragment);
  TileSwizzle accSwizzle = swizzle;
  if (fragment != IREE::GPU::MMAFragment::Acc) {
    accSwizzle = getIntrinsicSwizzleBeforeMovingCrossThreadOutermost(
        intrinsic, IREE::GPU::MMAFragment::Acc);
  }
  LLVM_DEBUG(
      llvm::dbgs() << fragment
                   << " intrinsic swizzle before moving CrossThread dims: "
                   << swizzle << "\n");
  TileSwizzle crossThreadOuterSwizzle =
      moveCrossThreadOutermost(swizzle, accSwizzle, fragment);
  LLVM_DEBUG(
      llvm::dbgs() << fragment
                   << " intrinsic swizzle after moving CrossThread dims: "
                   << crossThreadOuterSwizzle << "\n\n");
  return crossThreadOuterSwizzle;
}

} // namespace mlir::iree_compiler::IREE::GPU
