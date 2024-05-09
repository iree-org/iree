// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_PADTOINTRINSICS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

// Threshold used to determine whether a matmul dimension is 'very skinny'.
// Linked to LLVMGPU/KernelConfig.cpp
constexpr int64_t kVerySkinnyDimThreshold = 4;

static Value getPaddedValue(RewriterBase &rewriter, Location loc,
                            Value padSource, ArrayRef<OpFoldResult> padding) {
  auto sourceType = cast<RankedTensorType>(padSource.getType());
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  auto paddedShape =
      llvm::map_to_vector(llvm::zip_equal(sourceShape, padding), [](auto it) {
        std::optional<int64_t> padInt = getConstantIntValue(std::get<1>(it));
        if (ShapedType::isDynamic(std::get<0>(it)) || !padInt) {
          return ShapedType::kDynamic;
        }
        return std::get<0>(it) + padInt.value();
      });
  auto paddedResultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(sourceType.getElementType()));
  SmallVector<OpFoldResult> low(padding.size(), rewriter.getIndexAttr(0));
  Value paddedResult = rewriter.create<tensor::PadOp>(
      loc, paddedResultType, padSource, low, padding, paddingValue);
  return paddedResult;
}

static Value
getExpandedValue(RewriterBase &rewriter, Location loc, Value expandSource,
                 AffineMap &operandMap,
                 SmallVector<std::pair<int64_t, int64_t>> &dimsToExpand) {
  auto srcType = cast<RankedTensorType>(expandSource.getType());
  ArrayRef<int64_t> srcShape = srcType.getShape();
  SetVector<int64_t> operandDimsToExpand;
  SmallVector<std::optional<int64_t>> operandDimToExpandSize(srcType.getRank(),
                                                             std::nullopt);
  // Get dims to expand from operand's POV.
  for (auto [dimToExpand, sizeToExpand] : dimsToExpand) {
    std::optional<int64_t> maybeDim = operandMap.getResultPosition(
        getAffineDimExpr(dimToExpand, operandMap.getContext()));
    if (maybeDim) {
      operandDimsToExpand.insert(maybeDim.value());
      operandDimToExpandSize[maybeDim.value()] = sizeToExpand;
    }
  }
  if (operandDimsToExpand.empty()) {
    return expandSource;
  }

  // Form reassocation indices and expanded shape.
  SmallVector<ReassociationIndices> reassoc;
  SmallVector<int64_t> expandedShape;
  int64_t reassocOffset = 0;
  for (int i = 0; i < srcType.getRank(); i++) {
    if (operandDimToExpandSize[i]) {
      expandedShape.append({srcShape[i], operandDimToExpandSize[i].value()});
      reassoc.push_back(
          ReassociationIndices{reassocOffset + i, reassocOffset + i + 1});
      ++reassocOffset;
    } else {
      expandedShape.push_back(srcShape[i]);
      reassoc.push_back(ReassociationIndices{reassocOffset + i});
    }
  }

  return rewriter.create<tensor::ExpandShapeOp>(
      loc, RankedTensorType::Builder(srcType).setShape(expandedShape),
      expandSource, reassoc);
}

// expandMapsAndIterators expands the maps and iterators of an linalgOp at
// dimensions specified by `dimsToExpand`. The general idea is we want to view
// the dimensions specified in `dimsToExpand` as being splitted-up to two
// dimensions. Which means the src dim and the new child/sub dim will be
// contiguous.

// For example if we have:
// affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// iterator_types = ["parallel, "parallel", "reduce"]
// dimsToExpand = [1, 2]

// We will turn this into:
// affine_map<(d0, d1_A, d1_B, d2_A, d2_B) -> (d0, d2_A, d2_B, d1_A, d1_B)>
// iterator_types = ["parallel, "parallel", "parallel", "reduce", "reduce"]
static void
expandMapsAndIterators(SmallVector<AffineMap> &expandedMaps,
                       SmallVector<utils::IteratorType> &expandedIterators,
                       SmallVector<std::pair<int64_t, int64_t>> &dimsToExpand) {
  int expandOffset = 0;
  auto dimsToExpandVec =
      llvm::to_vector_of<int64_t>(llvm::make_first_range(dimsToExpand));
  llvm::sort(dimsToExpandVec);
  for (auto [expandIdx, expandDim] : llvm::enumerate(dimsToExpandVec)) {
    // Creating iterator type for newly expanded/dst dim from it's expand
    // source dim.
    int64_t expandSrcDim = expandDim + expandOffset;
    expandedIterators.insert(expandedIterators.begin() + expandSrcDim,
                             expandedIterators[expandSrcDim]);
    // Updating map of each operand to handle newly expanded/dst dim
    // based on the location of it's expand source dim.
    for (AffineMap &map : expandedMaps) {
      int64_t expandSrcDim = expandDim + expandOffset;
      int64_t expandDstDim = expandSrcDim + 1;
      map = map.shiftDims(1, expandDstDim);
      std::optional<int64_t> maybeDim = map.getResultPosition(
          getAffineDimExpr(expandSrcDim, map.getContext()));
      if (!maybeDim)
        continue;
      map = map.insertResult(getAffineDimExpr(expandDstDim, map.getContext()),
                             maybeDim.value() + 1);
    }
    expandOffset++;
  }
}

static SmallVector<GPUMatmulShapeType>
getIntrinsics(linalg::LinalgOp linalgOp) {
  ArrayAttr mmaKinds = nullptr;
  auto executableTargets =
      IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(linalgOp);
  if (executableTargets.size() != 1)
    return {};
  auto targetAttr = executableTargets.front();
  FailureOr<ArrayAttr> candidateMmaKinds =
      getSupportedMmaTypes(targetAttr.getConfiguration());
  if (failed(candidateMmaKinds))
    return {};
  mmaKinds = *candidateMmaKinds;

  return llvm::map_to_vector(
      mmaKinds.getAsRange<IREE::GPU::MMAAttr>(), [](IREE::GPU::MMAAttr mma) {
        auto [mSize, nSize, kSize] = mma.getMNKShape();
        auto [aType, bType, cType] = mma.getABCElementTypes();
        return GPUMatmulShapeType{mSize, nSize, kSize, aType, bType, cType};
      });
}

static void padConvOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp) {
  if (!isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
    return;
  }
  // TODO: Handle other variants.
  if (!isa<linalg::Conv2DNhwcHwcfOp>(linalgOp))
    return;

  // Early exit if cannot find intrinsics or if multiple executable targets.
  SmallVector<GPUMatmulShapeType> intrinsics = getIntrinsics(linalgOp);
  if (intrinsics.empty())
    return;

  // Check that conv has met conditions to go down mfma.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
      mlir::linalg::inferConvolutionDims(linalgOp);
  assert(succeeded(convolutionDims) && "Could not infer contraction dims");

  if (convolutionDims->outputChannel.size() != 1 ||
      convolutionDims->inputChannel.size() != 1 ||
      convolutionDims->filterLoop.size() < 1 ||
      convolutionDims->outputImage.size() < 1 ||
      convolutionDims->depth.size() != 0) {
    return;
  }

  auto isAllOnesList = [](ArrayRef<int64_t> list) {
    return llvm::all_of(list, [](int64_t i) { return i == 1; });
  };

  // TODO: Support non-unit strides/dilations.
  if (!isAllOnesList(convolutionDims->strides) ||
      !isAllOnesList(convolutionDims->dilations)) {
    return;
  }

  int64_t mDim = convolutionDims->outputImage.back();
  int64_t nDim = convolutionDims->outputChannel.front();
  // TODO: Support NCHW convolutions. This is just a matmul_transpose_a,
  // however the distribution patterns currently do not support that variant.
  if (mDim > nDim) {
    return;
  }
  int64_t kDim = convolutionDims->inputChannel.front();
  int64_t mSize = bounds[mDim];
  int64_t nSize = bounds[nDim];
  int64_t kSize = bounds[kDim];

  // TODO: Generalize to other dimensions.
  // Try to search for pad value and check only filter dimension is blocked.
  SmallVector<std::array<int64_t, 3>> mnkPaddingCandidates;
  for (const GPUMatmulShapeType &intrinsic : intrinsics) {
    std::optional<int64_t> mPadding, nPadding, kPadding;
    auto getPadding = [](int64_t value, int64_t padTo) {
      return llvm::divideCeil(value, padTo) * padTo - value;
    };

    if (mSize % intrinsic.mSize != 0) {
      mPadding = getPadding(mSize, intrinsic.mSize);
    }

    if (nSize % intrinsic.nSize != 0) {
      nPadding = getPadding(nSize, intrinsic.nSize);
    }

    if (kSize % intrinsic.kSize != 0) {
      kPadding = getPadding(kSize, intrinsic.kSize);
    }

    if (!mPadding && !nPadding && !kPadding) {
      // Some intrinsic matches. Nothing to do.
      return;
    }
    mnkPaddingCandidates.push_back(
        {mPadding.value_or(0), nPadding.value_or(0), kPadding.value_or(0)});
  }
  if (mnkPaddingCandidates.empty()) {
    return;
  }

  std::array<int64_t, 3> mnkPadding = mnkPaddingCandidates.front();

  Value newInput = linalgOp.getDpsInputOperand(0)->get();
  Value newFilter = linalgOp.getDpsInputOperand(1)->get();
  Value newOuts = linalgOp.getDpsInitOperand(0)->get();

  Location loc = linalgOp.getLoc();
  OpFoldResult mPadding = rewriter.getIndexAttr(mnkPadding[0]);
  OpFoldResult nPadding = rewriter.getIndexAttr(mnkPadding[1]);
  OpFoldResult kPadding = rewriter.getIndexAttr(mnkPadding[2]);
  OpFoldResult zero = rewriter.getIndexAttr(0);
  if (!isConstantIntValue(mPadding, 0) || !isConstantIntValue(kPadding, 0)) {
    // For NHWC, the m-padding is for W and k-padding is for C
    newInput = getPaddedValue(rewriter, loc, newInput,
                              {zero, zero, mPadding, kPadding});
  }
  if (!isConstantIntValue(nPadding, 0) || !isConstantIntValue(kPadding, 0)) {
    // For HWCF, the n-padding is for F and k-padding is for C
    newFilter = getPaddedValue(rewriter, loc, newFilter,
                               {zero, zero, kPadding, nPadding});
  }
  if (!isConstantIntValue(mPadding, 0) || !isConstantIntValue(nPadding, 0)) {
    // For output, the m-padding is for W and k-padding is for F
    newOuts = getPaddedValue(rewriter, loc, newOuts,
                             {zero, zero, mPadding, nPadding});
  }

  linalg::LinalgOp paddedConv2dOp =
      mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                  ArrayRef<Value>{newInput, newFilter, newOuts});
  // Extract slice.
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(4, zero);
  SmallVector<OpFoldResult> strides(4, one);
  auto resultType = cast<RankedTensorType>(linalgOp->getResult(0).getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(resultShape[0]),
                                     rewriter.getIndexAttr(resultShape[1]),
                                     rewriter.getIndexAttr(resultShape[2]),
                                     rewriter.getIndexAttr(resultShape[3])};
  Value extracted = rewriter.createOrFold<tensor::ExtractSliceOp>(
      loc, paddedConv2dOp->getResults()[0], offsets, sizes, strides);
  rewriter.replaceOp(linalgOp, extracted);
}

static void padContractionLikeOp(RewriterBase &rewriter,
                                 linalg::LinalgOp linalgOp) {
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);

  if (failed(contractionDims)) {
    return;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return;
  }

  // Early exit if cannot find intrinsics or if multiple executable targets.
  SmallVector<GPUMatmulShapeType> intrinsics = getIntrinsics(linalgOp);
  if (intrinsics.empty())
    return;

  Location loc = linalgOp.getLoc();

  // Naive handling by only looking into most inner dimensions.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // If none of the shape is dynamic, we'd fallback to using pad to intrinsics.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  int64_t mSize = bounds[mDim];
  int64_t nSize = bounds[nDim];
  int64_t kSize = bounds[kDim];

  // Bail out on matvec-like/skinny matmul cases.
  if ((!ShapedType::isDynamic(mSize) && mSize <= kVerySkinnyDimThreshold) ||
      (!ShapedType::isDynamic(nSize) && nSize <= kVerySkinnyDimThreshold)) {
    return;
  }

  // Util fn to get a linalgOp dim's src operand and dim from operand's POV.
  auto getSrcOperandAndDim =
      [&](int64_t targetDim) -> std::optional<std::pair<Value, int64_t>> {
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      auto operandMap = linalgOp.getMatchingIndexingMap(operand);
      std::optional<unsigned> maybeDim = operandMap.getResultPosition(
          getAffineDimExpr(targetDim, operandMap.getContext()));
      if (maybeDim)
        return std::pair{operand->get(), maybeDim.value()};
    }
    return std::nullopt;
  };

  // Utils to compute padding size.
  OpFoldResult zero = rewriter.getIndexAttr(0);
  AffineExpr s0, s1; // problemSize, intrinsicSize
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineExpr padByExpr = (s0).ceilDiv(s1) * s1 - s0;
  auto getPadding = [&](OpFoldResult value, int64_t padTo) {
    return affine::makeComposedFoldedAffineApply(
        rewriter, loc, padByExpr, {value, rewriter.getIndexAttr(padTo)});
  };

  SmallVector<SmallVector<std::pair<int64_t, int64_t>>> dimsToExpandCandidates;
  SmallVector<SmallVector<int64_t>> expandSizesCandidates;
  SmallVector<std::array<OpFoldResult, 3>> mnkPaddingCandidates;
  // Compute Pad and Expand metadata for linalgOp's M, N, K dimensions. Multiple
  // candidates of the metadata will be formed based on different intrinsics.
  for (GPUMatmulShapeType &intrinsic : intrinsics) {
    std::optional<OpFoldResult> mPadding, nPadding, kPadding;
    SmallVector<std::pair<int64_t, int64_t>> dimsToExpandCandidate;
    if (mSize % intrinsic.mSize != 0 || ShapedType::isDynamic(mSize)) {
      OpFoldResult mSizeExpr = rewriter.getIndexAttr(mSize);
      if (ShapedType::isDynamic(mSize)) {
        auto mOperandDimPair = getSrcOperandAndDim(mDim);
        if (!mOperandDimPair)
          return;
        auto [mOperand, mOperandDim] = mOperandDimPair.value();
        mSizeExpr = rewriter.create<tensor::DimOp>(loc, mOperand, mOperandDim)
                        .getResult();
        dimsToExpandCandidate.emplace_back(mDim, intrinsic.mSize);
      }
      mPadding = getPadding(mSizeExpr, intrinsic.mSize);
    }

    if (nSize % intrinsic.nSize != 0 || ShapedType::isDynamic(nSize)) {
      OpFoldResult nSizeExpr = rewriter.getIndexAttr(nSize);
      if (ShapedType::isDynamic(nSize)) {
        auto nOperandDimPair = getSrcOperandAndDim(nDim);
        if (!nOperandDimPair)
          return;
        auto [nOperand, nOperandDim] = nOperandDimPair.value();
        nSizeExpr = rewriter.create<tensor::DimOp>(loc, nOperand, nOperandDim)
                        .getResult();
        dimsToExpandCandidate.emplace_back(nDim, intrinsic.nSize);
      }
      nPadding = getPadding(nSizeExpr, intrinsic.nSize);
    }

    if (kSize % intrinsic.kSize != 0 || ShapedType::isDynamic(kSize)) {
      OpFoldResult kSizeExpr = rewriter.getIndexAttr(kSize);
      if (ShapedType::isDynamic(kSize)) {
        auto kOperandDimPair = getSrcOperandAndDim(kDim);
        if (!kOperandDimPair)
          return;
        auto [kOperand, kOperandDim] = kOperandDimPair.value();
        kSizeExpr = rewriter.create<tensor::DimOp>(loc, kOperand, kOperandDim)
                        .getResult();
        dimsToExpandCandidate.emplace_back(kDim, intrinsic.kSize);
      }
      kPadding = getPadding(kSizeExpr, intrinsic.kSize);
    }

    if (!mPadding && !nPadding && !kPadding) {
      return;
    }

    mnkPaddingCandidates.push_back({mPadding.value_or(zero),
                                    nPadding.value_or(zero),
                                    kPadding.value_or(zero)});
    dimsToExpandCandidates.push_back(dimsToExpandCandidate);
  }
  if (mnkPaddingCandidates.empty()) {
    return;
  }

  // Choose intrinsic and corresponding padding and expand parameters.
  std::array<OpFoldResult, 3> mnkPadding = mnkPaddingCandidates.front();
  SmallVector<int64_t, 3> mnkDim = {mDim, nDim, kDim};
  SmallVector<std::pair<int64_t, int64_t>> dimsToExpand =
      dimsToExpandCandidates.front();

  Value newLhs = linalgOp.getDpsInputOperand(0)->get();
  Value newRhs = linalgOp.getDpsInputOperand(1)->get();
  Value newOuts = linalgOp.getDpsInitOperand(0)->get();

  auto lhsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  auto rhsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(1));
  auto outsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));

  // Util fn to get padding of an operand from it's map;
  auto getOperandPadding =
      [&](AffineMap operandMap) -> SmallVector<OpFoldResult> {
    auto operandRank = operandMap.getNumResults();
    if (operandRank == 0)
      return {};
    SmallVector<OpFoldResult> operandPadding(operandRank, zero);
    for (auto [targetDim, targetPad] : llvm::zip(mnkDim, mnkPadding)) {
      std::optional<unsigned> maybeDim = operandMap.getResultPosition(
          getAffineDimExpr(targetDim, operandMap.getContext()));
      if (!maybeDim)
        continue;
      operandPadding[maybeDim.value()] = targetPad;
    }
    return operandPadding;
  };

  // Propagate padding info to operands.
  SmallVector<OpFoldResult> lhsPadding = getOperandPadding(lhsMap);
  SmallVector<OpFoldResult> rhsPadding = getOperandPadding(rhsMap);
  SmallVector<OpFoldResult> outsPadding = getOperandPadding(outsMap);
  if (lhsPadding.empty() || rhsPadding.empty() || outsPadding.empty()) {
    return;
  }

  // Pad operands based on padding info.
  newLhs = getPaddedValue(rewriter, loc, newLhs, lhsPadding);
  newRhs = getPaddedValue(rewriter, loc, newRhs, rhsPadding);
  newOuts = getPaddedValue(rewriter, loc, newOuts, outsPadding);

  auto paddedMatmulOp = mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                                    ArrayRef<Value>{newLhs, newRhs, newOuts});
  Value paddedCompute = paddedMatmulOp->getResults()[0];

  // Expand dimensions if there are dynamic shapes.
  if (!dimsToExpand.empty()) {
    // Generating expanded indexing maps and iterator types.
    SmallVector<AffineMap> expandedMaps = linalgOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> expandedIterators =
        linalgOp.getIteratorTypesArray();
    expandMapsAndIterators(expandedMaps, expandedIterators, dimsToExpand);

    // Propagate expand info and expand operands accordingly.
    newLhs = getExpandedValue(rewriter, loc, newLhs, lhsMap, dimsToExpand);
    newRhs = getExpandedValue(rewriter, loc, newRhs, rhsMap, dimsToExpand);
    newOuts = getExpandedValue(rewriter, loc, newOuts, outsMap, dimsToExpand);

    // Create expanded contractionOp.
    auto expandedMatmulOp = rewriter.create<linalg::GenericOp>(
        loc, newOuts.getType(), ValueRange{newLhs, newRhs}, ValueRange{newOuts},
        expandedMaps, expandedIterators);
    expandedMatmulOp.getRegion().takeBody(linalgOp->getRegion(0));
    paddedCompute = expandedMatmulOp.getResults()[0];

    // Collapse back to non expanded shape if required.
    if (auto expandOutsOp =
            dyn_cast<tensor::ExpandShapeOp>(newOuts.getDefiningOp())) {
      paddedCompute = rewriter.create<tensor::CollapseShapeOp>(
          loc, expandOutsOp.getSrcType(), paddedCompute,
          expandOutsOp.getReassociationIndices());
    }
  }

  // Extract slice.
  auto resultType = cast<RankedTensorType>(linalgOp->getResult(0).getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();
  int64_t resultRank = resultType.getRank();
  auto one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(resultRank, zero), strides(resultRank, one),
      sizes;
  for (auto [dimIdx, dimSize] : llvm::enumerate(resultShape)) {
    if (ShapedType::isDynamic(dimSize))
      sizes.push_back(rewriter
                          .create<tensor::DimOp>(
                              loc, linalgOp.getDpsInitOperand(0)->get(), dimIdx)
                          .getResult());
    else
      sizes.push_back(rewriter.getIndexAttr(dimSize));
  }
  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(linalgOp, paddedCompute,
                                                      offsets, sizes, strides);
}

struct PadToIntrinsicsPass
    : public impl::PadToIntrinsicsBase<PadToIntrinsicsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void PadToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  auto funcOp = getOperation();
  bool padConvOps = padTargetType == PadTargetType::ConvOp ||
                    padTargetType == PadTargetType::All;
  bool padContractionOps = padTargetType == PadTargetType::ContractionOp ||
                           padTargetType == PadTargetType::All;
  SmallVector<linalg::LinalgOp> targetConvOps;
  SmallVector<linalg::LinalgOp> targetContractOps;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa<linalg::Conv2DNhwcHwcfOp>(linalgOp.getOperation()) && padConvOps) {
      // Add convOps into worklist.
      targetConvOps.push_back(linalgOp);
    } else if (isa<linalg::BatchMatmulOp, linalg::MatmulOp,
                   linalg::MatmulTransposeBOp>(linalgOp.getOperation()) &&
               padContractionOps) {
      // Add named contractionOps into worklist.
      targetContractOps.push_back(linalgOp);
    } else if (isa<linalg::GenericOp>(linalgOp.getOperation()) &&
               linalg::isaContractionOpInterface(linalgOp) &&
               padContractionOps) {
      // Add named generic contractionOps into worklist.
      targetContractOps.push_back(linalgOp);
    }
  });

  // Iterate through and pad ops in the worklists.
  IRRewriter rewriter(context);
  for (auto convOp : targetConvOps) {
    rewriter.setInsertionPoint(convOp);
    padConvOp(rewriter, convOp);
  }
  for (auto contractOp : targetContractOps) {
    rewriter.setInsertionPoint(contractOp);
    padContractionLikeOp(rewriter, contractOp);
  }
}

} // namespace mlir::iree_compiler::Preprocessing
