// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_PADTOINTRINSICSPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

// Threshold used to determine whether a matmul dimension is 'very skinny'.
// Based on the same variable in LLVMGPU/KernelConfig.cpp.
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

static SmallVector<GPUIntrinsicType>
getIntrinsics(linalg::LinalgOp linalgOp,
              ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargets) {
  IREE::GPU::TargetAttr target;
  if (executableTargets.size() == 1) {
    auto targetAttr = executableTargets.front();
    target = getGPUTargetAttr(targetAttr);
  } else {
    // For LIT testing, also directly search TargetAttr around the op.
    target = getGPUTargetAttr(linalgOp);
  }
  if (!target)
    return {};

  IREE::GPU::MMAOpsArrayAttr mmaKinds = target.getWgp().getMma();

  return llvm::map_to_vector(mmaKinds, [](IREE::GPU::MMAAttr mma) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    return GPUIntrinsicType{mSize, nSize, kSize, aType, bType, cType, mma};
  });
}

static void
padConvOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
          ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargets) {
  // Early exit if cannot find intrinsics or if multiple executable targets.
  SmallVector<GPUIntrinsicType> intrinsics =
      getIntrinsics(linalgOp, executableTargets);
  if (intrinsics.empty())
    return;

  // Check that conv has met conditions to go down mfma.
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
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
  // In NCHW convolutions, mDim > nDim and the position of the input with filter
  // tensors will be swapped in igemm passes later.
  bool isIGemmOperandSwapped = mDim > nDim;

  int64_t kDim = convolutionDims->inputChannel.front();
  int64_t mSize = bounds[mDim];
  int64_t nSize = bounds[nDim];
  int64_t kSize = bounds[kDim];

  auto inpElemType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType())
          .getElementType();
  auto kernelElemType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType())
          .getElementType();

  SmallVector<std::array<int64_t, 3>> mnkPaddingCandidates;
  for (const GPUMatmulShapeType &intrinsic : intrinsics) {

    if (!(inpElemType == intrinsic.aType &&
          kernelElemType == intrinsic.bType)) {
      continue;
    }

    std::optional<int64_t> mPadding, nPadding, kPadding;
    auto getPadding = [](int64_t value, int64_t padTo) {
      return llvm::divideCeil(value, padTo) * padTo - value;
    };

    auto mIntrinsicSize =
        isIGemmOperandSwapped ? intrinsic.nSizes[0] : intrinsic.mSizes[0];
    auto nIntrinsicSize =
        isIGemmOperandSwapped ? intrinsic.mSizes[0] : intrinsic.nSizes[0];

    if (mSize % intrinsic.mSizes[0] != 0) {
      mPadding = getPadding(mSize, mIntrinsicSize);
    }

    if (nSize % intrinsic.nSizes[0] != 0) {
      nPadding = getPadding(nSize, nIntrinsicSize);
    }

    if (kSize % intrinsic.kSizes[0] != 0) {
      kPadding = getPadding(kSize, intrinsic.kSizes[0]);
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
  Value newOutput = linalgOp.getDpsInitOperand(0)->get();

  auto indexingMaps = linalgOp.getIndexingMapsArray();
  auto inputMap = indexingMaps[0];
  auto filterMap = indexingMaps[1];
  auto outputMap = indexingMaps[2];

  Location loc = linalgOp.getLoc();
  OpFoldResult mPadding = rewriter.getIndexAttr(mnkPadding[0]);
  OpFoldResult nPadding = rewriter.getIndexAttr(mnkPadding[1]);
  OpFoldResult kPadding = rewriter.getIndexAttr(mnkPadding[2]);
  OpFoldResult zero = rewriter.getIndexAttr(0);

  auto createExprToIdMap = [](AffineMap map) {
    llvm::SmallDenseMap<AffineExpr, unsigned> exprToIdMap;
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      exprToIdMap[map.getResult(i)] = i;
    }
    return exprToIdMap;
  };

  auto applyPadding = [&](AffineMap map, OpFoldResult padding1,
                          OpFoldResult padding2, unsigned dim1, unsigned dim2,
                          Value &paddingTarget) {
    if (!isZeroInteger(padding1) || !isZeroInteger(padding2)) {
      llvm::SmallDenseMap<AffineExpr, unsigned> exprToIdMap =
          createExprToIdMap(map);
      auto id1 = exprToIdMap[getAffineDimExpr(dim1, map.getContext())];
      auto id2 = exprToIdMap[getAffineDimExpr(dim2, map.getContext())];

      llvm::SmallVector<OpFoldResult> paddingValues(map.getNumResults(), zero);
      paddingValues[id1] = padding1;
      paddingValues[id2] = padding2;
      paddingTarget =
          getPaddedValue(rewriter, loc, paddingTarget, paddingValues);
    }
  };

  applyPadding(inputMap, mPadding, kPadding, mDim, kDim, newInput);
  applyPadding(filterMap, nPadding, kPadding, nDim, kDim, newFilter);
  applyPadding(outputMap, mPadding, nPadding, mDim, nDim, newOutput);

  linalg::LinalgOp paddedConv2dOp =
      mlir::clone(rewriter, linalgOp, {newOutput.getType()},
                  ArrayRef<Value>{newInput, newFilter, newOutput});

  // Extract slice.
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  RankedTensorType outputType = cast<RankedTensorType>(newOutput.getType());
  int64_t outputRank = outputType.getRank();
  SmallVector<OpFoldResult> offsets(outputRank, zero);
  SmallVector<OpFoldResult> strides(outputRank, one);

  auto resultType = cast<RankedTensorType>(linalgOp->getResult(0).getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();
  SmallVector<OpFoldResult> sizes;
  for (int i = 0; i < outputRank; i++) {
    sizes.push_back(rewriter.getIndexAttr(resultShape[i]));
  }
  Value extracted = rewriter.createOrFold<tensor::ExtractSliceOp>(
      loc, paddedConv2dOp->getResults()[0], offsets, sizes, strides);
  rewriter.replaceOp(linalgOp, extracted);
}

static void padContractionLikeOp(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp,
    ArrayRef<IREE::HAL::ExecutableTargetAttr> executableTargets) {
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
  SmallVector<GPUIntrinsicType> intrinsics =
      getIntrinsics(linalgOp, executableTargets);
  if (intrinsics.empty())
    return;

  Location loc = linalgOp.getLoc();

  // Naive handling by only looking into most inner dimensions.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // If none of the shape is dynamic, we'd fallback to using pad to intrinsics.
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
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
    if (mSize % intrinsic.mSizes[0] != 0 || ShapedType::isDynamic(mSize)) {
      OpFoldResult mSizeExpr = rewriter.getIndexAttr(mSize);
      if (ShapedType::isDynamic(mSize)) {
        auto mOperandDimPair = getSrcOperandAndDim(mDim);
        if (!mOperandDimPair)
          return;
        auto [mOperand, mOperandDim] = mOperandDimPair.value();
        mSizeExpr = rewriter.create<tensor::DimOp>(loc, mOperand, mOperandDim)
                        .getResult();
        dimsToExpandCandidate.emplace_back(mDim, intrinsic.mSizes[0]);
      }
      mPadding = getPadding(mSizeExpr, intrinsic.mSizes[0]);
    }

    if (nSize % intrinsic.nSizes[0] != 0 || ShapedType::isDynamic(nSize)) {
      OpFoldResult nSizeExpr = rewriter.getIndexAttr(nSize);
      if (ShapedType::isDynamic(nSize)) {
        auto nOperandDimPair = getSrcOperandAndDim(nDim);
        if (!nOperandDimPair)
          return;
        auto [nOperand, nOperandDim] = nOperandDimPair.value();
        nSizeExpr = rewriter.create<tensor::DimOp>(loc, nOperand, nOperandDim)
                        .getResult();
        dimsToExpandCandidate.emplace_back(nDim, intrinsic.nSizes[0]);
      }
      nPadding = getPadding(nSizeExpr, intrinsic.nSizes[0]);
    }

    if (kSize % intrinsic.kSizes[0] != 0 || ShapedType::isDynamic(kSize)) {
      OpFoldResult kSizeExpr = rewriter.getIndexAttr(kSize);
      if (ShapedType::isDynamic(kSize)) {
        auto kOperandDimPair = getSrcOperandAndDim(kDim);
        if (!kOperandDimPair)
          return;
        auto [kOperand, kOperandDim] = kOperandDimPair.value();
        kSizeExpr = rewriter.create<tensor::DimOp>(loc, kOperand, kOperandDim)
                        .getResult();
        dimsToExpandCandidate.emplace_back(kDim, intrinsic.kSizes[0]);
      }
      kPadding = getPadding(kSizeExpr, intrinsic.kSizes[0]);
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
    : public impl::PadToIntrinsicsPassBase<PadToIntrinsicsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void PadToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  auto moduleOp = getOperation();
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    return signalPassFailure();
  }
  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return signalPassFailure();
  }

  bool padConvOps = padTargetType == PadTargetType::ConvOp ||
                    padTargetType == PadTargetType::All;
  bool padContractionOps = padTargetType == PadTargetType::ContractionOp ||
                           padTargetType == PadTargetType::All;
  SmallVector<linalg::LinalgOp> targetConvOps;
  SmallVector<linalg::LinalgOp> targetContractOps;
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (linalg::isaConvolutionOpInterface(linalgOp) && padConvOps) {
        targetConvOps.push_back(linalgOp);
      } else if (isa<linalg::BatchMatmulOp, linalg::MatmulOp,
                     linalg::MatmulTransposeBOp>(linalgOp.getOperation()) &&
                 padContractionOps) {
        targetContractOps.push_back(linalgOp);
      } else if (isa<linalg::GenericOp>(linalgOp.getOperation()) &&
                 linalg::isaContractionOpInterface(linalgOp) &&
                 padContractionOps) {
        targetContractOps.push_back(linalgOp);
      }
    });
  }

  // Iterate through and pad ops in the worklists.
  auto getRequiredExecutableTargetAttrs = [&](Operation *op) {
    SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    SmallVector<IREE::Stream::AffinityAttr> affinityAttrs;
    if (affinityAnalysis.tryInferExecutionAffinity(op, affinityAttrs)) {
      for (auto affinityAttr : affinityAttrs) {
        deviceAnalysis.gatherRequiredExecutableTargets(affinityAttr, op,
                                                       executableTargetAttrs);
      }
    }
    return executableTargetAttrs;
  };
  IRRewriter rewriter(context);
  for (auto convOp : targetConvOps) {
    rewriter.setInsertionPoint(convOp);
    auto executableTargetAttrs = getRequiredExecutableTargetAttrs(convOp);
    padConvOp(rewriter, convOp, executableTargetAttrs.getArrayRef());
  }
  for (auto contractOp : targetContractOps) {
    rewriter.setInsertionPoint(contractOp);
    auto executableTargetAttrs = getRequiredExecutableTargetAttrs(contractOp);
    padContractionLikeOp(rewriter, contractOp,
                         executableTargetAttrs.getArrayRef());
  }
}

} // namespace mlir::iree_compiler::Preprocessing
