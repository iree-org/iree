// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/WinogradConstants.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
namespace {

static void computeLoopParams(SmallVectorImpl<Value> &lbs,
                              SmallVectorImpl<Value> &ubs,
                              SmallVectorImpl<Value> &steps, Value tensor,
                              int numImageDims, Location loc,
                              OpBuilder &builder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<OpFoldResult> dimValues =
      tensor::getMixedSizes(builder, loc, tensor);
  for (int i = numImageDims; i < dimValues.size(); i++) {
    lbs.push_back(zero);
    ubs.push_back(getValueOrCreateConstantIndexOp(builder, loc, dimValues[i]));
    steps.push_back(one);
  }
}

/// Tile iree_linalg_ext.winograd.input_transform op.
/// TODO: Adopt getTiledImplementation with this.
static LogicalResult tileWinogradInputTransformOp(
    WinogradInputTransformOp inputOp, RewriterBase &rewriter,
    WinogradInputTransformOp &tiledWinogradInputTransformOp) {
  Location loc = inputOp.getLoc();
  auto funcOp = inputOp->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp) {
    return rewriter.notifyMatchFailure(inputOp,
                                       "Could not find parent function");
  }

  const int64_t inputTileSize = inputOp.getInputTileSize();
  const int64_t outputTileSize = inputOp.getOutputTileSize();
  switch (outputTileSize) {
  case 6:
    break;
  default:
    return failure();
  }

  Value input = inputOp.input();
  Value output = inputOp.output();
  auto outputType = output.getType().cast<ShapedType>();
  auto inputType = input.getType().cast<ShapedType>();
  SmallVector<int64_t> inputShape(inputType.getShape());
  const bool isNchw = inputOp.isNchw();
  if (isNchw) {
    permute<Permutation::NCHW_TO_NHWC>(inputShape);
  }
  Type elementType = outputType.getElementType();
  const std::array<int64_t, 2> imageDims = inputOp.nhwcImageDimensions();
  const size_t numImageDims = imageDims.size();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);

  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  SmallVector<Value> lbs, ubs, steps;
  computeLoopParams(lbs, ubs, steps, output, numImageDims, loc, rewriter);
  // Construct loops
  rewriter.setInsertionPoint(inputOp);
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps, ValueRange({output}),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });

  // Extract input slice
  auto one = rewriter.getIndexAttr(1);
  auto zero = rewriter.getIndexAttr(0);
  auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
  SmallVector<OpFoldResult> strides(inputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> sizes(inputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> offsets(inputOp.getInputOperandRank(), zero);
  SmallVector<Value> ivs;
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }
  for (int i = 0; i < inputShape.size(); i++) {
    if (!imageDimsSet.contains(i)) {
      offsets[i] = ivs[i];
    } else {
      rewriter.setInsertionPointToStart(loopNest.loops[i].getBody());
      AffineExpr dim0;
      auto it = rewriter.getAffineConstantExpr(inputTileSize);
      auto ot = rewriter.getAffineConstantExpr(outputTileSize);
      auto delta = rewriter.getAffineConstantExpr(inputShape[i]);
      bindDims(rewriter.getContext(), dim0);
      AffineMap scaleMap =
          AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
      offsets[i] = rewriter.createOrFold<affine::AffineApplyOp>(
          loc, scaleMap, ValueRange{ivs[i]});
      AffineMap minMap =
          AffineMap::get(1, 0, {-dim0 + delta, it}, rewriter.getContext());
      sizes[i] = rewriter.createOrFold<affine::AffineMinOp>(
          loc, minMap,
          ValueRange{
              getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i])});
    }
  }
  rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
  auto tensorType = RankedTensorType::get(
      SmallVector<int64_t>(numImageDims, ShapedType::kDynamic), elementType);
  if (isNchw) {
    permute<Permutation::NHWC_TO_NCHW>(offsets);
    permute<Permutation::NHWC_TO_NCHW>(sizes);
  }
  Value dynamicSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, tensorType, input, offsets, sizes, strides);

  // Extract output slice
  auto stridesOutputSlice =
      SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
  auto offsetsOutputSlice = SmallVector<OpFoldResult>(numImageDims, zero);
  offsetsOutputSlice.append(ivs.begin(), ivs.end());
  auto sizesOutputSlice =
      SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
  sizesOutputSlice[0] = sizesOutputSlice[1] = inputTileSizeAttr;
  tensorType = RankedTensorType::get(inputTileSquare, elementType);
  Value iterArg = loopNest.loops.back().getRegionIterArg(0);
  Value outputSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, tensorType, iterArg, offsetsOutputSlice, sizesOutputSlice,
      stridesOutputSlice);

  IntegerAttr outputTileSizeI64Attr =
      rewriter.getI64IntegerAttr(inputOp.getOutputTileSize());
  IntegerAttr kernelSizeI64Attr =
      rewriter.getI64IntegerAttr(inputOp.getKernelSize());
  DenseI64ArrayAttr imageDimensionsDenseI64ArrayAttr =
      rewriter.getDenseI64ArrayAttr(inputOp.imageDimensions());
  tiledWinogradInputTransformOp = rewriter.create<WinogradInputTransformOp>(
      loc, tensorType, dynamicSlice, outputSlice, outputTileSizeI64Attr,
      kernelSizeI64Attr, imageDimensionsDenseI64ArrayAttr);

  // Insert results into output slice
  Value updatedOutput = rewriter.create<tensor::InsertSliceOp>(
      loc, tiledWinogradInputTransformOp.getResult()[0], iterArg,
      offsetsOutputSlice, sizesOutputSlice, stridesOutputSlice);

  // Replace returned value
  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          loopNest.loops.back().getBody()->getTerminator())) {
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, updatedOutput);
  }
  inputOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);

  return success();
}

static void computeForallAndForUpperBounds(OpBuilder &builder, Location loc,
                                           SmallVector<Value> ubs,
                                           uint64_t workgroupSize,
                                           SmallVector<Value> &forallUbs,
                                           SmallVector<Value> &forUbs,
                                           SmallVector<int64_t> &forallInds,
                                           SmallVector<int64_t> &forInds) {
  uint64_t totalIters = 1;
  for (int64_t idx = 0; idx < ubs.size() - 1; ++idx) {
    auto ub = ubs[idx];
    auto constUb = getConstantIntValue(ub);
    if (!constUb.has_value()) {
      forUbs.push_back(ub);
      forInds.push_back(idx);
      continue;
    }
    totalIters *= constUb.value();
    if (totalIters <= workgroupSize) {
      forallUbs.push_back(ub);
      forallInds.push_back(idx);
    } else {
      forUbs.push_back(ub);
      forInds.push_back(idx);
    }
  }
  auto innerTile = getConstantIntValue(ubs.back());
  if (!innerTile.has_value()) {
    forUbs.push_back(ubs.back());
    forInds.push_back(ubs.size() - 1);
    return;
  }
  if (innerTile.value() * totalIters <= workgroupSize) {
    forallUbs.push_back(ubs.back());
    forallInds.push_back(ubs.size() - 1);
    return;
  }
  int64_t forallInnerTile = llvm::bit_floor(workgroupSize / totalIters);
  int64_t forInnerTile = innerTile.value() / forallInnerTile;
  Value forallUb = builder.create<arith::ConstantIndexOp>(loc, forallInnerTile);
  Value forUb = builder.create<arith::ConstantIndexOp>(loc, forInnerTile);
  forallUbs.push_back(forallUb);
  forallInds.push_back(ubs.size() - 1);
  forUbs.push_back(forUb);
  forInds.push_back(ubs.size() - 1);
}

/// Tile iree_linalg_ext.winograd.input_transform op.
/// TODO: Adopt getTiledImplementation with this.
static LogicalResult tileWinogradInputTransformOpWithForall(
    WinogradInputTransformOp inputOp, RewriterBase &rewriter,
    WinogradInputTransformOp &tiledWinogradInputTransformOp) {
  Location loc = inputOp.getLoc();
  auto funcOp = inputOp->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp) {
    return rewriter.notifyMatchFailure(inputOp,
                                       "Could not find parent function");
  }

  auto workgroupSizes = llvm::map_to_vector(
      getEntryPoint(funcOp)->getWorkgroupSize().value(),
      [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });
  int64_t workgroupSize = computeProduct(workgroupSizes);
  const int64_t inputTileSize = inputOp.getInputTileSize();
  const int64_t outputTileSize = inputOp.getOutputTileSize();
  switch (outputTileSize) {
  case 6:
    break;
  default:
    return failure();
  }

  Value input = inputOp.input();
  Value output = inputOp.output();
  auto outputType = output.getType().cast<ShapedType>();
  auto inputType = input.getType().cast<ShapedType>();
  SmallVector<int64_t> inputShape(inputType.getShape());
  const bool isNchw = inputOp.isNchw();
  if (isNchw) {
    permute<Permutation::NCHW_TO_NHWC>(inputShape);
  }
  Type elementType = outputType.getElementType();
  const std::array<int64_t, 2> imageDims = inputOp.nhwcImageDimensions();
  const size_t numImageDims = imageDims.size();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);

  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  SmallVector<Value> lbs, ubs, steps;
  computeLoopParams(lbs, ubs, steps, output, numImageDims, loc, rewriter);
  // Construct loops
  // SmallVector<Value> dest;
  // if (failed(tensor::getOrCreateDestinations(rewriter, loc, inputOp, dest)))
  //   return inputOp->emitOpError("failed to get destination tensors");
  auto getThreadMapping = [&](int64_t dim) {
    auto mappingIdInt = std::min<int64_t>(
        dim + static_cast<uint64_t>(gpu::MappingId::LinearDim0),
        gpu::getMaxEnumValForMappingId());
    return mlir::gpu::GPUThreadMappingAttr::get(
        inputOp->getContext(), gpu::symbolizeMappingId(mappingIdInt).value());
  };
  SmallVector<int64_t> forallInds, forInds;
  SmallVector<Value> forallUbs, forUbs;
  computeForallAndForUpperBounds(rewriter, loc, ubs, workgroupSize, forallUbs,
                                 forUbs, forallInds, forInds);
  SmallVector<Attribute> idDims;
  for (int i = 0; i < forallUbs.size(); i++) {
    idDims.push_back(getThreadMapping(i));
  }
  std::reverse(idDims.begin(), idDims.end());
  ArrayAttr mapping = rewriter.getArrayAttr(idDims);
  scf::LoopNest loopNest;
  auto loopNestSelect = [&](SmallVector<Value> vals) {
    return llvm::map_to_vector(forInds, [&](auto ind) { return vals[ind]; });
  };
  if (!forInds.empty()) {
    rewriter.setInsertionPoint(inputOp);
    loopNest = scf::buildLoopNest(
        rewriter, loc, loopNestSelect(lbs), forUbs, loopNestSelect(steps),
        ValueRange({output}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });
  }

  Location forallLoc =
      loopNest.loops.empty() ? loc : loopNest.loops.back()->getLoc();
  Value forallOut = loopNest.loops.empty()
                        ? output
                        : loopNest.loops.back().getRegionIterArg(0);
  if (!loopNest.loops.empty()) {
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
  } else {
    rewriter.setInsertionPoint(inputOp);
  }
  scf::ForallOp forallOp = rewriter.create<scf::ForallOp>(
      forallLoc, getAsOpFoldResult((forallUbs)), forallOut, mapping);
  Value iterArg = forallOp.getRegionOutArgs()[0];
  SmallVector<Value> forAllIvs = forallOp.getInductionVars();
  int64_t forIdx = 0, forallIdx = 0;
  SmallVector<Value> ivs;
  SetVector<int64_t> forIndsSet(forInds.begin(), forInds.end());
  SetVector<int64_t> forallIndsSet(forallInds.begin(), forallInds.end());
  for (int64_t idx = 0; idx < ubs.size() - 1; ++idx) {
    if (forIndsSet.contains(idx)) {
      ivs.push_back(loopNest.loops[forIdx++].getInductionVar());
    } else {
      ivs.push_back(forAllIvs[forallIdx++]);
    }
  }
  if (forallUbs.size() + forUbs.size() > ubs.size()) {
    rewriter.setInsertionPointToStart(forallOp.getBody());
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0);
    bindSymbols(rewriter.getContext(), s1);
    AffineMap map = AffineMap::get(0, 2, {s0 * s1}, rewriter.getContext());
    Value iv = rewriter.createOrFold<affine::AffineApplyOp>(
        forallOp->getLoc(), map,
        ValueRange{loopNest.loops[forIdx++].getInductionVar(),
                   forAllIvs[forallIdx++]});
    ivs.push_back(iv);
  } else if (forIndsSet.contains(ubs.size() - 1)) {
    ivs.push_back(loopNest.loops[forIdx++].getInductionVar());
  } else {
    ivs.push_back(forAllIvs[forallIdx++]);
  }

  // Extract input slice
  auto one = rewriter.getIndexAttr(1);
  auto zero = rewriter.getIndexAttr(0);
  auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
  SmallVector<OpFoldResult> strides(inputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> sizes(inputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> offsets(inputOp.getInputOperandRank(), zero);
  forIdx = 0;

  rewriter.setInsertionPointToStart(forallOp.getBody());
  for (int i = 0; i < inputShape.size(); i++) {
    OpBuilder::InsertionGuard g(rewriter);
    if (!forallIndsSet.contains(i)) {
      rewriter.setInsertionPoint(
          loopNest.loops[forIdx++].getBody()->getTerminator());
    }
    if (!imageDimsSet.contains(i)) {
      offsets[i] = ivs[i];
    } else {
      AffineExpr dim0;
      auto it = rewriter.getAffineConstantExpr(inputTileSize);
      auto ot = rewriter.getAffineConstantExpr(outputTileSize);
      auto delta = rewriter.getAffineConstantExpr(inputShape[i]);
      bindDims(rewriter.getContext(), dim0);
      AffineMap scaleMap =
          AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
      offsets[i] = rewriter.createOrFold<affine::AffineApplyOp>(
          loc, scaleMap, ValueRange{ivs[i]});
      AffineMap minMap =
          AffineMap::get(1, 0, {-dim0 + delta, it}, rewriter.getContext());
      sizes[i] = rewriter.createOrFold<affine::AffineMinOp>(
          loc, minMap,
          ValueRange{
              getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i])});
    }
  }
  rewriter.setInsertionPoint(forallOp.getBody()->getTerminator());
  auto tensorType = RankedTensorType::get(
      SmallVector<int64_t>(numImageDims, ShapedType::kDynamic), elementType);
  if (isNchw) {
    permute<Permutation::NHWC_TO_NCHW>(offsets);
    permute<Permutation::NHWC_TO_NCHW>(sizes);
  }
  Value dynamicSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, tensorType, input, offsets, sizes, strides);

  // Extract output slice
  auto stridesOutputSlice =
      SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
  auto offsetsOutputSlice = SmallVector<OpFoldResult>(numImageDims, zero);
  offsetsOutputSlice.append(ivs.begin(), ivs.end());
  auto sizesOutputSlice =
      SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
  sizesOutputSlice[0] = sizesOutputSlice[1] = inputTileSizeAttr;
  tensorType = RankedTensorType::get(inputTileSquare, elementType);
  // Value outputSlice = rewriter.create<tensor::ExtractSliceOp>(
  //     loc, tensorType, iterArg, offsetsOutputSlice, sizesOutputSlice,
  //     stridesOutputSlice);
  Value outputSlice =
      rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(), elementType);

  IntegerAttr outputTileSizeI64Attr =
      rewriter.getI64IntegerAttr(inputOp.getOutputTileSize());
  IntegerAttr kernelSizeI64Attr =
      rewriter.getI64IntegerAttr(inputOp.getKernelSize());
  DenseI64ArrayAttr imageDimensionsDenseI64ArrayAttr =
      rewriter.getDenseI64ArrayAttr(inputOp.imageDimensions());
  tiledWinogradInputTransformOp = rewriter.create<WinogradInputTransformOp>(
      loc, tensorType, dynamicSlice, outputSlice, outputTileSizeI64Attr,
      kernelSizeI64Attr, imageDimensionsDenseI64ArrayAttr);

  // Insert results into output slice
  rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
  rewriter.create<tensor::ParallelInsertSliceOp>(
      loc, tiledWinogradInputTransformOp.getResult()[0], iterArg,
      offsetsOutputSlice, sizesOutputSlice, stridesOutputSlice);

  if (loopNest.loops.empty()) {
    inputOp.getResults()[0].replaceAllUsesWith(forallOp.getResults()[0]);
    return success();
  }

  // Replace returned value
  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          loopNest.loops.back().getBody()->getTerminator())) {
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, forallOp.getResult(0));
  }
  inputOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);

  return success();
}

/// Decompose tiled iree_linalg_ext.winograd.input_transform op.
/// TODO: Adopt decomposeOperation with this.
static LogicalResult decomposeTiledWinogradInputTransformOp(
    WinogradInputTransformOp tiledWinogradInputTransformOp,
    RewriterBase &rewriter) {
  Location loc = tiledWinogradInputTransformOp.getLoc();
  auto funcOp = tiledWinogradInputTransformOp
                    ->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp) {
    return rewriter.notifyMatchFailure(tiledWinogradInputTransformOp,
                                       "Could not find parent function");
  }
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  Value dynamicSlice = tiledWinogradInputTransformOp.input();
  Value outputSlice = tiledWinogradInputTransformOp.output();
  assert(tiledWinogradInputTransformOp.getInputOperandRank() == 2 &&
         "input operand expected to have rank-2");
  assert(tiledWinogradInputTransformOp.getOutputOperandRank() == 2 &&
         "output operand expected to have rank-2");
  auto one = rewriter.getIndexAttr(1);
  auto zero = rewriter.getIndexAttr(0);
  const int64_t inputTileSize =
      tiledWinogradInputTransformOp.getInputTileSize();
  const std::array<int64_t, 2> imageDims =
      tiledWinogradInputTransformOp.nhwcImageDimensions();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);
  Type elementType =
      tiledWinogradInputTransformOp.getOutputOperandType().getElementType();
  Value zeroF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(elementType));
  // Value scratch =
  //     rewriter.create<tensor::EmptyOp>(loc, inputTileSquare, elementType);
  const float *BT{nullptr};
  const float *B{nullptr};
  B = IREE::LinalgExt::Winograd::B_6x6_3x3;
  BT = IREE::LinalgExt::Winograd::BT_6x6_3x3;
  Value BTV = IREE::LinalgExt::createValueFrom2DConstant(
      BT, inputTileSize, inputTileSize, loc, rewriter);
  Value BV = IREE::LinalgExt::createValueFrom2DConstant(
      B, inputTileSize, inputTileSize, loc, rewriter);

  auto inputExtractSliceOp =
      dynamicSlice.getDefiningOp<tensor::ExtractSliceOp>();
  SmallVector<OpFoldResult> staticOffsets = inputExtractSliceOp.getOffsets();
  SmallVector<OpFoldResult> staticSizes = inputExtractSliceOp.getSizes();
  // Harcoding input rank as 4 here - since we'd be getting a tiled version with
  // rank 2. We are always expected to either have a rank 4 version of this op,
  // or rank 2 (tiled). And at this point in the flow, it is guaranteed to be a
  // rank 2 version of the op as ensured by the assertion above. Copy input
  // slice into zeroed padded scratch space
  SmallVector<OpFoldResult> offsets(2, zero);
  SmallVector<OpFoldResult> sizes(4, one);
  SmallVector<OpFoldResult> strides(2, one);
  unsigned staticSizeIndexCounter = 0;
  for (int i = 0; i < 4; i++) {
    if (!imageDimsSet.contains(i)) {
      continue;
    }
    sizes[i] = staticSizes[staticSizeIndexCounter++];
  }
  SmallVector<OpFoldResult> sliceOffsets;
  SmallVector<OpFoldResult> sliceSizes;
  const bool isNchw = tiledWinogradInputTransformOp.isNchw();
  if (isNchw) {
    permute<Permutation::NHWC_TO_NCHW>(sizes);
  }
  for (const int64_t dim : tiledWinogradInputTransformOp.imageDimensions()) {
    sliceSizes.push_back(sizes[dim]);
  }
  OpBuilder::InsertionGuard afterTiledWinogradInputTransformOp(rewriter);
  rewriter.setInsertionPointAfter(tiledWinogradInputTransformOp);
  // linalg::FillOp fillOp = rewriter.create<linalg::FillOp>(
  //     loc, ValueRange{zeroF32}, ValueRange{scratch});
  // Value inputSlice = rewriter.create<tensor::InsertSliceOp>(
  //     loc, dynamicSlice, fillOp.result(), offsets, sliceSizes, strides);
  auto inputSliceType = RankedTensorType::get(inputTileSquare, elementType);
  SmallVector<OpFoldResult> padLow(
      inputTileSquare.size(), rewriter.getZeroAttr(rewriter.getIndexType()));
  SmallVector<OpFoldResult> padHigh;
  auto dynamicSliceSizes = tensor::getMixedSizes(rewriter, loc, dynamicSlice);

  for (auto [idx, size] : llvm::enumerate(dynamicSliceSizes)) {
    AffineExpr d0;
    bindDims(rewriter.getContext(), d0);
    auto ub = rewriter.getAffineConstantExpr(inputTileSquare[idx]);
    AffineMap padHighMap =
        AffineMap::get(1, 0, {ub - d0}, rewriter.getContext());
    padHigh.push_back(rewriter.createOrFold<affine::AffineApplyOp>(
        loc, padHighMap,
        ValueRange{getValueOrCreateConstantIndexOp(rewriter, loc, size)}));
  }
  Value inputSlice = rewriter.create<tensor::PadOp>(
      loc, inputSliceType, dynamicSlice, padLow, padHigh, zeroF32);

  // Create computation
  Value result, AMatrix, BMatrix;
  linalg::MatmulOp matmulOp;
  Type tensorType = outputSlice.getType();
  for (int i = 0; i < 2; i++) {
    linalg::FillOp fillOp = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zeroF32}, ValueRange{outputSlice});
    if (i == 0) {
      AMatrix = inputSlice;
      BMatrix = BV;
    } else {
      AMatrix = BTV;
      BMatrix = result;
    }
    matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, tensorType, ValueRange{AMatrix, BMatrix}, fillOp.result());
    result = matmulOp.getResult(0);
  }
  tiledWinogradInputTransformOp.getResult()[0].replaceAllUsesWith(result);
  return success();
}

/// The input to WinogradInputTransformOp op is either (N, H, W, C) or (N, C,
/// H, W) but the output to this op is always (T, T, N, H', W', C). Since the
/// first two dimensions are used for the inner matrix multiplication, we
/// create the loop nest over (N, H', W', C).
LogicalResult
tileAndDecomposeWinogradInputTransformOp(WinogradInputTransformOp inputOp,
                                         RewriterBase &rewriter, bool onlyTile,
                                         bool useForall) {
  WinogradInputTransformOp tiledWinogradInputTransformOp;
  if (useForall) {
    if (failed(tileWinogradInputTransformOpWithForall(
            inputOp, rewriter, tiledWinogradInputTransformOp))) {
      return failure();
    }
  } else if (failed(tileWinogradInputTransformOp(
                 inputOp, rewriter, tiledWinogradInputTransformOp))) {
    return failure();
  }
  if (onlyTile) {
    return success();
  }
  return decomposeTiledWinogradInputTransformOp(tiledWinogradInputTransformOp,
                                                rewriter);
}

} // namespace

namespace {

/// Tile iree_linalg_ext.winograd.output_transform op.
/// TODO: Adopt getTiledImplementation with this.
static LogicalResult tileWinogradOutputTransformOp(
    WinogradOutputTransformOp outputOp, RewriterBase &rewriter,
    WinogradOutputTransformOp &tiledWinogradOutputTransformOp) {
  Location loc = outputOp.getLoc();
  auto funcOp = outputOp->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp) {
    return rewriter.notifyMatchFailure(outputOp,
                                       "Could not find parent function");
  }

  const int64_t inputTileSize = outputOp.getInputTileSize();
  const int64_t outputTileSize = outputOp.getOutputTileSize();
  switch (outputTileSize) {
  case 6:
    break;
  default:
    return failure();
  }

  Value input = outputOp.input();
  Value output = outputOp.output();
  auto outputType = output.getType().cast<ShapedType>();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  Type elementType = outputType.getElementType();
  const std::array<int64_t, 2> imageDims = outputOp.nhwcImageDimensions();
  const size_t numImageDims = imageDims.size();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);

  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  SmallVector<Value> lbs, ubs, steps;
  computeLoopParams(lbs, ubs, steps, input, numImageDims, loc, rewriter);
  // Construct loops
  rewriter.setInsertionPoint(outputOp);
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps, ValueRange({output}),
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
          ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });

  // Extract input slice
  rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
  auto one = rewriter.getIndexAttr(1);
  auto zero = rewriter.getIndexAttr(0);
  auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
  auto outputTileSizeAttr = rewriter.getIndexAttr(outputTileSize);
  SmallVector<OpFoldResult> strides(outputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> sizes(outputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> offsets(numImageDims, zero);
  sizes[0] = sizes[1] = inputTileSizeAttr;
  SmallVector<Value> ivs;
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }
  offsets.append(ivs.begin(), ivs.end());
  auto tensorType = RankedTensorType::get(inputTileSquare, elementType);
  tensor::ExtractSliceOp extractSliceOp =
      rewriter.create<tensor::ExtractSliceOp>(loc, tensorType, input, offsets,
                                              sizes, strides);
  Value inputSlice = extractSliceOp.getResult();

  // Extract output slice
  strides = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), one);
  offsets = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), zero);
  sizes = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), one);
  for (int i = 0; i < outputShape.size(); i++) {
    if (!imageDimsSet.contains(i)) {
      offsets[i] = ivs[i];
    } else {
      rewriter.setInsertionPointToStart(loopNest.loops[i].getBody());
      AffineExpr dim0;
      auto ot = rewriter.getAffineConstantExpr(outputTileSize);
      bindDims(rewriter.getContext(), dim0);
      AffineMap scaleMap =
          AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
      offsets[i] = rewriter.createOrFold<affine::AffineApplyOp>(
          loc, scaleMap, ValueRange{ivs[i]});
      sizes[i] = outputTileSizeAttr;
    }
  }
  rewriter.setInsertionPointAfter(extractSliceOp);
  tensorType = RankedTensorType::get(
      SmallVector<int64_t>(numImageDims, outputTileSize), elementType);
  Value iterArg = loopNest.loops.back().getRegionIterArg(0);
  if (outputOp.isNchw()) {
    permute<Permutation::NHWC_TO_NCHW>(offsets);
    permute<Permutation::NHWC_TO_NCHW>(sizes);
  }
  Value outputSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, tensorType, iterArg, offsets, sizes, strides);

  IntegerAttr outputTileSizeI64Attr =
      rewriter.getI64IntegerAttr(outputOp.getOutputTileSize());
  IntegerAttr kernelSizeI64Attr =
      rewriter.getI64IntegerAttr(outputOp.getKernelSize());
  DenseI64ArrayAttr imageDimensionsDenseI64ArrayAttr =
      rewriter.getDenseI64ArrayAttr(outputOp.imageDimensions());
  tiledWinogradOutputTransformOp = rewriter.create<WinogradOutputTransformOp>(
      loc, tensorType, inputSlice, outputSlice, outputTileSizeI64Attr,
      kernelSizeI64Attr, imageDimensionsDenseI64ArrayAttr);

  // Insert results into output slice
  Value updatedOutput = rewriter.create<tensor::InsertSliceOp>(
      loc, tiledWinogradOutputTransformOp.getResult()[0], iterArg, offsets,
      sizes, strides);

  // Replace returned value
  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          loopNest.loops.back().getBody()->getTerminator())) {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, updatedOutput);
  }
  outputOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);
  return success();
}

/// Tile iree_linalg_ext.winograd.output_transform op.
/// TODO: Adopt getTiledImplementation with this.
static LogicalResult tileWinogradOutputTransformOpWithForall(
    WinogradOutputTransformOp outputOp, RewriterBase &rewriter,
    WinogradOutputTransformOp &tiledWinogradOutputTransformOp) {
  Location loc = outputOp.getLoc();
  auto funcOp = outputOp->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp) {
    return rewriter.notifyMatchFailure(outputOp,
                                       "Could not find parent function");
  }
  auto workgroupSizes = llvm::map_to_vector(
      getEntryPoint(funcOp)->getWorkgroupSize().value(),
      [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });
  int64_t workgroupSize = computeProduct(workgroupSizes);

  const int64_t inputTileSize = outputOp.getInputTileSize();
  const int64_t outputTileSize = outputOp.getOutputTileSize();
  switch (outputTileSize) {
  case 6:
    break;
  default:
    return failure();
  }

  Value input = outputOp.input();
  Value output = outputOp.output();
  auto outputType = output.getType().cast<ShapedType>();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  Type elementType = outputType.getElementType();
  const std::array<int64_t, 2> imageDims = outputOp.nhwcImageDimensions();
  const size_t numImageDims = imageDims.size();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  SmallVector<int64_t> inputTileSquare(imageDims.size(), inputTileSize);

  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  SmallVector<Value> lbs, ubs, steps;
  computeLoopParams(lbs, ubs, steps, input, numImageDims, loc, rewriter);
  // Construct loops
  // SmallVector<Value> dest;
  // if (failed(tensor::getOrCreateDestinations(rewriter, loc, inputOp, dest)))
  //   return inputOp->emitOpError("failed to get destination tensors");
  auto getThreadMapping = [&](int64_t dim) {
    auto mappingIdInt = std::min<int64_t>(
        dim + static_cast<uint64_t>(gpu::MappingId::LinearDim0),
        gpu::getMaxEnumValForMappingId());
    return mlir::gpu::GPUThreadMappingAttr::get(
        outputOp->getContext(), gpu::symbolizeMappingId(mappingIdInt).value());
  };
  SmallVector<int64_t> forallInds, forInds;
  SmallVector<Value> forallUbs, forUbs;
  computeForallAndForUpperBounds(rewriter, loc, ubs, workgroupSize, forallUbs,
                                 forUbs, forallInds, forInds);
  SmallVector<Attribute> idDims;
  for (int i = 0; i < forallUbs.size(); i++) {
    idDims.push_back(getThreadMapping(i));
  }
  std::reverse(idDims.begin(), idDims.end());
  ArrayAttr mapping = rewriter.getArrayAttr(idDims);
  scf::LoopNest loopNest;
  auto loopNestSelect = [&](SmallVector<Value> vals) {
    return llvm::map_to_vector(forInds, [&](auto ind) { return vals[ind]; });
  };
  if (!forInds.empty()) {
    rewriter.setInsertionPoint(outputOp);
    loopNest = scf::buildLoopNest(
        rewriter, loc, loopNestSelect(lbs), forUbs, loopNestSelect(steps),
        ValueRange({output}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });
  }

  Location forallLoc =
      loopNest.loops.empty() ? loc : loopNest.loops.back()->getLoc();
  Value forallOut = loopNest.loops.empty()
                        ? output
                        : loopNest.loops.back().getRegionIterArg(0);
  if (!loopNest.loops.empty()) {
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
  } else {
    rewriter.setInsertionPoint(outputOp);
  }
  scf::ForallOp forallOp = rewriter.create<scf::ForallOp>(
      forallLoc, getAsOpFoldResult((forallUbs)), forallOut, mapping);
  Value iterArg = forallOp.getRegionOutArgs()[0];
  SmallVector<Value> forAllIvs = forallOp.getInductionVars();
  int64_t forIdx = 0, forallIdx = 0;
  SmallVector<Value> ivs;
  SetVector<int64_t> forIndsSet(forInds.begin(), forInds.end());
  SetVector<int64_t> forallIndsSet(forallInds.begin(), forallInds.end());
  for (int64_t idx = 0; idx < ubs.size() - 1; ++idx) {
    if (forIndsSet.contains(idx)) {
      ivs.push_back(loopNest.loops[forIdx++].getInductionVar());
    } else {
      ivs.push_back(forAllIvs[forallIdx++]);
    }
  }
  if (forallUbs.size() + forUbs.size() > ubs.size()) {
    rewriter.setInsertionPointToStart(forallOp.getBody());
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0);
    bindSymbols(rewriter.getContext(), s1);
    AffineMap map = AffineMap::get(0, 2, {s0 * s1}, rewriter.getContext());
    Value iv = rewriter.createOrFold<affine::AffineApplyOp>(
        forallOp->getLoc(), map,
        ValueRange{loopNest.loops[forIdx++].getInductionVar(),
                   forAllIvs[forallIdx++]});
    ivs.push_back(iv);
  } else if (forIndsSet.contains(ubs.size() - 1)) {
    ivs.push_back(loopNest.loops[forIdx++].getInductionVar());
  } else {
    ivs.push_back(forAllIvs[forallIdx++]);
  }

  // Extract input slice
  rewriter.setInsertionPoint(forallOp.getBody()->getTerminator());
  auto one = rewriter.getIndexAttr(1);
  auto zero = rewriter.getIndexAttr(0);
  auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
  auto outputTileSizeAttr = rewriter.getIndexAttr(outputTileSize);
  SmallVector<OpFoldResult> strides(outputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> sizes(outputOp.getInputOperandRank(), one);
  SmallVector<OpFoldResult> offsets(numImageDims, zero);
  sizes[0] = sizes[1] = inputTileSizeAttr;
  offsets.append(ivs.begin(), ivs.end());
  auto tensorType = RankedTensorType::get(inputTileSquare, elementType);
  tensor::ExtractSliceOp extractSliceOp =
      rewriter.create<tensor::ExtractSliceOp>(loc, tensorType, input, offsets,
                                              sizes, strides);
  Value inputSlice = extractSliceOp.getResult();

  // Extract output slice
  strides = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), one);
  offsets = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), zero);
  sizes = SmallVector<OpFoldResult>(outputOp.getOutputOperandRank(), one);
  forIdx = 0;
  for (int i = 0; i < outputShape.size(); i++) {
    OpBuilder::InsertionGuard g(rewriter);
    if (!forallIndsSet.contains(i)) {
      rewriter.setInsertionPoint(
          loopNest.loops[forIdx++].getBody()->getTerminator());
    }
    if (!imageDimsSet.contains(i)) {
      offsets[i] = ivs[i];
    } else {
      AffineExpr dim0;
      auto ot = rewriter.getAffineConstantExpr(outputTileSize);
      bindDims(rewriter.getContext(), dim0);
      AffineMap scaleMap =
          AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
      offsets[i] = rewriter.createOrFold<affine::AffineApplyOp>(
          loc, scaleMap, ValueRange{ivs[i]});
      sizes[i] = outputTileSizeAttr;
    }
  }
  tensorType = RankedTensorType::get(
      SmallVector<int64_t>(numImageDims, outputTileSize), elementType);
  if (outputOp.isNchw()) {
    permute<Permutation::NHWC_TO_NCHW>(offsets);
    permute<Permutation::NHWC_TO_NCHW>(sizes);
  }
  Value outputSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, tensorType, iterArg, offsets, sizes, strides);

  IntegerAttr outputTileSizeI64Attr =
      rewriter.getI64IntegerAttr(outputOp.getOutputTileSize());
  IntegerAttr kernelSizeI64Attr =
      rewriter.getI64IntegerAttr(outputOp.getKernelSize());
  DenseI64ArrayAttr imageDimensionsDenseI64ArrayAttr =
      rewriter.getDenseI64ArrayAttr(outputOp.imageDimensions());
  tiledWinogradOutputTransformOp = rewriter.create<WinogradOutputTransformOp>(
      loc, tensorType, inputSlice, outputSlice, outputTileSizeI64Attr,
      kernelSizeI64Attr, imageDimensionsDenseI64ArrayAttr);

  rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
  rewriter.create<tensor::ParallelInsertSliceOp>(
      loc, tiledWinogradOutputTransformOp.getResult()[0], iterArg, offsets,
      sizes, strides);

  if (loopNest.loops.empty()) {
    outputOp.getResults()[0].replaceAllUsesWith(forallOp.getResults()[0]);
    return success();
  }

  // Replace returned value
  if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
          loopNest.loops.back().getBody()->getTerminator())) {
    OpBuilder::InsertionGuard yieldGuard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, forallOp.getResult(0));
  }
  outputOp.getResults()[0].replaceAllUsesWith(loopNest.results[0]);
  return success();
}

/// Decompose tiled iree_linalg_ext.winograd.output_transform op.
/// TODO: Adopt decomposeOperation with this.
static LogicalResult decomposeTiledWinogradOutputTransformOp(
    WinogradOutputTransformOp tiledWinogradOutputTransformOp,
    RewriterBase &rewriter) {
  Location loc = tiledWinogradOutputTransformOp.getLoc();
  auto funcOp = tiledWinogradOutputTransformOp
                    ->getParentOfType<mlir::FunctionOpInterface>();
  if (!funcOp) {
    return rewriter.notifyMatchFailure(tiledWinogradOutputTransformOp,
                                       "Could not find parent function");
  }
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  Value inputSlice = tiledWinogradOutputTransformOp.input();
  Value outputSlice = tiledWinogradOutputTransformOp.output();
  assert(tiledWinogradOutputTransformOp.getInputOperandRank() == 2 &&
         "input operand expected to have rank-2");
  assert(tiledWinogradOutputTransformOp.getOutputOperandRank() == 2 &&
         "output operand expected to have rank-2");
  ShapedType outputType = tiledWinogradOutputTransformOp.getOutputOperandType();
  Type elementType = outputType.getElementType();
  const float *AT{nullptr};
  const float *A{nullptr};
  A = IREE::LinalgExt::Winograd::A_6x6_3x3;
  AT = IREE::LinalgExt::Winograd::AT_6x6_3x3;
  const int64_t inputTileSize =
      tiledWinogradOutputTransformOp.getInputTileSize();
  const int64_t outputTileSize =
      tiledWinogradOutputTransformOp.getOutputTileSize();
  /// The two values below are the transpose(A) [ATV]
  /// and A [AV] constant matrices that convert the output
  /// tile from the Winograd domain to the original domain.
  Value ATV = IREE::LinalgExt::createValueFrom2DConstant(
      AT, outputTileSize, inputTileSize, loc, rewriter);
  Value AV = IREE::LinalgExt::createValueFrom2DConstant(
      A, inputTileSize, outputTileSize, loc, rewriter);
  Value zeroF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(elementType));
  SmallVector<int64_t> scratchShape = {inputTileSize, outputTileSize};
  Value scratch =
      rewriter.create<tensor::EmptyOp>(loc, scratchShape, elementType);
  // Create computation
  OpBuilder::InsertionGuard afterTiledWinogradOutputTransformOp(rewriter);
  rewriter.setInsertionPointAfter(tiledWinogradOutputTransformOp);
  Value result, AMatrix, BMatrix;
  linalg::MatmulOp matmulOp;
  linalg::FillOp fillOp;
  Value tmp;
  for (int i = 0; i < 2; i++) {
    tmp = i == 0 ? scratch : outputSlice;
    fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                             ValueRange{tmp});
    if (i == 0) {
      AMatrix = inputSlice;
      BMatrix = AV;
    } else {
      AMatrix = ATV;
      BMatrix = result;
    }
    matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, tmp.getType(), ValueRange{AMatrix, BMatrix}, fillOp.result());
    result = matmulOp.getResult(0);
  }
  tiledWinogradOutputTransformOp.getResult()[0].replaceAllUsesWith(result);
  return success();
}

/// The input to WinogradOutputTransformOp is always (T, T, N, H', W', C)
/// but the output is either (N, H, W, C) or (N, C, H, W).
LogicalResult
tileAndDecomposeWinogradOutputTransformOp(WinogradOutputTransformOp outputOp,
                                          RewriterBase &rewriter, bool onlyTile,
                                          bool useForall) {
  WinogradOutputTransformOp tiledWinogradOutputTransformOp;
  if (useForall) {
    if (failed(tileWinogradOutputTransformOpWithForall(
            outputOp, rewriter, tiledWinogradOutputTransformOp))) {
      return failure();
    }
  } else if (failed(tileWinogradOutputTransformOp(
                 outputOp, rewriter, tiledWinogradOutputTransformOp))) {
    return failure();
  }
  if (onlyTile) {
    return success();
  }
  return decomposeTiledWinogradOutputTransformOp(tiledWinogradOutputTransformOp,
                                                 rewriter);
}

} // namespace

namespace {
struct TileAndDecomposeWinogradTransformPass
    : public TileAndDecomposeWinogradTransformBase<
          TileAndDecomposeWinogradTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  TileAndDecomposeWinogradTransformPass() = default;
  TileAndDecomposeWinogradTransformPass(bool onlyTile, bool useForall) {
    this->onlyTile = onlyTile;
    this->useForall = useForall;
  }
  TileAndDecomposeWinogradTransformPass(
      const TileAndDecomposeWinogradTransformPass &pass) {
    onlyTile = pass.onlyTile;
    useForall = pass.useForall;
  }

  void runOnOperation() override;
};
} // namespace

LogicalResult reifyWinogradTransform(mlir::FunctionOpInterface funcOp,
                                     bool onlyTile, bool useForall) {
  IRRewriter rewriter(funcOp.getContext());
  LogicalResult resultOfTransformations = success();
  funcOp.walk([&](WinogradInputTransformOp inputOp) {
    if (failed(tileAndDecomposeWinogradInputTransformOp(inputOp, rewriter,
                                                        onlyTile, useForall))) {
      resultOfTransformations = failure();
    }
    return WalkResult::advance();
  });
  funcOp.walk([&](WinogradOutputTransformOp outputOp) {
    if (failed(tileAndDecomposeWinogradOutputTransformOp(
            outputOp, rewriter, onlyTile, useForall))) {
      resultOfTransformations = failure();
    }
    return WalkResult::advance();
  });
  return resultOfTransformations;
}

void TileAndDecomposeWinogradTransformPass::runOnOperation() {
  if (failed(reifyWinogradTransform(getOperation(), onlyTile, useForall))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDecomposeWinogradTransformPass(bool onlyTile, bool useForall) {
  return std::make_unique<TileAndDecomposeWinogradTransformPass>(onlyTile,
                                                                 useForall);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
