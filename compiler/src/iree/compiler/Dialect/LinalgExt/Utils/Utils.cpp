// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-linalgExt-utils"

namespace mlir::iree_compiler::IREE::LinalgExt {

static bool hasAllOneValues(ArrayRef<int64_t> attr) {
  return llvm::all_of(attr, [](int64_t element) { return element == 1; });
}

OpFoldResult computeProductUsingAffine(OpBuilder &builder, Location loc,
                                       ArrayRef<OpFoldResult> vals) {
  auto mulMap = AffineMap::get(
      2, 0, {builder.getAffineDimExpr(0) * builder.getAffineDimExpr(1)});
  OpFoldResult product = builder.getIndexAttr(1);
  for (OpFoldResult val : vals) {
    product = affine::makeComposedFoldedAffineApply(builder, loc, mulMap,
                                                    {product, val});
  }
  return product;
}

OpFoldResult addOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto addMap = AffineMap::get(2, 0, {d0 + d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, addMap, {a, b});
}

OpFoldResult subOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  return affine::makeComposedFoldedAffineApply(
      builder, loc, AffineMap::get(2, 0, {d0 - d1}, builder.getContext()),
      {a, b});
}

OpFoldResult mulOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto mulMap = AffineMap::get(2, 0, {d0 * d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, mulMap, {a, b});
}

OpFoldResult mulAddOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                        OpFoldResult b, OpFoldResult c) {
  AffineExpr d0, d1, d2;
  bindDims(builder.getContext(), d0, d1, d2);
  auto mulAddMap = AffineMap::get(3, 0, {d0 * d1 + d2});
  return affine::makeComposedFoldedAffineApply(builder, loc, mulAddMap,
                                               {a, b, c});
}

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return arith::ConstantIndexOp::create(builder, loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case([&](RankedTensorType t) -> Value {
        return builder.createOrFold<tensor::DimOp>(loc, v, dim);
      })
      .Case([&](MemRefType t) -> Value {
        return builder.createOrFold<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getIndexAttr(t.getDimSize(dim));
}

SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc,
                                  Value shapedTypeValue) {
  return llvm::map_to_vector(
      llvm::seq<int64_t>(0,
                         cast<ShapedType>(shapedTypeValue.getType()).getRank()),
      [&](int64_t dim) { return getDim(builder, loc, shapedTypeValue, dim); });
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<Range> slice) {
  return getSlice(b, loc, src,
                  llvm::map_to_vector(slice, [](Range x) { return x.offset; }),
                  llvm::map_to_vector(slice, [](Range x) { return x.size; }),
                  llvm::map_to_vector(slice, [](Range x) { return x.stride; }));
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
                    ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(src.getType())
      .Case([&](RankedTensorType t) -> Operation * {
        return tensor::ExtractSliceOp::create(b, loc, src, offsets, sizes,
                                              strides);
      })
      .Case([&](MemRefType type) -> Operation * {
        return memref::SubViewOp::create(b, loc, src, offsets, sizes, strides);
      })
      .Default([&](Type t) -> Operation * {
        assert(false && "invalid type");
        return nullptr;
      });
}

Value castValue(OpBuilder &b, Location loc, Value src, ShapedType type) {
  return TypeSwitch<Type, Value>(src.getType())
      .Case([&](RankedTensorType t) -> Value {
        assert(isa<RankedTensorType>(type) && "expected compatible type");
        return tensor::CastOp::create(b, loc, type, src)->getResult(0);
      })
      .Case([&](MemRefType type) -> Value {
        assert(isa<MemRefType>(type) && "expected compatible type");
        return memref::CastOp::create(b, loc, type, src)->getResult(0);
      })
      .Default([&](Type t) {
        assert(false && "invalid type");
        return nullptr;
      });
}

SmallVector<int64_t> computeInterchangeFromDimPos(ArrayRef<int64_t> dimsPos,
                                                  int64_t rank) {
  SmallVector<int64_t> interchangeVector;
  interchangeVector.reserve(dimsPos.size());
  // First map dims and their position. For example, dims_pos = [2, 0] will map
  // to:
  // [
  //  [ key: 2, value: 0]
  //  [ key: 0, value: 1]
  // ]
  // where key is the idx in dims_pos while value its position in dims_pos.
  DenseMap<int64_t, int64_t> dimsAndPosMapping;
  for (int64_t dimsIdx = 0, end = dimsPos.size(); dimsIdx < end; dimsIdx++) {
    dimsAndPosMapping[dimsPos[dimsIdx]] = dimsIdx;
  }

  // Scan the position in order and insert the value in the map
  // to compute the interchange vector.
  for (int64_t dimsIdx = 0; dimsIdx < rank; dimsIdx++) {
    if (dimsAndPosMapping.count(dimsIdx)) {
      interchangeVector.push_back(dimsAndPosMapping[dimsIdx]);
    }
  }
  return interchangeVector;
}

Value createValueFrom2DConstant(const float *val, int64_t rows, int64_t cols,
                                Location loc, RewriterBase &rewriter) {
  ArrayRef<float> vector(val, rows * cols);
  SmallVector<int64_t> shape{rows, cols};
  return arith::ConstantOp::create(
      rewriter, loc,
      DenseFPElementsAttr::get(
          RankedTensorType::get(shape, rewriter.getF32Type()), vector));
}

SmallVector<int64_t> asShapeWithAnyValueAsDynamic(ArrayRef<OpFoldResult> ofrs) {
  SmallVector<int64_t> result;
  for (auto o : ofrs) {
    // Have to do this first, as getConstantIntValue special-cases constants.
    if (dyn_cast<Value>(o)) {
      result.push_back(ShapedType::kDynamic);
    } else {
      result.push_back(getConstantIntValue(o).value_or(ShapedType::kDynamic));
    }
  }
  return result;
}

SmallVector<AffineExpr> getDimExprsForSymbols(MLIRContext *context,
                                              unsigned numDims,
                                              unsigned numSymbols) {
  return llvm::map_to_vector(
      llvm::seq<unsigned>(0, numSymbols), [&](unsigned symbolNumber) {
        return getAffineDimExpr(symbolNumber + numDims, context);
      });
}

AffineMap convertDimsToSymbols(AffineMap map, unsigned numDims,
                               unsigned numSymbols,
                               SmallVector<AffineExpr> &symbolReplacements) {
  return map.replaceDimsAndSymbols(/*dimReplacements=*/ArrayRef<AffineExpr>{},
                                   symbolReplacements, numDims + numSymbols, 0);
}
SmallVector<AffineMap>
convertDimsToSymbols(ArrayRef<AffineMap> maps, unsigned numDims,
                     unsigned numSymbols,
                     SmallVector<AffineExpr> &symbolReplacements) {
  return llvm::map_to_vector(maps, [&](AffineMap map) {
    return convertDimsToSymbols(map, numDims, numSymbols, symbolReplacements);
  });
}
SmallVector<AffineMap> convertDimsToSymbols(MLIRContext *context,
                                            ArrayRef<AffineMap> maps,
                                            unsigned numDims,
                                            unsigned numSymbols) {
  auto symbolReplacements = getDimExprsForSymbols(context, numDims, numSymbols);
  return convertDimsToSymbols(maps, numDims, numSymbols, symbolReplacements);
}

//===---------------------------------------------------------------------===//
// Classification of ops that change bit-widths
//===---------------------------------------------------------------------===//

enum class BitWidthChangeInfo {
  kNull,
  kExtend,
  kTruncate,
};

static BitWidthChangeInfo isBitExtendOrTruncateOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return BitWidthChangeInfo::kNull;
  }

  if (genericOp.getNumDpsInits() != 1) {
    return BitWidthChangeInfo::kNull;
  }

  // Check that the all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return BitWidthChangeInfo::kNull;
  }

  // Check that all operands that have the highest rank have bit width
  // less than the output bit-width.
  DenseMap<int64_t, SmallVector<OpOperand *>> rankBuckets;
  int64_t maxOperandRank = 0;
  for (OpOperand *input : genericOp.getDpsInputOperands()) {
    auto inputType = dyn_cast<RankedTensorType>(input->get().getType());
    if (!inputType) {
      continue;
    }
    int64_t currRank = inputType.getRank();
    maxOperandRank = std::max(currRank, maxOperandRank);
    rankBuckets[currRank].push_back(input);
  }
  if (maxOperandRank == 0 || rankBuckets[maxOperandRank].empty()) {
    return BitWidthChangeInfo::kNull;
  }

  unsigned int maxInputElementBitWidth = 0;
  OpOperand *inputOperand;
  for (OpOperand *operand : rankBuckets[maxOperandRank]) {
    RankedTensorType tensorType =
        cast<RankedTensorType>(operand->get().getType());
    Type elementType = tensorType.getElementType();
    if (!elementType.isIntOrFloat()) {
      return BitWidthChangeInfo::kNull;
    }
    unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
    if (elementBitWidth > maxInputElementBitWidth) {
      maxInputElementBitWidth = elementBitWidth;
      inputOperand = operand;
    }
  }
  if (!inputOperand) {
    return BitWidthChangeInfo::kNull;
  }
  Type inputElementType =
      cast<RankedTensorType>(inputOperand->get().getType()).getElementType();

  // Check that the identity input element bitwidth is smaller than the output
  // element bitwidth.
  RankedTensorType outputType =
      dyn_cast<RankedTensorType>(genericOp->getResultTypes()[0]);
  if (!outputType) {
    return BitWidthChangeInfo::kNull;
  }
  Type outputElementType = outputType.getElementType();
  if (!outputElementType.isIntOrFloat()) {
    return BitWidthChangeInfo::kNull;
  }

  unsigned inputBitWidth = inputElementType.getIntOrFloatBitWidth();
  unsigned outputBitWidth = outputElementType.getIntOrFloatBitWidth();

  // Checks specific to bit extend operations.
  if (inputBitWidth < outputBitWidth) {
    // Since these are cloned into dispatches, avoid expensive operations.
    for (Operation &op : *genericOp.getBody()) {
      if (op.getDialect() == op.getContext()->getLoadedDialect("math")) {
        return BitWidthChangeInfo::kNull;
      }
    }
    return BitWidthChangeInfo::kExtend;
  }

  // Checks specific to bit truncate operations.
  if (outputBitWidth < inputBitWidth) {
    // For now enforce that the input and output ranks match for truncates.
    if (maxOperandRank != outputType.getRank()) {
      return BitWidthChangeInfo::kNull;
    }
    return BitWidthChangeInfo::kTruncate;
  }

  return BitWidthChangeInfo::kNull;
}

bool isBitExtendOp(Operation *op) {
  return isBitExtendOrTruncateOp(op) == BitWidthChangeInfo::kExtend;
}

bool isBitTruncateOp(Operation *op) {
  return isBitExtendOrTruncateOp(op) == BitWidthChangeInfo::kTruncate;
}

//===---------------------------------------------------------------------===//
// Classification of other ops
//===---------------------------------------------------------------------===//

bool isBroadcastingOp(linalg::LinalgOp op) {
  if (isa<linalg::BroadcastOp>(op)) {
    return true;
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp) {
    return false;
  }

  // Only allow a single input and init.
  if (genericOp.getNumDpsInits() != 1 || genericOp.getNumDpsInputs() != 1) {
    return false;
  }

  // Check that the all loops are parallel.
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return false;
  }

  // Check that indexing maps are broadcasting.
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  auto inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  auto outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (inMap.getNumResults() >= outMap.getNumResults()) {
    return false;
  }
  if (!inMap.isProjectedPermutation() || !outMap.isIdentity()) {
    return false;
  }
  return llvm::hasSingleElement(op.getBlock()->getOperations());
}

bool isGatherlikeOp(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return false;
  }

  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }

  auto &region = genericOp->getRegion(0);
  if (!llvm::hasSingleElement(region)) {
    return false;
  }

  // `yieldOp` should yield a single value from a `tensor.extract`
  auto yieldOp = cast<linalg::YieldOp>(region.front().getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return false;
  }

  llvm::SetVector<Operation *> sliceOps;
  BackwardSliceOptions options;
  bool hasTensorExtract = false;
  options.filter = [&](Operation *currOp) {
    if (isa<tensor::ExtractOp>(currOp)) {
      // Exclude it from the slice, but mark the boolean above to `true` to
      // annotate that a `tensor.extract` was hit. Stop the traversal to avoid
      // unnecessary traversal.
      hasTensorExtract = true;
      return false;
    }
    return currOp->getBlock() == genericOp.getBody();
  };
  [[maybe_unused]] LogicalResult result =
      getBackwardSlice(yieldOp.getOperand(0), &sliceOps, options);
  assert(result.succeeded());
  return hasTensorExtract;
}

//===---------------------------------------------------------------------===//
// IGEMM details for generic convolutions
//===---------------------------------------------------------------------===//

/// Classification of IGEMM dims from the image/im2col perspective.
/// Batch includes both batch and depth (group) dims. M is output spatial
/// dims. InputChannel and FilterLoop are the two subcategories of reduction
/// dims. The canonical im2col output order is:
///   [Batch, M, InputChannel, FilterLoop].
enum class Im2colDimKind {
  Batch,
  M,
  InputChannel,
  FilterLoop,
};

/// Computes the output permutation for the im2col tensor to match the
/// dimension order of the input tensor.
static SmallVector<int64_t> computeIm2colOutputPermutation(
    AffineMap inputMapGEMM, const linalg::ConvolutionDimensions &convDims,
    const DenseMap<int64_t, AffineExpr> &convToIgemmDimMap) {
  llvm::SmallDenseSet<unsigned, 4> batchDimSet(convDims.batch.begin(),
                                               convDims.batch.end());
  llvm::SmallDenseSet<unsigned, 4> depthDimSet(convDims.depth.begin(),
                                               convDims.depth.end());
  llvm::SmallDenseSet<unsigned, 4> outputImageDimSet(
      convDims.outputImage.begin(), convDims.outputImage.end());
  llvm::SmallDenseSet<unsigned, 4> inputChannelDimSet(
      convDims.inputChannel.begin(), convDims.inputChannel.end());
  llvm::SmallDenseSet<unsigned, 4> filterLoopDimSet(
      convDims.filterLoop.begin(), convDims.filterLoop.end());

  DenseMap<int64_t, SmallVector<int64_t>> igemmDimToConvDims;
  for (const auto &[convDim, igemmExpr] : convToIgemmDimMap) {
    int64_t igemmDim = cast<AffineDimExpr>(igemmExpr).getPosition();
    igemmDimToConvDims[igemmDim].push_back(convDim);
  }

  auto classifyIgemmDim =
      [&](int64_t igemmDim) -> Im2colDimKind {
    ArrayRef<int64_t> convDimGroup = igemmDimToConvDims[igemmDim];
    if (llvm::all_of(convDimGroup, [&](int64_t convDim) {
          return batchDimSet.contains(convDim) || depthDimSet.contains(convDim);
        })) {
      return Im2colDimKind::Batch;
    }
    if (llvm::all_of(convDimGroup, [&](int64_t convDim) {
          return outputImageDimSet.contains(convDim);
        })) {
      return Im2colDimKind::M;
    }
    if (llvm::all_of(convDimGroup, [&](int64_t convDim) {
          return inputChannelDimSet.contains(convDim);
        })) {
      return Im2colDimKind::InputChannel;
    }
    if (llvm::all_of(convDimGroup, [&](int64_t convDim) {
          return filterLoopDimSet.contains(convDim);
        })) {
      return Im2colDimKind::FilterLoop;
    }
    // Mixed inputChannel + filterLoop after collapse: classify as FilterLoop
    // since inputChannel dims are absorbed into the filter loop group.
    assert(llvm::all_of(convDimGroup, [&](int64_t convDim) {
          return inputChannelDimSet.contains(convDim) ||
                 filterLoopDimSet.contains(convDim);
        }) && "unexpected IGEMM input dim classification");
    return Im2colDimKind::FilterLoop;
  };

  SmallVector<int64_t> batchPositions;
  SmallVector<int64_t> mPositions;
  SmallVector<int64_t> inputChannelPositions;
  SmallVector<int64_t> filterLoopPositions;
  for (auto [actualPos, expr] : llvm::enumerate(inputMapGEMM.getResults())) {
    int64_t igemmDim = cast<AffineDimExpr>(expr).getPosition();
    switch (classifyIgemmDim(igemmDim)) {
    case Im2colDimKind::Batch:
      batchPositions.push_back(actualPos);
      break;
    case Im2colDimKind::M:
      mPositions.push_back(actualPos);
      break;
    case Im2colDimKind::InputChannel:
      inputChannelPositions.push_back(actualPos);
      break;
    case Im2colDimKind::FilterLoop:
      filterLoopPositions.push_back(actualPos);
      break;
    }
  }

  SmallVector<int64_t> outputPerm(inputMapGEMM.getNumResults(), -1);
  int64_t canonicalPos = 0;
  auto assignPositions = [&](ArrayRef<int64_t> actualPositions) {
    for (int64_t actualPos : actualPositions) {
      outputPerm[actualPos] = canonicalPos++;
    }
  };
  assignPositions(batchPositions);
  assignPositions(mPositions);
  assignPositions(inputChannelPositions);
  assignPositions(filterLoopPositions);
  return outputPerm;
}

/// Remaps the result expressions of an affine map by substituting each
/// AffineDimExpr with the corresponding expression from 'dims'.
static SmallVector<AffineExpr> remapMapResults(AffineMap map,
                                               ArrayRef<AffineExpr> dims) {
  SmallVector<AffineExpr> results;
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    results.push_back(dims[cast<AffineDimExpr>(expr).getPosition()]);
  }
  return results;
}

/// Inserts filter loop dims into the input dim order adjacent to their
/// corresponding input channel dims. Each filter loop dim is placed before
/// the next input channel dim, or after the previous one if no next channel
/// dim exists. When no input channel dims exist at all, filter loop dims are
/// grouped after the last spatial dim.
static void insertFilterLoopDims(
    SmallVectorImpl<unsigned> &inputDimOrder,
    const linalg::ConvolutionDimensions &convDims,
    const llvm::SmallDenseSet<unsigned, 4> &inputChannelDimSet,
    const llvm::SmallDenseSet<unsigned, 4> &filterLoopDimSet) {
  llvm::SmallDenseSet<unsigned, 4> outputImageDimSet(
      convDims.outputImage.begin(), convDims.outputImage.end());
  auto findDimIndex = [&](ArrayRef<unsigned> dims, unsigned dim) -> int64_t {
    auto it = llvm::find(dims, dim);
    return it == dims.end() ? -1 : std::distance(dims.begin(), it);
  };

  auto findNextInputChannelIndex = [&](ArrayRef<unsigned> dims,
                                       int64_t start) -> int64_t {
    for (int64_t idx = start; idx < static_cast<int64_t>(dims.size()); ++idx) {
      if (inputChannelDimSet.contains(dims[idx])) {
        return idx;
      }
    }
    return -1;
  };

  auto findPrevInputChannelIndex = [&](ArrayRef<unsigned> dims,
                                       int64_t start) -> int64_t {
    for (int64_t idx = start; idx >= 0; --idx) {
      if (inputChannelDimSet.contains(dims[idx])) {
        return idx;
      }
    }
    return -1;
  };

  auto findInsertIndexAfterChannel = [&](ArrayRef<unsigned> dims,
                                         int64_t channelIndex) -> int64_t {
    int64_t insertIndex = channelIndex + 1;
    while (insertIndex < static_cast<int64_t>(dims.size()) &&
           filterLoopDimSet.contains(dims[insertIndex])) {
      ++insertIndex;
    }
    return insertIndex;
  };

  for (auto [outputImageDim, filterLoopDim] :
       llvm::zip_equal(convDims.outputImage, convDims.filterLoop)) {
    int64_t outputImageIndex = findDimIndex(inputDimOrder, outputImageDim);
    assert(outputImageIndex >= 0 && "expected output image dim in input order");

    int64_t nextChannelIndex =
        findNextInputChannelIndex(inputDimOrder, outputImageIndex + 1);
    if (nextChannelIndex >= 0) {
      inputDimOrder.insert(inputDimOrder.begin() + nextChannelIndex,
                           filterLoopDim);
      continue;
    }

    int64_t prevChannelIndex =
        findPrevInputChannelIndex(inputDimOrder, outputImageIndex);
    if (prevChannelIndex >= 0) {
      int64_t insertIndex =
          findInsertIndexAfterChannel(inputDimOrder, prevChannelIndex);
      inputDimOrder.insert(inputDimOrder.begin() + insertIndex, filterLoopDim);
      continue;
    }

    // No input channel dims found in either direction. Insert after the last
    // spatial/filter-loop dim so all filter loop dims are grouped together
    // at the end of the spatial region (enabling collapse).
    int64_t lastSpatialIndex = -1;
    for (int64_t idx = 0; idx < static_cast<int64_t>(inputDimOrder.size());
         ++idx) {
      if (outputImageDimSet.contains(inputDimOrder[idx]) ||
          filterLoopDimSet.contains(inputDimOrder[idx])) {
        lastSpatialIndex = idx;
      }
    }
    int64_t insertIndex = lastSpatialIndex + 1;
    inputDimOrder.insert(inputDimOrder.begin() + insertIndex,
                         filterLoopDim);
  }
}

/// Builds the result expressions for the image-side GEMM map in expanded
/// (uncollapsed) form. Walks the original input map, replaces convolved
/// spatial expressions with their spatial dim, skips filter loop dims
/// (inserted separately by insertFilterLoopDims), and maps each dim through
/// convToIgemmDimMap.
static SmallVector<AffineExpr> buildExpandedInputGEMMResults(
    AffineMap inputMap, const linalg::ConvolutionDimensions &convDims,
    const DenseMap<int64_t, AffineExpr> &convToIgemmDimMap) {
  llvm::SmallDenseSet<unsigned, 4> inputChannelDimSet(
      convDims.inputChannel.begin(), convDims.inputChannel.end());
  llvm::SmallDenseSet<unsigned, 4> filterLoopDimSet(
      convDims.filterLoop.begin(), convDims.filterLoop.end());

  // Build the base dim list in the order implied by the original input map.
  // For convolved spatial expressions like `ow + kw`, keep only the spatial
  // dim in the base list and insert the filter-loop dims separately below.
  llvm::SetVector<unsigned> seenDims;
  SmallVector<unsigned> inputDimOrder;
  for (AffineExpr inputExpr : inputMap.getResults()) {
    auto spatialIt = llvm::find_if(convDims.outputImage, [&](unsigned dim) {
      return inputExpr.isFunctionOfDim(dim);
    });
    if (spatialIt != convDims.outputImage.end()) {
      if (seenDims.insert(*spatialIt)) {
        inputDimOrder.push_back(*spatialIt);
      }
      continue;
    }

    unsigned dim = cast<AffineDimExpr>(inputExpr).getPosition();
    if (filterLoopDimSet.contains(dim) || !seenDims.insert(dim)) {
      continue;
    }
    inputDimOrder.push_back(dim);
  }
  insertFilterLoopDims(inputDimOrder, convDims, inputChannelDimSet,
                       filterLoopDimSet);

  SmallVector<AffineExpr> inputDims;
  inputDims.reserve(inputDimOrder.size());
  for (unsigned dim : inputDimOrder) {
    inputDims.push_back(convToIgemmDimMap.at(dim));
  }
  return inputDims;
}

// Permutes reduction dims in the loop space so their order matches the
// image-side map. Since reduction dims are always at the end of the source
// dims, this is just a permutation among the reduction positions.
static IGEMMGenericConvDetails canonicalizeReductionOrder(
    IGEMMGenericConvDetails details) {
  int64_t inputMapIndex = details.isOutputChannelFirst ? 1 : 0;
  AffineMap inputMapGEMM = details.igemmContractionMaps[inputMapIndex];
  int64_t rank = inputMapGEMM.getNumDims();
  MLIRContext *ctx = inputMapGEMM.getContext();

  // Collect reduction positions in loop order (ascending).
  SmallVector<int64_t> reductionPositions;
  llvm::SmallDenseSet<int64_t, 4> reductionDimSet;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (details.igemmLoopIterators[dim] == utils::IteratorType::reduction) {
      reductionPositions.push_back(dim);
      reductionDimSet.insert(dim);
    }
  }

  // Walk the image-side map to get reduction dims in image-side order.
  SmallVector<int64_t> imageReductionOrder;
  for (AffineExpr expr : inputMapGEMM.getResults()) {
    int64_t dim = cast<AffineDimExpr>(expr).getPosition();
    if (reductionDimSet.contains(dim)) {
      imageReductionOrder.push_back(dim);
    }
  }

  // Build remapping: imageReductionOrder[i] -> reductionPositions[i].
  // This places reduction dims in the image-side order while keeping them
  // in the same tail positions of the loop space.
  DenseMap<int64_t, int64_t> reductionRemap;
  bool isIdentity = true;
  for (auto [i, imageRedDim] : llvm::enumerate(imageReductionOrder)) {
    reductionRemap[imageRedDim] = reductionPositions[i];
    if (imageRedDim != reductionPositions[i]) {
      isIdentity = false;
    }
  }

  // Remap a dim expr, only touching reduction dims.
  auto remapExpr = [&](AffineExpr expr) -> AffineExpr {
    int64_t dim = cast<AffineDimExpr>(expr).getPosition();
    auto it = reductionRemap.find(dim);
    if (it != reductionRemap.end()) {
      return getAffineDimExpr(it->second, ctx);
    }
    return expr;
  };

  if (!isIdentity) {
    // Apply remapping to all indexing maps.
    for (AffineMap &map : details.igemmContractionMaps) {
      SmallVector<AffineExpr> newResults;
      for (AffineExpr expr : map.getResults()) {
        newResults.push_back(remapExpr(expr));
      }
      map = AffineMap::get(rank, map.getNumSymbols(), newResults, ctx);
    }
    // Permute loop bounds and iterators at the reduction positions.
    SmallVector<int64_t> newBounds = details.igemmLoopBounds;
    SmallVector<utils::IteratorType> newIterators = details.igemmLoopIterators;
    for (auto [i, imageRedDim] : llvm::enumerate(imageReductionOrder)) {
      newBounds[reductionPositions[i]] = details.igemmLoopBounds[imageRedDim];
      newIterators[reductionPositions[i]] =
          details.igemmLoopIterators[imageRedDim];
    }
    details.igemmLoopBounds = newBounds;
    details.igemmLoopIterators = newIterators;
    // Update convToIgemmDimMap.
    for (auto &[convDim, igemmExpr] : details.convToIgemmDimMap) {
      igemmExpr = remapExpr(igemmExpr);
    }
  }

  return details;
}

/// Translates iteration-space reassociation indices into operand-space
/// reassociation indices using the given affine map. Each iteration group
/// is mapped to the corresponding operand dim positions.
static SmallVector<ReassociationIndices> getOperandReassociation(
    AffineMap map, ArrayRef<ReassociationIndices> iterationReassociation) {
  DenseMap<int64_t, int64_t> iterationToOperandDim;
  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    iterationToOperandDim[cast<AffineDimExpr>(expr).getPosition()] = idx;
  }

  SmallVector<ReassociationIndices> operandReassociation;
  for (ReassociationIndicesRef group : iterationReassociation) {
    ReassociationIndices operandGroup;
    for (int64_t iterationDim : group) {
      auto it = iterationToOperandDim.find(iterationDim);
      if (it != iterationToOperandDim.end()) {
        operandGroup.push_back(it->second);
      }
    }
    if (operandGroup.empty()) {
      continue;
    }
    assert(llvm::is_sorted(operandGroup) &&
           "expected operand dims to preserve iteration order");
    assert((operandGroup.size() <= 1 ||
            operandGroup.back() - operandGroup.front() + 1 ==
                static_cast<int64_t>(operandGroup.size())) &&
           "expected operand dims to be contiguous");
    operandReassociation.push_back(std::move(operandGroup));
  }
  llvm::sort(operandReassociation, [](ReassociationIndicesRef lhs,
                                      ReassociationIndicesRef rhs) {
    return lhs.front() < rhs.front();
  });
  return operandReassociation;
}

/// Collapses loop bounds by multiplying together bounds in each group.
static SmallVector<int64_t> collapseLoopBounds(
    ArrayRef<int64_t> loopBounds,
    ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<int64_t> collapsedLoopBounds;
  collapsedLoopBounds.reserve(reassociation.size());
  for (ReassociationIndicesRef group : reassociation) {
    int64_t collapsedSize = 1;
    for (int64_t dim : group) {
      collapsedSize *= loopBounds[dim];
    }
    collapsedLoopBounds.push_back(collapsedSize);
  }
  return collapsedLoopBounds;
}

/// Collapses iterator types by taking the type of the first dim in each group.
static SmallVector<utils::IteratorType> collapseIteratorTypes(
    ArrayRef<utils::IteratorType> iteratorTypes,
    ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<utils::IteratorType> collapsedIteratorTypes;
  collapsedIteratorTypes.reserve(reassociation.size());
  for (ReassociationIndicesRef group : reassociation) {
    collapsedIteratorTypes.push_back(iteratorTypes[group.front()]);
  }
  return collapsedIteratorTypes;
}

/// Collapses iteration dimensions in a set of affine maps according to the
/// given reassociation indices. Adjacent iteration dims that map to the same
/// collapsed dim produce a single result dim in the output map.
static SmallVector<AffineMap> collapseAffineMaps(
    ArrayRef<AffineMap> maps, ArrayRef<ReassociationIndices> reassociation) {
  assert(!maps.empty() && "expected non-empty maps");
  int64_t numDims = maps.front().getNumDims();
  SmallVector<int64_t> iterationToCollapsedDim(numDims, -1);
  for (auto [collapsedDim, group] : llvm::enumerate(reassociation)) {
    for (int64_t dim : group) {
      iterationToCollapsedDim[dim] = collapsedDim;
    }
  }

  SmallVector<AffineMap> collapsedMaps;
  collapsedMaps.reserve(maps.size());
  for (AffineMap map : maps) {
    SmallVector<AffineExpr> collapsedResults;
    for (AffineExpr expr : map.getResults()) {
      int64_t collapsedDim =
          iterationToCollapsedDim[cast<AffineDimExpr>(expr).getPosition()];
      AffineExpr collapsedExpr =
          getAffineDimExpr(collapsedDim, map.getContext());
      if (collapsedResults.empty() ||
          collapsedResults.back() != collapsedExpr) {
        collapsedResults.push_back(collapsedExpr);
      }
    }
    collapsedMaps.push_back(AffineMap::get(reassociation.size(),
                                           map.getNumSymbols(),
                                           collapsedResults, map.getContext()));
  }
  return collapsedMaps;
}

/// Updates convToIgemmDimMap to refer to collapsed iteration dims.
static DenseMap<int64_t, AffineExpr> collapseConvToIgemmDimMap(
    const DenseMap<int64_t, AffineExpr> &convToIgemmDimMap,
    ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<int64_t> iterationToCollapsedDim;
  for (auto [collapsedDim, group] : llvm::enumerate(reassociation)) {
    iterationToCollapsedDim.resize(
        std::max<int64_t>(iterationToCollapsedDim.size(), group.back() + 1), -1);
    for (int64_t dim : group) {
      iterationToCollapsedDim[dim] = collapsedDim;
    }
  }

  DenseMap<int64_t, AffineExpr> collapsedMap;
  MLIRContext *ctx = convToIgemmDimMap.begin()->second.getContext();
  for (const auto &[convDim, igemmExpr] : convToIgemmDimMap) {
    int64_t expandedDim = cast<AffineDimExpr>(igemmExpr).getPosition();
    collapsedMap[convDim] =
        getAffineDimExpr(iterationToCollapsedDim[expandedDim], ctx);
  }
  return collapsedMap;
}

/// Identifies groups of adjacent reduction dims that can be collapsed.
/// Two adjacent reduction dims can be collapsed if they appear as a
/// preserved sequence in all indexing maps (checked via
/// areDimSequencesPreserved).
static SmallVector<ReassociationIndices> getCollapsibleIGEMMIterationGroups(
    ArrayRef<AffineMap> maps, ArrayRef<utils::IteratorType> iteratorTypes) {
  SmallVector<ReassociationIndices> reassociation;
  int64_t rank = iteratorTypes.size();
  for (int64_t dim = 0; dim < rank;) {
    ReassociationIndices group = {dim};
    while (iteratorTypes[dim] == utils::IteratorType::reduction &&
           group.back() + 1 < rank &&
           iteratorTypes[group.back() + 1] == utils::IteratorType::reduction) {
      ReassociationIndices candidate = group;
      candidate.push_back(group.back() + 1);
      if (!linalg::areDimSequencesPreserved(maps, {candidate})) {
        break;
      }
      group.push_back(group.back() + 1);
    }
    reassociation.push_back(std::move(group));
    dim = reassociation.back().back() + 1;
  }
  return reassociation;
}

/// Builds the fully expanded (uncollapsed) IGEMM details for a convolution.
/// The expanded form has an identity mapping from conv loop dims to IGEMM
/// loop dims. Reduction dims are then canonicalized to match the image-side
/// order to enable subsequent collapsing.
static FailureOr<IGEMMGenericConvDetails>
getExpandedIGEMMGenericConvDetails(linalg::LinalgOp linalgOp) {
  auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
  MLIRContext *ctx = linalgOp->getContext();
  if (failed(convDimsOrFailure)) {
    return failure();
  }
  const mlir::linalg::ConvolutionDimensions &convDims = *convDimsOrFailure;
  LLVM_DEBUG({
    llvm::dbgs() << "conv: " << linalgOp;
    llvm::dbgs() << "\nconv batch dim: ";
    llvm::interleaveComma(convDims.batch, llvm::dbgs());
    llvm::dbgs() << "\nconv output window dims: ";
    llvm::interleaveComma(convDims.outputImage, llvm::dbgs());
    llvm::dbgs() << "\nconv output channel dim: ";
    llvm::interleaveComma(convDims.outputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv filter window dims: ";
    llvm::interleaveComma(convDims.filterLoop, llvm::dbgs());
    llvm::dbgs() << "\nconv input channel dims: ";
    llvm::interleaveComma(convDims.inputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv depth dims: ";
    llvm::interleaveComma(convDims.depth, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  Value input = linalgOp.getDpsInputs()[0];
  Value filter = linalgOp.getDpsInputs()[1];
  Value output = linalgOp.getDpsInits()[0];
  auto inputType = cast<ShapedType>(input.getType());
  auto filterType = cast<ShapedType>(filter.getType());

  if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
    LDBG() << "[unimplemented] expected 'filterType' and 'inputType' to have "
              "static shapes.";
    return failure();
  }

  // TODO: Support pooling operations.
  if (convDims.outputChannel.empty()) {
    LDBG() << "[unimplemented] expected no pooling operations.";
    return failure();
  }

  auto indexingMaps = linalgOp.getIndexingMapsArray();
  auto inputMap = indexingMaps[0];
  auto filterMap = indexingMaps[1];
  auto outputMap = indexingMaps[2];

  bool isOutputChannelFirst = false;
  auto outputChannelPos = convDims.outputChannel;
  auto outputImagePos = convDims.outputImage;

  std::optional<int64_t> outputChannelLastDim = outputMap.getResultPosition(
      getAffineDimExpr(outputChannelPos.back(), outputMap.getContext()));
  std::optional<int64_t> outputImageFirstDim = outputMap.getResultPosition(
      getAffineDimExpr(outputImagePos[0], outputMap.getContext()));
  if (!outputImageFirstDim || !outputChannelLastDim) {
    LDBG() << "output image or output channel dim not found in output.";
    return failure();
  }
  if (outputChannelLastDim.value() < outputImageFirstDim.value()) {
    isOutputChannelFirst = true;
  }

  int64_t loopRank = linalgOp.getNumLoops();
  SmallVector<AffineExpr> dims(loopRank);
  bindDimsList<AffineExpr>(ctx, dims);

  DenseMap<int64_t, AffineExpr> convToIgemmDimMap;
  for (int64_t dim = 0; dim < loopRank; ++dim) {
    convToIgemmDimMap[dim] = dims[dim];
  }

  SmallVector<ReassociationIndices> filterReassocIndices;
  filterReassocIndices.reserve(filterMap.getNumResults());
  for (int64_t idx = 0; idx < filterMap.getNumResults(); ++idx) {
    filterReassocIndices.push_back({idx});
  }

  auto inputMapGEMM = AffineMap::get(
      loopRank, 0,
      buildExpandedInputGEMMResults(inputMap, convDims, convToIgemmDimMap),
      ctx);
  auto filterMapGEMM =
      AffineMap::get(loopRank, 0, remapMapResults(filterMap, dims), ctx);
  auto resultMap =
      AffineMap::get(loopRank, 0, remapMapResults(outputMap, dims), ctx);

  SmallVector<AffineMap> indexingGEMMMaps;
  if (isOutputChannelFirst) {
    indexingGEMMMaps.push_back(filterMapGEMM);
    indexingGEMMMaps.push_back(inputMapGEMM);
  } else {
    indexingGEMMMaps.push_back(inputMapGEMM);
    indexingGEMMMaps.push_back(filterMapGEMM);
  }
  indexingGEMMMaps.push_back(resultMap);

  SmallVector<int64_t> igemmLoopBounds =
      linalgOp.getStaticLoopRanges();

  SmallVector<utils::IteratorType> igemmLoopIterators;
  for (utils::IteratorType iteratorType : linalgOp.getIteratorTypesArray()) {
    igemmLoopIterators.push_back(iteratorType);
  }

  IGEMMGenericConvDetails igemmDetails;
  igemmDetails.igemmContractionMaps = indexingGEMMMaps;
  igemmDetails.igemmOperands = isOutputChannelFirst
                                   ? SmallVector<Value>({filter, input})
                                   : SmallVector<Value>({input, filter});
  igemmDetails.igemmOperands.push_back(output);
  igemmDetails.igemmLoopBounds = igemmLoopBounds;
  igemmDetails.filterReassocIndices = filterReassocIndices;
  igemmDetails.isOutputChannelFirst = isOutputChannelFirst;
  igemmDetails.convDims = convDims;
  igemmDetails.convToIgemmDimMap = convToIgemmDimMap;
  igemmDetails.igemmLoopIterators = igemmLoopIterators;
  return canonicalizeReductionOrder(std::move(igemmDetails));
}

/// Collapses adjacent same-kind dims in the expanded IGEMM details.
/// Produces the final collapsed contraction maps, loop bounds, iterator
/// types, filter reassociation indices, and im2col output permutation.
static IGEMMGenericConvDetails collapseIGEMMGenericConvDetails(
    const IGEMMGenericConvDetails &expandedDetails) {
  SmallVector<ReassociationIndices> iterationReassociation =
      getCollapsibleIGEMMIterationGroups(expandedDetails.igemmContractionMaps,
                                         expandedDetails.igemmLoopIterators);
  if (llvm::all_of(iterationReassociation, [](ReassociationIndicesRef group) {
        return group.size() == 1;
      })) {
    IGEMMGenericConvDetails result = expandedDetails;
    int64_t inputMapIndex = result.isOutputChannelFirst ? 1 : 0;
    result.im2colOutputPerm = computeIm2colOutputPermutation(
        result.igemmContractionMaps[inputMapIndex], result.convDims,
        result.convToIgemmDimMap);
    return result;
  }

  IGEMMGenericConvDetails collapsedDetails = expandedDetails;
  collapsedDetails.igemmContractionMaps =
      collapseAffineMaps(expandedDetails.igemmContractionMaps,
                         iterationReassociation);
  collapsedDetails.igemmLoopBounds = collapseLoopBounds(
      expandedDetails.igemmLoopBounds, iterationReassociation);
  collapsedDetails.igemmLoopIterators =
      collapseIteratorTypes(expandedDetails.igemmLoopIterators,
                            iterationReassociation);
  collapsedDetails.convToIgemmDimMap = collapseConvToIgemmDimMap(
      expandedDetails.convToIgemmDimMap, iterationReassociation);

  AffineMap expandedFilterMap =
      expandedDetails.igemmContractionMaps[expandedDetails.isOutputChannelFirst
                                               ? 0
                                               : 1];
  collapsedDetails.filterReassocIndices =
      getOperandReassociation(expandedFilterMap, iterationReassociation);

  int64_t inputMapIndex = expandedDetails.isOutputChannelFirst ? 1 : 0;
  AffineMap collapsedInputMap =
      collapsedDetails.igemmContractionMaps[inputMapIndex];
  collapsedDetails.im2colOutputPerm =
      computeIm2colOutputPermutation(
          collapsedInputMap, collapsedDetails.convDims,
          collapsedDetails.convToIgemmDimMap);
  return collapsedDetails;
}

FailureOr<IGEMMGenericConvDetails>
getIGEMMGenericConvDetails(linalg::LinalgOp linalgOp) {
  FailureOr<IGEMMGenericConvDetails> expandedDetails =
      getExpandedIGEMMGenericConvDetails(linalgOp);
  if (failed(expandedDetails)) {
    return failure();
  }
  return collapseIGEMMGenericConvDetails(*expandedDetails);
}

//===---------------------------------------------------------------------===//
// Checking for horizontally fused contraction ops.
// Copied over from implementation of `isaContractionInterfaceImpl` in Linalg
//===---------------------------------------------------------------------===//

/// If the value is defined by a chain of unary side effect-free, go up the
/// use-def chain until the first value that isn't defined by such an op.
// TODO: relax to multi-operands with constants, which are technically unary ops
// as needed (e.g. add5).
static Value getSourceSkipUnary(Value value) {
  Operation *op = value.getDefiningOp();
  while (op && op->getNumOperands() == 1) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface || !iface.hasNoEffect()) {
      break;
    }
    value = op->getOperand(0);
    op = value.getDefiningOp();
  }
  return value;
}

struct ContractionOpSequenceArgs {
  std::pair<BlockArgument, BlockArgument> operands;
  BlockArgument accumulator;
};
static std::optional<ContractionOpSequenceArgs>
isContractionOpSequence(Value yielded,
                        function_ref<bool(Operation *, Operation *)> isaPair) {
  Operation *reductionOp = yielded.getDefiningOp();
  if (reductionOp->getNumResults() != 1 || reductionOp->getNumOperands() != 2) {
    return std::nullopt;
  }

  Value reductionLHS = getSourceSkipUnary(reductionOp->getOperand(0));
  Value reductionRHS = getSourceSkipUnary(reductionOp->getOperand(1));

  BlockArgument updated = dyn_cast<BlockArgument>(reductionRHS);
  Value contributed = reductionLHS;
  if (!updated) {
    updated = dyn_cast<BlockArgument>(reductionLHS);
    if (!updated) {
      return std::nullopt;
    }
    contributed = reductionRHS;
  }
  contributed = getSourceSkipUnary(contributed);

  Operation *elementwiseOp = contributed.getDefiningOp();
  if (!elementwiseOp || elementwiseOp->getNumResults() != 1 ||
      elementwiseOp->getNumOperands() != 2) {
    return std::nullopt;
  }

  if (!isaPair(elementwiseOp, reductionOp)) {
    return std::nullopt;
  }

  auto elementwiseLHS = dyn_cast_if_present<BlockArgument>(
      getSourceSkipUnary(elementwiseOp->getOperand(0)));
  auto elementwiseRHS = dyn_cast_if_present<BlockArgument>(
      getSourceSkipUnary(elementwiseOp->getOperand(1)));
  if (!elementwiseLHS || !elementwiseRHS) {
    return std::nullopt;
  }

  return ContractionOpSequenceArgs{{elementwiseLHS, elementwiseRHS}, updated};
}

/// Returns true if the two operations are of the kinds specified by a pair of
/// consecutive template arguments.
template <typename AddOpTy, typename MulOpTy, typename... Args>
static bool isPairTemplateImpl(Operation *add, Operation *mul) {
  static_assert(sizeof...(Args) % 2 == 0,
                "expected an even number of template arguments");
  if (isa<AddOpTy>(add) && isa<MulOpTy>(mul)) {
    return true;
  }

  if constexpr (sizeof...(Args) > 0) {
    return isPairTemplateImpl<Args...>(add, mul);
  } else {
    return false;
  }
}

/// Returns true if the block is a body of a contraction with the kinds of
/// operations given pairwise by template arguments.
template <typename... Args>
static std::optional<ContractionOpSequenceArgs>
isContractionOpSequence(Value yielded) {
  return isContractionOpSequence(yielded, &isPairTemplateImpl<Args...>);
}

/// Recognize an operation that is horizontally fused contraction.
/// TODO: The logic below is quite convoluted. Might be better
/// off having a dedicated operation for this.
bool isaHorizontallyFusedContraction(Operation *op) {
  auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return false;
  }
  if (linalgOp->getNumResults() == 1) {
    return false;
  }
  // Check that the number of `ins` is one more than the number of results.
  if (linalgOp.getNumDpsInputs() != linalgOp->getNumResults() + 1) {
    return false;
  }
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  if (!llvm::all_of(indexingMaps, [](AffineMap m) {
        return m.isProjectedPermutation() && !m.isPermutation();
      })) {
    return false;
  }

  llvm::SetVector<BlockArgument> rhsArgs;
  llvm::SetVector<BlockArgument> outArgs;
  for (auto [index, yieldedVal] :
       llvm::enumerate(linalgOp.getBlock()->getTerminator()->getOperands())) {
    std::optional<ContractionOpSequenceArgs> args =
        isContractionOpSequence<arith::MulFOp, arith::AddFOp, arith::MulIOp,
                                arith::AddIOp, complex::MulOp, complex::AddOp,
                                arith::AndIOp, arith::OrIOp>(yieldedVal);
    if (!args) {
      return false;
    }
    BlockArgument lhs = args->operands.first;
    BlockArgument rhs = args->operands.second;

    // One of the block arguments must be argument 0, corresponding to the LHS.
    if (lhs.getArgNumber() != 0) {
      if (rhs.getArgNumber() != 0) {
        return false;
      }
      std::swap(lhs, rhs);
    }
    assert(rhs.getArgNumber() != 0 && "cannot have rhs be arg number 0");
    if (rhs.getArgNumber() != index + 1) {
      return false;
    }
    BlockArgument accumulator = args->accumulator;
    if (accumulator.getArgNumber() != index + linalgOp.getNumDpsInputs()) {
      return false;
    }
  }

  // Check that they have valid m, n and k dims.
  ArrayRef<AffineMap> indexingMapsRef(indexingMaps);
  AffineMap lhsIndexingMap = indexingMaps.front();

  auto getResultDims = [](AffineMap m) {
    auto r = llvm::map_range(m.getResults(), [](AffineExpr e) {
      return cast<AffineDimExpr>(e).getPosition();
    });
    return llvm::SmallDenseSet<unsigned>(r.begin(), r.end());
  };
  llvm::SmallDenseSet<unsigned> lhsDims = getResultDims(lhsIndexingMap);

  // Check that all the horizontally fused gemms have common N-dims. M and K
  // dims are already known consistent since they are what the LHS has.
  std::optional<llvm::SmallDenseSet<unsigned>> refNDimsSet;
  for (auto [rhsIndexingMap, outputIndexingMap] :
       llvm::zip_equal(indexingMapsRef.slice(1, linalgOp.getNumDpsInputs() - 1),
                       indexingMapsRef.take_back(linalgOp.getNumDpsInits()))) {
    llvm::SmallDenseSet<unsigned> rhsDims = getResultDims(rhsIndexingMap);
    llvm::SmallDenseSet<unsigned> outsDims = getResultDims(outputIndexingMap);
    llvm::SmallDenseSet<unsigned> mDims = lhsDims;
    llvm::set_intersect(mDims, outsDims);
    if (mDims.empty()) {
      return false;
    }
    llvm::SmallDenseSet<unsigned> nDims = rhsDims;
    llvm::set_intersect(nDims, outsDims);
    if (nDims.empty()) {
      return false;
    }
    llvm::SmallDenseSet<unsigned> kDims = lhsDims;
    llvm::set_intersect(kDims, rhsDims);
    if (kDims.empty()) {
      return false;
    }

    if (refNDimsSet) {
      if (!llvm::all_of(nDims, [&](unsigned nDim) {
            return refNDimsSet->contains(nDim);
          })) {
        return false;
      }
    } else {
      refNDimsSet = std::move(nDims);
    }
  }
  return true;
}

bool isArgmaxOp(linalg::GenericOp genericOp) {
  // Check for 2 results(value, index), and 1 input
  if (genericOp.getNumDpsInits() != 2) {
    return false;
  }

  if (genericOp.getNumDpsInputs() != 1) {
    return false;
  }

  // Argmax will require 1D reduction.
  if (genericOp.getNumReductionLoops() != 1) {
    return false;
  }

  // TODO: Add better affine map checks.
  auto indexingMaps = genericOp.getIndexingMapsArray();
  if (!indexingMaps[0].isIdentity()) {
    return false;
  }

  // Check that initial value is negative Infinite.
  // TODO: Move this check to ukernel once we implement
  //       variant to handle non neg-Inf initial value.
  Value initVal = genericOp.getDpsInitOperand(0)->get();
  auto fillOp = initVal.getDefiningOp<linalg::FillOp>();
  if (!fillOp) {
    return false;
  }
  Value fillVal = fillOp.getDpsInputOperand(0)->get();
  if (!matchPattern(fillVal, m_NegInfFloat())) {
    return false;
  }

  // Work back from linalg.yield and check body of genericOp.
  // The genericOp should yield the result of an arith.select,
  // preceded by an arith.cmpf, arith.maximumf, and arith.extui
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;

  // Producer of linalg.yield 1st arg is arith.maximumf
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return false;
    }
    if (!matchPattern(producer, m_Op<arith::MaximumFOp>())) {
      return false;
    }
  }

  // Producer of linalg.yield op 2nd arg is arith.select
  // TODO: Add check that select is selecting between linalg.index and index of
  // current max.
  {
    producerOutput = yieldOp->getOperand(1);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return false;
    }
    if (!matchPattern(producer, m_Op<arith::SelectOp>())) {
      return false;
    }
    auto selectOp = cast<arith::SelectOp>(producerOutput.getDefiningOp());
    Value trueVal = selectOp.getTrueValue();
    if (auto castOp = trueVal.getDefiningOp<arith::IndexCastOp>()) {
      trueVal = castOp.getIn();
    }

    // Ensure the true value is directly produced by linalg.index.
    auto indexOp = trueVal.getDefiningOp<linalg::IndexOp>();
    if (!indexOp) {
      return false;
    }
  }

  // Producer of arith.select op is arith.cmpf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return false;
    }
    auto producerCmpFOp = dyn_cast<arith::CmpFOp>(producer);
    if (!producerCmpFOp ||
        producerCmpFOp.getPredicate() != arith::CmpFPredicate::OGT) {
      return false;
    }

    // Check that in and out of cmpf are loop variables.
    // Currently first operand is disabled because it may be mixed type
    // which would lead it to be extf(%arg0).
    // TODO: Add better mixed type support check.
    if (producer->getOperand(1) != genericOp.getBody()->getArgument(1)) {
      return false;
    }
  }

  return true;
}

bool hasOnlyScalarInputs(linalg::GenericOp linalgOp) {
  // Check if there are any non-scalar inputs or non-scalar captures in the
  // region.
  for (Value input : linalgOp.getDpsInputs()) {
    if (isa<ShapedType>(input.getType())) {
      return false;
    }
  }

  bool foundNonScalar = false;
  visitUsedValuesDefinedAbove(linalgOp.getRegion(), [&](OpOperand *operand) {
    if (isa<ShapedType>(operand->get().getType())) {
      foundNonScalar = true;
    }
  });

  return !foundNonScalar;
}

bool isPureMatmul(Operation *op) {
  auto matmulOp = dyn_cast_if_present<linalg::MatmulOp>(op);
  return matmulOp &&
         linalg::MatmulOp::isDefaultIndexingMaps(matmulOp.getIndexingMaps());
}

bool isPureBatchMatmul(Operation *op) {
  auto batchMatmulOp = dyn_cast_if_present<linalg::BatchMatmulOp>(op);
  return batchMatmulOp && linalg::BatchMatmulOp::isDefaultIndexingMaps(
                              batchMatmulOp.getIndexingMaps());
}

// Indicates whether the given linalg op represents a transpose. In particular,
// it requires a single input where the indexing maps are full permutations and
// non-equal.
bool isaTransposeOpInterface(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops()) {
    return false;
  }

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1) {
    return false;
  }
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isPermutation() ||
      !mapRange.back().isPermutation() || mapRange.front() == mapRange.back()) {
    return false;
  }
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
