// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "iree-linalgExt-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::LinalgExt {

static bool hasAllOneValues(ArrayRef<int64_t> attr) {
  return llvm::all_of(attr, [](int64_t element) { return element == 1; });
}

OpFoldResult addOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto addMap = AffineMap::get(2, 0, {d0 + d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, addMap, {a, b});
}

OpFoldResult mulOfrs(OpBuilder &builder, Location loc, OpFoldResult a,
                     OpFoldResult b) {
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  auto addMap = AffineMap::get(2, 0, {d0 * d1});
  return affine::makeComposedFoldedAffineApply(builder, loc, addMap, {a, b});
}

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.createOrFold<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
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
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return b.create<tensor::ExtractSliceOp>(loc, src, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
      })
      .Default([&](Type t) -> Operation * {
        assert(false && "invalid type");
        return nullptr;
      });
}

Value castValue(OpBuilder &b, Location loc, Value src, ShapedType type) {
  return TypeSwitch<Type, Value>(src.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        assert(isa<RankedTensorType>(type) && "expected compatible type");
        return b.create<tensor::CastOp>(loc, type, src)->getResult(0);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        assert(isa<MemRefType>(type) && "expected compatible type");
        return b.create<memref::CastOp>(loc, type, src)->getResult(0);
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
  return rewriter.create<arith::ConstantOp>(
      loc, DenseFPElementsAttr::get(
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
  mlir::getBackwardSlice(yieldOp.getOperand(0), &sliceOps, options);
  return hasTensorExtract;
}

FailureOr<IGEMMGenericConvDetails>
getIGEMMGenericConvDetails(linalg::LinalgOp linalgOp) {

  auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
  MLIRContext *ctx = linalgOp->getContext();
  if (failed(convDimsOrFailure))
    return failure();
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
    llvm::dbgs() << "\nconv depth multiplier: ";
    llvm::interleaveComma(convDims.depth, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  Value input = linalgOp.getDpsInputs()[0];
  Value filter = linalgOp.getDpsInputs()[1];
  Value output = linalgOp.getDpsInits()[0];
  auto inputType = llvm::cast<ShapedType>(input.getType());
  auto filterType = llvm::cast<ShapedType>(filter.getType());
  auto outputType = llvm::cast<ShapedType>(output.getType());

  if (!filterType.hasStaticShape() || !inputType.hasStaticShape()) {
    LDBG("[unimplemented] expected 'filterType' and 'inputType' to have static "
         "shapes.");
    return failure();
  }

  // TODO: Support dilation.
  if (!hasAllOneValues(convDims.dilations)) {
    LDBG("[unimplemented] expected no dilations (expected dilations to all be "
         "one).");
    return failure();
  }
  // TODO: Support depthwise.
  if (!convDims.depth.empty()) {
    LDBG("[unimplemented] expected no depth");
    return failure();
  }

  // TODO: Support pooling operations. For pooling ops, the input/output channel
  // size will be categorized as the additional batch dimension.
  if (convDims.outputChannel.empty() || convDims.inputChannel.empty()) {
    LDBG("[unimplemented] expected no pooling operations");
    return failure();
  }
  auto filterShape = filterType.getShape();
  auto outputShape = outputType.getShape();
  auto indexingMaps = linalgOp.getIndexingMapsArray();
  auto filterMap = indexingMaps[1];

  SmallVector<int64_t> reductionDims;
  for (auto iter : llvm::enumerate(linalgOp.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(iter.value())) {
      reductionDims.push_back(iter.index());
    }
  }
  SmallVector<int64_t> filterkPos;
  for (auto reductionDim : reductionDims) {
    std::optional<int64_t> maybeDim = filterMap.getResultPosition(
        getAffineDimExpr(reductionDim, filterMap.getContext()));
    filterkPos.push_back(maybeDim.value());
  }
  // group together adjacent reduction dimensions in the filter
  SmallVector<ReassociationIndices> collapsedFilterReductionDim;
  int64_t prevFilterIndex = filterkPos[0];
  int64_t currCollapsedIndex = 0;
  collapsedFilterReductionDim.push_back({filterkPos[0]});
  SmallVector<int64_t> kShape = {filterShape[filterkPos[0]]};
  for (auto currPos : llvm::ArrayRef(filterkPos).drop_front()) {
    if (prevFilterIndex == currPos - 1) {
      collapsedFilterReductionDim[currCollapsedIndex].push_back(currPos);
    } else {
      collapsedFilterReductionDim.push_back({currPos});
      ++currCollapsedIndex;
    }
    prevFilterIndex = currPos;
  }

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> filterIterators;
  SmallVector<int64_t> filterNdims;
  for (auto outputChannel : convDims.outputChannel) {
    std::optional<int64_t> maybeDim = filterMap.getResultPosition(
        getAffineDimExpr(outputChannel, filterMap.getContext()));
    filterNdims.push_back(maybeDim.value());
  }
  SmallVector<ReassociationIndices> filterReassocIndices;
  // Interleave the parallel dims with the reduction dims.
  int64_t filterNdimPos = 0;
  for (auto collapsedDim : collapsedFilterReductionDim) {
    for (int i = filterNdimPos; i < filterNdims.size(); i++) {
      if (filterNdims[i] < collapsedDim[0]) {
        filterReassocIndices.push_back({filterNdims[i]});
        filterIterators.push_back(parallel);
        filterNdimPos = i + 1;
      } else {
        break;
      }
    }
    filterIterators.push_back(reduction);
    filterReassocIndices.push_back(collapsedDim);
  }
  // insert any leftover parallel dims in the end.
  for (int i = filterNdimPos; i < filterNdims.size(); i++) {
    filterReassocIndices.push_back({filterNdims[i]});
    filterIterators.push_back(parallel);
  }
  SmallVector<int64_t> reshapedFilterShape(filterReassocIndices.size(), 1);
  for (auto [idx, indices] : llvm::enumerate(filterReassocIndices)) {
    for (auto index : indices) {
      reshapedFilterShape[idx] *= filterShape[index];
    }
  }

  int64_t numBDims = (convDims.batch).size();
  int64_t numMDims = (convDims.outputImage).size();
  int64_t numNDims = (convDims.outputChannel).size();
  int64_t numParallelDims = numBDims + numMDims + numNDims;
  int64_t numKDims = collapsedFilterReductionDim.size();
  SmallVector<utils::IteratorType> genericIterators(numParallelDims, parallel);
  genericIterators.insert(genericIterators.end(), numKDims, reduction);

  SmallVector<AffineExpr> dims(numParallelDims + numKDims);
  bindDimsList<AffineExpr>(ctx, dims);
  auto resultMap = AffineMap::get(
      numParallelDims + numKDims, 0,
      SmallVector<AffineExpr>(dims.begin(), dims.begin() + numParallelDims),
      ctx);

  bool isOutputChannelFirst = false;
  auto outputChannelPos = convDims.outputChannel;
  auto outputImagePos = convDims.outputImage;
  if (outputChannelPos.back() < outputImagePos[0])
    isOutputChannelFirst = true;

  // prepare the input map.
  SmallVector<AffineExpr> inputDims;
  // Add the batch dimensions.
  inputDims.insert(inputDims.end(), dims.begin(), dims.begin() + numBDims);
  int64_t starting_m_pos =
      isOutputChannelFirst ? numBDims + numNDims : numBDims;
  // Add the M dims.
  inputDims.insert(inputDims.end(), dims.begin() + starting_m_pos,
                   dims.begin() + starting_m_pos + numMDims);
  // Add the reduction dims.
  inputDims.insert(inputDims.end(), dims.begin() + numParallelDims, dims.end());
  auto inputMapGEMM =
      AffineMap::get(numParallelDims + numKDims, 0, inputDims, ctx);

  // prepare filter map.
  SmallVector<AffineExpr> filterDims;
  int64_t curr_n_pos = isOutputChannelFirst ? numBDims : numBDims + numMDims;
  int64_t curr_k_pos = numBDims + numMDims + numNDims;

  for (auto iter : filterIterators) {
    if (iter == parallel) {
      filterDims.push_back(dims[curr_n_pos++]);
    } else if (iter == reduction) {
      filterDims.push_back(dims[curr_k_pos++]);
    }
  }
  auto filterMapGEMM =
      AffineMap::get(numParallelDims + numKDims, 0, filterDims, ctx);

  SmallVector<AffineMap> indexingGEMMMaps;
  if (isOutputChannelFirst) {
    indexingGEMMMaps.push_back(filterMapGEMM);
    indexingGEMMMaps.push_back(inputMapGEMM);
  } else {
    indexingGEMMMaps.push_back(inputMapGEMM);
    indexingGEMMMaps.push_back(filterMapGEMM);
  }
  indexingGEMMMaps.push_back(resultMap);
  IGEMMGenericConvDetails igemmDetails;
  igemmDetails.igemmContractionMaps = indexingGEMMMaps;
  igemmDetails.igemmOperands = isOutputChannelFirst
                                   ? SmallVector<Value>({filter, input})
                                   : SmallVector<Value>({input, filter});
  igemmDetails.igemmOperands.push_back(output);
  SmallVector<int64_t> igemmLoopBounds;
  igemmLoopBounds.insert(igemmLoopBounds.end(), outputShape.begin(),
                         outputShape.begin() + numParallelDims);

  SmallVector<utils::IteratorType> igemmLoopIterators(outputShape.size(),
                                                      parallel);

  for (auto iter : llvm::enumerate(filterIterators)) {
    if (iter.value() == reduction) {
      igemmLoopBounds.push_back(reshapedFilterShape[iter.index()]);
      igemmLoopIterators.push_back(reduction);
    }
  }
  igemmDetails.igemmLoopBounds = igemmLoopBounds;
  igemmDetails.filterReassocIndices = filterReassocIndices;
  igemmDetails.isOutputChannelFirst = isOutputChannelFirst;
  igemmDetails.convDims = convDims;
  igemmDetails.igemmLoopIterators = igemmLoopIterators;

  return igemmDetails;
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
    if (!iface || !iface.hasNoEffect())
      break;
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

  auto elementwiseLHS = dyn_cast_or_null<BlockArgument>(
      getSourceSkipUnary(elementwiseOp->getOperand(0)));
  auto elementwiseRHS = dyn_cast_or_null<BlockArgument>(
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
  if (isa<AddOpTy>(add) && isa<MulOpTy>(mul))
    return true;

  if constexpr (sizeof...(Args) > 0)
    return isPairTemplateImpl<Args...>(add, mul);
  else
    return false;
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
bool isaHorizontallyFusedContraction(linalg::LinalgOp linalgOp) {
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
  }
  return true;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
