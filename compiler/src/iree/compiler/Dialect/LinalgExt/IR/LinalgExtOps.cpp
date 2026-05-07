// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/Repeated.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static Type getComplexElementTypeOrSelf(Type ty) {
  if (auto complex = dyn_cast_if_present<ComplexType>(ty)) {
    return complex.getElementType();
  }
  return ty;
}

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ArrayRef<OpOperand *> inputOperands, MutableOperandRange outputOperands) {
  for (OpOperand *operand : inputOperands) {
    if (!isa<MemRefType>(operand->get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : outputOperands) {
    if (!isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

/// Return true if `dimsPos` is invalid. It is invalid when: a) it contains
/// duplicate. b) At least one dimension is out of bound (`dimPos` is >= 0 and <
/// rank). c) the number of elements in `dimsPos` is > than `rank`.
static bool isInvalid(ArrayRef<int64_t> dimsPos, int64_t rank) {
  // early exit.
  if (dimsPos.size() > rank) {
    return true;
  }
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos) {
    uniqued.insert(dim);
  }
  if (dimsPos.size() != uniqued.size()) {
    return true;
  }
  return llvm::any_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos < 0 || dimPos >= rank; });
}

static bool allIndexingsAreProjectedPermutation(IndexingMapOpInterface op) {
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

/// Emit an error and return failure when `seq` is invalid. It is only valid
/// when it is a permutation of the sequence 0...length(seq) - 1.
static LogicalResult
isPermSequence(function_ref<InFlightDiagnostic()> emitError,
               ArrayRef<int64_t> seq) {
  BitVector seen(seq.size(), false);
  for (auto [idx, dim] : llvm::enumerate(seq)) {
    if (dim < 0 || dim >= seq.size()) {
      return emitError().attachNote() << "element (" << dim << ") at index#"
                                      << idx << " is out of bounds";
    }
    if (seen.test(dim)) {
      return emitError().attachNote()
             << "element (" << dim << ") at index#" << idx << " is a duplicate";
    }
    seen.set(dim);
  }
  return success();
}

/// Returns true if the dimension of `sourceShape` is smaller than the dimension
/// of the `limitShape`.
static bool isSmallerThan(ArrayRef<int64_t> sourceShape,
                          ArrayRef<int64_t> limitShape) {
  assert(
      sourceShape.size() == limitShape.size() &&
      "expected source shape rank, and limit of the shape to have same rank");
  return llvm::all_of(llvm::zip_equal(sourceShape, limitShape),
                      [](std::tuple<int64_t, int64_t> it) {
                        int64_t sourceExtent = std::get<0>(it);
                        int64_t limit = std::get<1>(it);
                        return ShapedType::isDynamic(sourceExtent) ||
                               ShapedType::isDynamic(limit) ||
                               sourceExtent <= limit;
                      });
}

static AffineMap getLeadingDimsProjectionMap(MLIRContext *ctx, int64_t dimCount,
                                             int64_t projectedDimCount) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(projectedDimCount);
  for (int64_t i = 0; i < projectedDimCount; ++i) {
    exprs.push_back(getAffineDimExpr(i, ctx));
  }
  return AffineMap::get(dimCount, /*symbolCount=*/0, exprs, ctx);
}

static bool isSupportedMaskElementType(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  return intType && (intType.getWidth() == 1 || intType.getWidth() == 8);
}

/// Helper function to verify both `scatter` and `gather`. Since both ops share
/// the same semantics, we can use the same function to verify them. Note: this
/// is written from the perspective of `scatter` op. For gather, `updateType`
/// maps to the type of the output and `originalType` maps to the type of the
/// `source`.
template <typename OpTy>
static LogicalResult
verifyGatherScatter(OpTy op, int64_t sliceRank, ShapedType originalType,
                    ShapedType updateType, StringRef originalName,
                    StringRef updateName) {
  static_assert(llvm::is_one_of<OpTy, GatherOp, ScatterOp>::value,
                "applies to only gather or scatter operations");

  auto indicesType = op.getIndicesType();
  if (indicesType.getRank() < 1 ||
      !(isa<IntegerType>(indicesType.getElementType()) ||
        indicesType.getElementType().isIndex())) {
    return op->emitOpError("expected indices to be of rank 1 or greater and of "
                           "integer or index element type");
  }

  ArrayRef<int64_t> dimMap = op.getDimensionMap();
  if (failed(isPermSequence(
          [&]() { return op->emitOpError("dimension map is invalid."); },
          dimMap))) {
    return failure();
  }

  if (dimMap.size() == 0) {
    return op->emitOpError("dimension map must have at least one element");
  }

  const size_t indexDepth = op.getIndexDepth();
  const auto originalSliceRank = originalType.getRank() - indexDepth;
  if (originalSliceRank < 0) {
    return op->emitOpError("expected " + originalName +
                           " rank to be greater or equal to index depth");
  }
  if (updateType.getRank() < originalSliceRank) {
    return op->emitOpError("expected " + updateName +
                           " to be at least the rank of non indexed " +
                           originalName + " dims");
  }
  const size_t batchRank = updateType.getRank() - originalSliceRank;

  if (updateType.getRank() - batchRank != originalSliceRank) {
    return op->emitOpError("expected rank of " + updateName +
                           " value - batch rank to be "
                           "equal to rank of " +
                           originalName + " value - index depth");
  }

  if ((indicesType.getRank() != batchRank || indexDepth != 1) &&
      indicesType.getRank() != batchRank + 1) {
    return op->emitOpError("expected indices to be equal to batch rank "
                           "or batch rank + 1");
  }

  {
    // Validate the shape of indices and update value match for the first
    // `batchRank` dims.
    auto [indicesIt, updateIt] =
        llvm::mismatch(indicesType.getShape().take_front(batchRank),
                       updateType.getShape().take_front(batchRank));
    if (indicesIt != indicesType.getShape().take_front(batchRank).end()) {
      return op->emitOpError("mismatch in shape of indices and " + updateName +
                             " value at dim#")
             << (indicesIt - indicesType.getShape().begin());
    }
  }
  if (batchRank + 1 < indicesType.getShape().size() &&
      dimMap.size() != indicesType.getShape().back()) {
    return op->emitOpError(
        "size of dimension map must match the last dimension of indices");
  }

  if (std::optional<ShapedType> maybeMaskType = op.getMaskType()) {
    auto maskType = *maybeMaskType;
    if (!isSupportedMaskElementType(maskType.getElementType())) {
      return op->emitOpError(
          "expected mask to have i1 or storage-legalized i8 element type");
    }
    if (maskType.getRank() != static_cast<int64_t>(batchRank)) {
      return op->emitOpError("expected mask rank to match batch rank");
    }
    for (auto dim : llvm::seq<int64_t>(0, static_cast<int64_t>(batchRank))) {
      if (maskType.isDynamicDim(dim) || updateType.isDynamicDim(dim)) {
        continue;
      }
      if (maskType.getDimSize(dim) != updateType.getDimSize(dim)) {
        return op->emitOpError("mask shape must match batch dimensions at dim#")
               << dim;
      }
    }
  }

  {
    for (auto idx : llvm::seq<int64_t>(0, sliceRank)) {
      int64_t updateDim = idx + batchRank;
      int64_t origDim = idx + indexDepth;
      if (originalType.isDynamicDim(origDim) ||
          updateType.isDynamicDim(updateDim)) {
        continue;
      }
      if (originalType.getDimSize(origDim) !=
          updateType.getDimSize(updateDim)) {
        return op->emitOpError("shape of " + updateName + " value dim#")
               << (updateDim)
               << " must match " + originalName + " value at dim#" << (origDim);
      }
    }
  }

  return success();
}

/// For each of the operand in `operands` this function maps the static sizes of
/// dimensions to their affine dim expressions.
template <typename OpTy>
static void populateMap(OpTy op, MutableArrayRef<OpOperand> operands,
                        llvm::DenseMap<AffineExpr, int64_t> &affineExprToSize) {
  for (OpOperand &opOperand : operands) {
    if (op.isScalar(&opOperand)) {
      continue;
    }
    Value src = opOperand.get();
    auto sourceType = cast<RankedTensorType>(src.getType());
    auto sourceMap = op.getMatchingIndexingMap(&opOperand);

    // Get the `sourceShape` of the `sourceType`. If the operand is a result of
    // `tensor.cast` operation and source of the cast operation has a static
    // shape, then assign it to the `sourceShape`.
    auto castOp = src.getDefiningOp<tensor::CastOp>();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    if (castOp && tensor::canFoldIntoConsumerOp(castOp)) {
      sourceShape = castOp.getSource().getType().getShape();
    }

    // If the source shape's dimension has a static shape, map the affine dim
    // expression to the known static size.
    for (unsigned i = 0; i < sourceShape.size(); ++i) {
      if (sourceType.isDynamicDim(i)) {
        continue;
      }
      if (auto affineDimExpr =
              dyn_cast<AffineDimExpr>(sourceMap.getResult(i))) {
        affineExprToSize.try_emplace(affineDimExpr, sourceShape[i]);
      }
    }
  }
}

/// Creates new operand w.r.t 'opOperand' of `op` with static sizes
/// mapped in `affineExprToSize`. New operands are created in `newOperands` and
/// their result types is stored in `resultTypes`. If `opOperand` requires no
/// change then `changeNeeded` is false and same operand is added in the
/// `newOperands` list.
template <typename OpTy>
static void createNewOperandWithStaticSizes(
    Location loc, PatternRewriter &rewriter, OpOperand *opOperand,
    const llvm::DenseMap<AffineExpr, int64_t> &affineExprToSize, OpTy op,
    SmallVector<Value> &newOperands, SmallVector<Type> &resultTypes,
    bool &changeNeeded) {
  Value src = opOperand->get();
  newOperands.push_back(src);
  if (op.isScalar(opOperand)) {
    return;
  }
  auto sourceType = cast<RankedTensorType>(src.getType());
  Type resultType = sourceType;
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  AffineMap sourceMap = op.getMatchingIndexingMap(opOperand);
  SmallVector<int64_t> newShape;

  // If operand is updated with new shape, `newOperandNeeded` will be
  // true.
  bool newOperandNeeded = false;
  for (unsigned i = 0; i < sourceShape.size(); ++i) {
    int64_t dimShape = sourceShape[i];
    AffineExpr dimExpr = sourceMap.getResult(i);
    if (!affineExprToSize.contains(dimExpr) || !sourceType.isDynamicDim(i)) {
      newShape.push_back(dimShape);
      continue;
    }
    // Dimension has a dynamic shape and corresponding affine dim
    // expression is present in the map. So assign the size for the
    // given affine dim expression to the dimension.
    newShape.push_back(affineExprToSize.at(dimExpr));
    newOperandNeeded = true;
  }
  resultType = RankedTensorType::get(newShape, sourceType.getElementType(),
                                     sourceType.getEncoding());
  if (newOperandNeeded) {
    changeNeeded = true;
    // Get the new operand value given its size and element type by
    // casting it.
    Value newOperand = tensor::CastOp::create(rewriter, loc, resultType, src);
    unsigned index = opOperand->getOperandNumber();
    newOperands[index] = newOperand;
  }
  if (op.isDpsInit(opOperand)) {
    resultTypes.push_back(resultType);
  }
}

namespace {
/// Pattern to make an operation more static by looking at the affine dim
/// expressions of other, more static, operands. This requires the operation to
/// implement the DPS interface and to have indexing maps.
template <typename OpTy>
struct StaticizeLinalgExtOp : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (!allIndexingsAreProjectedPermutation(op)) {
      return failure();
    }

    Location loc = op.getLoc();

    // For each of the affine dim expression, check if the size is known. If
    // known add that in the map.
    llvm::DenseMap<AffineExpr, int64_t> affineExprToSize;
    populateMap(op, op->getOpOperands(), affineExprToSize);

    SmallVector<Value> newOperands;
    SmallVector<Type> resultTypes;
    newOperands.reserve(op->getNumOperands());
    resultTypes.reserve(op.getNumDpsInits());

    // Iterate over all the operands and update the static sizes.
    bool changeNeeded = false;
    for (OpOperand &opOperand : op->getOpOperands()) {
      createNewOperandWithStaticSizes(loc, rewriter, &opOperand,
                                      affineExprToSize, op, newOperands,
                                      resultTypes, changeNeeded);
    }
    if (!changeNeeded) {
      return failure();
    }

    // Clone op.
    Operation *newOp = clone(rewriter, op, resultTypes, newOperands);
    SmallVector<Value> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      Type newType = newResult.getType();
      Type oldType = oldResult.getType();
      replacements.push_back(
          (newType != oldType) ? tensor::CastOp::create(rewriter, loc, oldType,
                                                        cast<Value>(newResult))
                               : cast<Value>(newResult));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  ShapedType originalType = getOriginalType();
  ShapedType updateType = getUpdateType();
  ScatterOp op = *this;

  if (failed(verifyGatherScatter(op, getUpdateSliceRank(), originalType,
                                 updateType, "original", "update"))) {
    return failure();
  }

  Block *body = &getRegion().front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!getComplexElementTypeOrSelf(arg0Type).isIntOrFloat() ||
      !getComplexElementTypeOrSelf(arg1Type).isIntOrFloat()) {
    return op->emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return op->emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != originalType.getElementType()) {
    return op->emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of original value "
           << originalType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op->emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }

  return success();
}

LogicalResult
ScatterOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<int64_t> ScatterOp::getStaticLoopRanges() {
  // Scatter loop ranges are loop ranges for update.
  return SmallVector<int64_t>(getUpdateType().getShape());
}

SmallVector<AffineMap> ScatterOp::getIndexingMapsForOperands() {
  Builder builder(getContext());
  SmallVector<AffineMap> maps = {
      builder.getMultiDimIdentityMap(getUpdateType().getRank()),
      builder.getMultiDimIdentityMap(getIndicesType().getRank())};
  if (getMask()) {
    maps.push_back(getLeadingDimsProjectionMap(
        getContext(), getUpdateType().getRank(), getBatchRank()));
  }
  maps.push_back(/*output=*/AffineMap(nullptr));
  return maps;
}

SmallVector<AffineMap> ScatterOp::getIndexingMapsForResults() {
  return {AffineMap(nullptr)};
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  return verifyGatherScatter(*this, getOutputSliceRank(), getSourceType(),
                             getOutputType(), "source", "output");
}

LogicalResult
GatherOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<int64_t> GatherOp::getStaticLoopRanges() {
  return SmallVector<int64_t>(getOutputType().getShape());
}

SmallVector<AffineMap> GatherOp::getIndexingMapsForOperands() {
  Builder builder(getContext());
  SmallVector<AffineMap> maps = {
      AffineMap(nullptr),
      builder.getMultiDimIdentityMap(getIndicesType().getRank())};
  if (getMask()) {
    maps.push_back(getLeadingDimsProjectionMap(
        getContext(), getOutputType().getRank(), getBatchRank()));
  }
  maps.push_back(builder.getMultiDimIdentityMap(getOutputType().getRank()));
  return maps;
}

SmallVector<AffineMap> GatherOp::getIndexingMapsForResults() {
  Builder builder(getContext());
  return SmallVector<AffineMap>{
      builder.getMultiDimIdentityMap(getOutputType().getRank())};
}

namespace {
struct ConvertGatherToExtract : OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support memref case.
    if (!gatherOp.hasPureTensorSemantics() || gatherOp.getMask()) {
      return failure();
    }

    auto loc = gatherOp.getLoc();
    ArrayRef<int64_t> indicesShape = gatherOp.getIndicesType().getShape();
    ArrayRef<int64_t> batchShape =
        indicesShape.take_front(gatherOp.getBatchRank());
    if (!llvm::all_of(batchShape, [](int64_t size) { return size == 1; })) {
      return failure();
    }

    // Get all `indexDepth` indices as scalars.
    SmallVector<Value> indices(
        indicesShape.size(), arith::ConstantIndexOp::create(rewriter, loc, 0));
    SmallVector<OpFoldResult> offsets(gatherOp.getIndexDepth());
    for (int64_t i = 0; i < gatherOp.getIndexDepth(); ++i) {
      indices.back() = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value elem = tensor::ExtractOp::create(rewriter, loc,
                                             gatherOp.getIndices(), indices);
      offsets[i] = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(), elem)
                       .getResult();
    }

    applyPermutationToVector(offsets, gatherOp.getDimensionMap());
    int64_t sourceRank = gatherOp.getSourceType().getRank();
    offsets.resize(sourceRank, rewriter.getIndexAttr(0));

    // Create the new `tensor.extract_slice`.
    SmallVector<OpFoldResult> strides(sourceRank, rewriter.getIndexAttr(1));
    SmallVector<int64_t> resultShape(gatherOp.getIndexDepth(), 1);
    SmallVector<OpFoldResult> sizes(gatherOp.getIndexDepth(),
                                    rewriter.getIndexAttr(1));
    for (int64_t i = gatherOp.getIndexDepth(); i < sourceRank; ++i) {
      sizes.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, gatherOp.getSource(), i));
      resultShape.push_back(gatherOp.getSourceType().getDimSize(i));
    }
    auto resultType =
        cast<RankedTensorType>(gatherOp.getSourceType()).clone(resultShape);
    auto sliceOp = tensor::ExtractSliceOp::create(rewriter, loc, resultType,
                                                  gatherOp.getSource(), offsets,
                                                  sizes, strides);

    // `sliceOp` may differ from the expected result type by leading unit
    // dimensions. Reshape so that the types match.
    int64_t sliceRank = sliceOp.getResultType().getRank();
    int64_t gatherRank = gatherOp.getOutputType().getRank();
    if (sliceRank < gatherRank) {
      SmallVector<ReassociationIndices> reassoc(1);
      llvm::append_range(reassoc[0], llvm::seq(gatherOp.getBatchRank()));
      for (int64_t i = 0; i < sourceRank - 1; ++i) {
        reassoc.emplace_back(1, i + gatherOp.getBatchRank());
      }

      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          gatherOp, gatherOp.getOutputType(), sliceOp.getResult(), reassoc);
    } else if (sliceRank > gatherRank) {
      SmallVector<ReassociationIndices> reassoc(1);
      llvm::append_range(reassoc[0], llvm::seq(sliceRank - gatherRank + 1));
      for (int64_t i = sliceRank - gatherRank + 1; i < sliceRank; ++i) {
        reassoc.emplace_back(1, i);
      }

      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          gatherOp, gatherOp.getOutputType(), sliceOp.getResult(), reassoc);
    } else {
      rewriter.replaceOp(gatherOp, sliceOp);
    }
    return success();
  }
};
} // namespace

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *ctx) {
  results.add<ConvertGatherToExtract>(ctx);
}

namespace {
/// Convert an identity map_load or map_store to a copy operation.
/// We keep the copy to preserve DPS semantics.
template <typename OpTy>
struct ConvertIdentityMapLoadStoreToCopy : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!op.isIdentity()) {
      return failure();
    }
    if (op.isVectorized()) {
      return failure();
    }
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    Value source;
    if constexpr (std::is_same_v<OpTy, MapLoadOp>) {
      source = op.getSource();
    } else {
      source = op.getInput();
    }
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, source, op.getOutput());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MapLoadOp
//===----------------------------------------------------------------------===//

LogicalResult MapLoadOp::verify() {
  if (getSourceType().getElementType() != getOutputType().getElementType()) {
    return emitOpError("expected source and output element types to match");
  }
  Region &transformRegion = getTransformationRegion();
  Block &transformBody = transformRegion.getBlocks().front();
  if (transformBody.getNumArguments() != getOutputRank()) {
    return emitOpError("expected number of block arguments to be equal "
                       "to the output rank");
  }
  if (!llvm::all_of(transformBody.getArgumentTypes(),
                    llvm::IsaPred<IndexType>)) {
    return emitOpError("expected block arguments to be index types");
  }
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  if (yieldOp->getNumOperands() != getSourceRank() + 1) {
    return yieldOp.emitOpError(
        "expected transformation_region to yield a "
        "value for each source dimension and a padding value");
  }
  for (int operandIdx = 0; operandIdx < getSourceRank(); ++operandIdx) {
    if (!isa<IndexType>(yieldOp.getOperandTypes()[operandIdx])) {
      return yieldOp.emitOpError("expected yielded indices to be index types");
    }
  }
  Type paddingType = yieldOp.getOperandTypes()[getSourceRank()];
  Type elementType = getSourceType().getElementType();
  if (paddingType != elementType) {
    return yieldOp.emitOpError("expected yielded padding value type to match "
                               "source element type");
  }
  return success();
}

void MapLoadOp::insertTransformationAtStart(
    OpBuilder &builder,
    function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
        transformationBuilder,
    int64_t numOutputIndices) {
  Block &transformBody = getTransformationRegion().front();
  SmallVector<BlockArgument> oldOutputIndices(transformBody.getArguments());
  llvm::Repeated<Type> indexTypes(numOutputIndices, builder.getIndexType());
  SmallVector<Location> locs(numOutputIndices, getLoc());

  // Create the new block arguments for the new output indices, and transform
  // them using the callback.
  SmallVector<BlockArgument> newOutputIndices(
      transformBody.addArguments(indexTypes, locs));
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&transformBody);
  SmallVector<Value> newOutputIndicesTransformed(
      transformationBuilder(newOutputIndices));

  // Replace the old output indices with the results of the transformation on
  // the new output indices.
  assert(oldOutputIndices.size() == newOutputIndicesTransformed.size() &&
         "expected transformation to produce the same number of Values as the "
         "previous number of output indices.");
  for (auto [oldIdx, newIdx] :
       llvm::zip_equal(oldOutputIndices, newOutputIndicesTransformed)) {
    oldIdx.replaceAllUsesWith(newIdx);
  }
  transformBody.eraseArguments(0, oldOutputIndices.size());
}

/// Shared implementation for inlining the transformation body of map_load
/// and map_store ops.
static void inlineMapLoadStoreBodyImpl(
    OpBuilder &b, Location loc, Region &transformRegion,
    ValueRange transformBodyIndices,
    function_ref<void(OpBuilder &, Location, ArrayRef<Value>)> bodyBuilder) {
  Block &transformBlock = transformRegion.front();
  IRMapping mapping;
  // Map the induction variables of the loop nest to the block arguments of the
  // transformation body.
  for (auto [idx, arg] : llvm::enumerate(transformBlock.getArguments())) {
    mapping.map(arg, transformBodyIndices[idx]);
  }
  // Clone the operations within the transformation body to the current
  // insertion point, and map their results to the new cloned operations'
  // results.
  for (Operation &op : transformBlock.without_terminator()) {
    Operation *clonedOp = b.clone(op, mapping);
    for (auto [result, clonedResult] :
         llvm::zip_equal(op.getResults(), clonedOp->getResults())) {
      mapping.map(result, clonedResult);
    }
  }

  // Get the cloned values that were yielded by the transformation body to pass
  // to the bodyBuilder.
  SmallVector<Value> mappedYieldedValues = llvm::map_to_vector(
      transformBlock.getTerminator()->getOperands(),
      [&](Value operand) -> Value { return mapping.lookupOrDefault(operand); });
  bodyBuilder(b, loc, mappedYieldedValues);
}

void MapLoadOp::inlineMapLoadBody(
    OpBuilder &b, Location loc, ValueRange transformBodyIndices,
    function_ref<void(OpBuilder &, Location, ArrayRef<Value>)> bodyBuilder) {
  inlineMapLoadStoreBodyImpl(b, loc, getTransformationRegion(),
                             transformBodyIndices, bodyBuilder);
}

bool MapLoadOp::isIdentity() {
  if (getSourceType() != getOutputType()) {
    return false;
  }
  // Bail out on dynamic shapes.
  if (!getSourceType().hasStaticShape()) {
    return false;
  }
  // Check that the block arguments are directly yielded in the order that they
  // are defined in the block (excluding padding).
  Block &transformBody = getTransformationRegion().getBlocks().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  for (unsigned i = 0; i < getSourceRank(); ++i) {
    auto yieldedBbArg = dyn_cast<BlockArgument>(yieldOp.getOperand(i));
    if (yieldedBbArg != transformBody.getArgument(i)) {
      return false;
    }
  }
  return true;
}

Value MapLoadOp::getPaddingValue() {
  Block &transformBody = getTransformationRegion().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  return yieldOp.getOperand(yieldOp.getNumOperands() - 1);
}

void MapLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *ctx) {
  results.add<ConvertIdentityMapLoadStoreToCopy<MapLoadOp>>(ctx);
}

MapLoadOp MapLoadOp::createIdentityMapLoad(OpBuilder &builder, Location loc,
                                           Value source, Value output) {
  assert(source.getType() == output.getType() &&
         "expected source and output types to match");
  SmallVector<Type> resultType;
  if (isa<RankedTensorType>(output.getType())) {
    resultType.push_back(output.getType());
  }
  auto mapLoadOp = MapLoadOp::create(builder, loc, resultType, source, output);

  // Add the transformation block with an identity transformation.
  Region &region = mapLoadOp.getTransformationRegion();
  auto outputType = cast<ShapedType>(output.getType());
  SmallVector<Location> blockArgLocs(outputType.getRank(), loc);
  llvm::Repeated<Type> indexTypes(outputType.getRank(), builder.getIndexType());
  OpBuilder::InsertionGuard guard(builder);
  Block *block =
      builder.createBlock(&region, region.end(), indexTypes, blockArgLocs);
  SmallVector<Value> yieldedValues(block->getArguments());

  // Add a poison padding value. The identity transformation shouldn't need it
  // since source and output have the same shape. Using poison indicates that
  // no real padding is needed, and allows foldPadIntoMapLoad to detect
  // whether it's safe to set a new padding value.
  Type elementType = outputType.getElementType();
  Value padding = ub::PoisonOp::create(builder, loc, elementType);
  yieldedValues.push_back(padding);
  IREE::LinalgExt::YieldOp::create(builder, loc, yieldedValues);
  return mapLoadOp;
}

//===----------------------------------------------------------------------===//
// MapStoreOp
//===----------------------------------------------------------------------===//

MapStoreOp MapStoreOp::createIdentityMapStore(OpBuilder &builder, Location loc,
                                              Value input, Value output) {
  assert(input.getType() == output.getType() &&
         "expected input and output types to match");
  SmallVector<Type> resultType;
  if (isa<RankedTensorType>(output.getType())) {
    resultType.push_back(output.getType());
  }
  auto mapStoreOp = MapStoreOp::create(builder, loc, resultType, input, output);

  // Add the transformation block with an identity transformation.
  Region &region = mapStoreOp.getTransformationRegion();
  auto inputType = cast<ShapedType>(input.getType());
  SmallVector<Location> blockArgLocs(inputType.getRank(), loc);
  llvm::Repeated<Type> indexTypes(inputType.getRank(), builder.getIndexType());
  OpBuilder::InsertionGuard guard(builder);
  Block *block =
      builder.createBlock(&region, region.end(), indexTypes, blockArgLocs);
  SmallVector<Value> yieldedValues(block->getArguments());
  Value mask = arith::ConstantIntOp::create(builder, loc, /*value=*/1,
                                            /*width=*/1);
  yieldedValues.push_back(mask);
  IREE::LinalgExt::YieldOp::create(builder, loc, yieldedValues);
  return mapStoreOp;
}

LogicalResult MapStoreOp::verify() {
  if (getInputType().getElementType() != getOutputType().getElementType()) {
    return emitOpError("expected input and output element types to match");
  }
  if (getInputType().getRank() == 0) {
    return emitOpError("expected input type to have non-zero rank");
  }
  Region &transformRegion = getTransformationRegion();
  Block &transformBody = transformRegion.getBlocks().front();
  if (transformBody.getNumArguments() != getInputRank()) {
    return emitOpError("expected number of block arguments to be equal "
                       "to the input rank");
  }
  if (!llvm::all_of(transformBody.getArgumentTypes(),
                    llvm::IsaPred<IndexType>)) {
    return emitOpError("expected block arguments to be index types");
  }
  return success();
}

LogicalResult MapStoreOp::verifyRegions() {
  Block &transformBody = getTransformationRegion().getBlocks().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  if (yieldOp->getNumOperands() != getOutputRank() + 1) {
    return yieldOp.emitOpError("expected transformation_region to yield a "
                               "value for each output dimension and a mask");
  }
  for (int operandIdx = 0; operandIdx < getOutputRank(); ++operandIdx) {
    if (!isa<IndexType>(yieldOp.getOperandTypes()[operandIdx])) {
      return yieldOp.emitOpError("expected yielded indices to be index types");
    }
  }
  auto maskType =
      dyn_cast<IntegerType>(yieldOp.getOperandTypes()[getOutputRank()]);
  if (!maskType || maskType.getIntOrFloatBitWidth() != 1) {
    return yieldOp.emitOpError("expected yielded mask to be i1 type");
  }
  return success();
}

Value MapStoreOp::getInputIndex(int64_t position) {
  Block &body = getTransformationRegion().front();
  return body.getArguments()[position];
}

Value MapStoreOp::getOutputIndex(int64_t position) {
  // It shouldn't be possible to return the mask, the last operand of the yield,
  // through this function as that's not an index. Therefore, this assert here.
  assert(position < getOutputRank() &&
         "The output index position being requested should be smaller than the "
         "output rank.");
  Block &body = getTransformationRegion().front();
  auto yield = cast<IREE::LinalgExt::YieldOp>(body.getTerminator());
  return yield.getOperand(position);
}

Value MapStoreOp::getMask() {
  Block &body = getTransformationRegion().front();
  auto yield = cast<IREE::LinalgExt::YieldOp>(body.getTerminator());
  return yield.getOperand(yield.getNumOperands() - 1);
}

void MapStoreOp::insertTransformationAtStart(
    OpBuilder &builder,
    function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
        transformationBuilder,
    int64_t numSourceIndices) {
  Block &transformBody = getTransformationRegion().front();
  SmallVector<BlockArgument> oldSourceIndices(transformBody.getArguments());
  llvm::Repeated<Type> indexTypes(numSourceIndices, builder.getIndexType());
  SmallVector<Location> locs(numSourceIndices, getLoc());

  // Create the new block arguments for the new source indices, and transform
  // them using the callback.
  SmallVector<BlockArgument> newSourceIndices(
      transformBody.addArguments(indexTypes, locs));
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&transformBody);
  SmallVector<Value> newSourceIndicesTransformed(
      transformationBuilder(newSourceIndices));

  // Replace the old source indices with the results of the transformation on
  // the new source indices.
  assert(oldSourceIndices.size() == newSourceIndicesTransformed.size() &&
         "expected transformation to produce the same number of Values as the "
         "previous number of source indices.");
  for (auto [oldIdx, newIdx] :
       llvm::zip_equal(oldSourceIndices, newSourceIndicesTransformed)) {
    SmallVector<OpOperand *> uses(llvm::make_pointer_range(oldIdx.getUses()));
    for (OpOperand *use : uses) {
      use->set(newIdx);
    }
  }
  transformBody.eraseArguments(0, oldSourceIndices.size());
}

void MapStoreOp::inlineMapStoreBody(
    OpBuilder &b, Location loc, ValueRange transformBodyIndices,
    function_ref<void(OpBuilder &, Location, ArrayRef<Value>)> bodyBuilder) {
  inlineMapLoadStoreBodyImpl(b, loc, getTransformationRegion(),
                             transformBodyIndices, bodyBuilder);
}

bool MapStoreOp::isIdentity() {
  if (getInputType() != getOutputType()) {
    return false;
  }

  // Check that the mask is always true.
  Block &transformBody = getTransformationRegion().getBlocks().front();
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(transformBody.getTerminator());
  Value mask = yieldOp->getOperands().back();
  std::optional<int64_t> constMask = getConstantIntValue(mask);
  if (!constMask.has_value() || *constMask == 0) {
    return false;
  }

  // Check that the block arguments are directly yielded in the order that they
  // are defined in the block.
  for (int i = 0; i < getOutputRank(); ++i) {
    auto yieldedBbArg = dyn_cast<BlockArgument>(yieldOp.getOperand(i));
    if (yieldedBbArg != transformBody.getArgument(i)) {
      return false;
    }
  }
  return true;
}

void MapStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *ctx) {
  results.add<ConvertIdentityMapLoadStoreToCopy<MapStoreOp>>(ctx);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

LogicalResult SortOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs()) {
    return op->emitOpError("does not expect to take any inputs");
  }
  if (getNumDpsInits() == 0) {
    return op->emitOpError("expected at least one `outs` operand");
  }

  Block &block = getRegion().front();
  size_t numOutputs = getNumDpsInits();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op->emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int64_t rank = getOperandRank();
  int sortDim = getDimension();
  if (sortDim < 0 || sortDim >= rank) {
    return op->emitOpError("dimension must be within (0, ") << rank << "]";
  }

  ArrayRef<int64_t> shape = getOperandShape();
  for (auto [index, operand] : llvm::enumerate(getOutputs())) {
    auto operandType = getOperandType(index);
    if (operandType.getRank() != rank) {
      return op->emitOpError("expected operand ")
             << index << " to be rank " << rank << ", same as other operands";
    }
    if (operandType.getShape() != shape) {
      return op->emitOpError("expected operand ")
             << index << " to have same shape as other operands";
    }
    Type elemType = operandType.getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op->emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op->emitOpError("should yield exactly one operand");
  }
  auto ty = dyn_cast<IntegerType>(yieldOp.getOperand(0).getType());
  if (!ty || ty.getWidth() != 1) {
    return op->emitOpError("should yield i1 type");
  }

  return success();
}

LogicalResult
SortOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

namespace {

/// This pattern removes unused results from SortOp. The SortOp uses the
/// Destination Passing Style interface so it's results are tied to it's
/// operands as well as it's comparator block arguments. So, to remove unused
/// results we must also remove the associated operands and block arguments.
///
/// For example:
///
/// %0:2 = iree_linalg_ext.sort dimension(1) outs(%arg0, %arg1:
/// tensor<?x10xf32>, tensor<?x10xi64>) {
///   ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
///    %42 = arith.cmpf oge, %arg2, %arg3 : f32
///    iree_linalg_ext.yield %42 : i1
/// } -> tensor<?x10xf32>, tensor<?x10xi64>
///
/// ->
///
/// %0 = iree_linalg_ext.sort dimension(1) outs(%arg0: tensor<?x10xf32>) {
///   ^bb0(%arg2: f32, %arg3: f32):
///    %42 = arith.cmpf oge, %arg2, %arg3 : f32
///    iree_linalg_ext.yield %42 : i1
/// } -> tensor<?x10xf32>
///
/// Note: that we will not remove unused results if their associated block
/// arguments are used within the comparator because that's needed for op
/// functionality.
struct RemoveUnusedSortOpResults : OpRewritePattern<IREE::LinalgExt::SortOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::LinalgExt::SortOp sortOp,
                                PatternRewriter &rewriter) const override {
    // To avoid problems in dispatches associated with unused results, prune
    // them here.
    Location loc = sortOp->getLoc();
    auto operands = sortOp.getOutputs();
    auto results = sortOp.getResults();
    unsigned numRes = sortOp.getNumResults();

    // # TODO(#20831): Add support for removing unused operands when the op has
    // pure buffer semantics.
    if (sortOp.hasPureBufferSemantics()) {
      return failure();
    }

    Block &block = sortOp.getRegion().front();
    auto blockArgs = block.getArguments();
    SmallVector<Value> usedBlockArgs, usedOperands, usedResults;
    SmallVector<Type> usedResultTypes;
    BitVector eraseArg(numRes * 2, false);
    for (auto idx : llvm::seq<unsigned>(numRes)) {
      // If result or associated block arg is used, do not erase.
      if (!results[idx].use_empty() || !blockArgs[2 * idx].use_empty() ||
          !blockArgs[2 * idx + 1].use_empty()) {
        usedOperands.push_back(operands[idx]);
        usedResults.push_back(results[idx]);
        usedResultTypes.push_back(results[idx].getType());
        continue;
      }
      eraseArg.set(2 * idx, 2 * idx + 2);
    }

    // Bail out if no pruning required.
    if (eraseArg.none()) {
      return failure();
    }

    // Create new op using only operands associated to used results or block
    // args.
    auto newSortOp = IREE::LinalgExt::SortOp::create(
        rewriter, loc, usedResultTypes,
        /*inputs=*/ValueRange{}, usedOperands, sortOp.getDimension());
    newSortOp.getRegion().takeBody(sortOp.getRegion());
    newSortOp.getRegion().front().eraseArguments(eraseArg);
    rewriter.replaceAllUsesWith(usedResults, newSortOp.getResults());
    rewriter.eraseOp(sortOp);
    return success();
  }
};
} // namespace

void SortOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *ctx) {
  results.add<RemoveUnusedSortOpResults>(ctx);
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

LogicalResult FftOp::verify() {
  Operation *op = getOperation();
  auto length = getFftLength();
  // After tiling, it could be dynamic shape. (Because
  // subview/subtensor does not inference the type correctly
  // on (1 << x)) cases).
  if (ShapedType::isDynamic(length)) {
    return success();
  }
  if (length & (length - 1)) {
    return op->emitOpError("only powers of 2 are handled currently");
  }
  if (!isScalar(getDpsInputOperand(0))) {
    return op->emitOpError("expected to carry `stage` input");
  }
  return success();
}

LogicalResult
FftOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

MutableOperandRange FftOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/hasCoeff() ? 3 : 1,
                             /*numInits=*/2);
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verify() {
  Operation *op = getOperation();
  auto accumulatorType = cast<ShapedType>(getAccumulator().getType());
  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return op->emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension()) {
      expectedAccumulatorShape.push_back(inputShapes[i]);
    }
  }
  if (llvm::any_of(llvm::zip_equal(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return ShapedType::isStatic(std::get<0>(s)) &&
                            ShapedType::isStatic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip_equal(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return ShapedType::isStatic(std::get<0>(s)) &&
                            ShapedType::isStatic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

LogicalResult ScanOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
ScanOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

ArrayAttr ScanOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  int64_t rank = getOperandRank();

  AffineMap inputOutputMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

  SmallVector<AffineExpr> accumulatorResults;
  accumulatorResults.reserve(rank - 1);
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim == getDimension()) {
      continue;
    }
    accumulatorResults.push_back(getAffineDimExpr(dim, ctx));
  }
  AffineMap accumulatorMap = AffineMap::get(rank, 0, accumulatorResults, ctx);

  Builder b(ctx);
  return b.getAffineMapArrayAttr(
      {inputOutputMap, inputOutputMap, accumulatorMap});
}

MutableOperandRange ScanOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/1, /*numInits=*/2);
}

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

LogicalResult TopkOp::verify() {
  Operation *op = getOperation();
  if (getDimension() >= getInputRank()) {
    return op->emitOpError("dimension exceeds rank");
  }
  // Ensure input/output element types match
  auto inputValuesType = cast<ShapedType>(getValues().getType());
  auto outputValuesType = cast<ShapedType>(getOutputValues().getType());
  if (inputValuesType.getElementType() != outputValuesType.getElementType()) {
    return op->emitOpError("expected input/output value types to be identical");
  }
  // Indices must be int if provided
  auto outputIndicesType = cast<ShapedType>(getOutputIndices().getType());
  if (Value inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices.getType());
    if (!inputIndicesType.getElementType().isInteger(32) ||
        !outputIndicesType.getElementType().isInteger(32)) {
      return op->emitOpError("expected input/output indices types to be int32");
    }
  }

  // Ranks must match
  if (inputValuesType.getRank() != outputValuesType.getRank()) {
    return op->emitOpError("expected input/output to have the same rank");
  }
  if (Value inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices.getType());
    if (inputIndicesType.getRank() != outputIndicesType.getRank()) {
      return op->emitOpError("expected input/output to have the same rank");
    }
  }
  // Input indices and values must have the same shape.
  if (Value inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices.getType());
    if (failed(verifyCompatibleShape(inputValuesType, inputIndicesType))) {
      return op->emitOpError("input indices/values shape must match");
    }
  }
  // Output indices and values must have the same shape.
  if (failed(verifyCompatibleShape(outputValuesType, outputIndicesType))) {
    return op->emitOpError("output indices/values shape must match");
  }
  // Input shape must match the output shape except for the dimension()
  uint64_t dim = getDimension();
  if (!llvm::all_of(
          llvm::enumerate(llvm::zip_equal(inputValuesType.getShape(),
                                          outputValuesType.getShape())),
          [dim](auto e) {
            if (e.index() == dim) {
              return true;
            }
            std::tuple<int64_t, int64_t> s = e.value();
            return succeeded(
                verifyCompatibleShape(std::get<0>(s), std::get<1>(s)));
          })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  // Check region compatibility
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return op->emitOpError("region block should have 2 arguments");
  }
  if (block.getArgument(0).getType() != inputValuesType.getElementType() ||
      block.getArgument(1).getType() != inputValuesType.getElementType()) {
    return op->emitOpError("region block types must match input");
  }
  auto terminatorOp = dyn_cast<YieldOp>(block.getTerminator());
  if (!terminatorOp || !terminatorOp.getOperand(0).getType().isInteger(1)) {
    return op->emitOpError("region block must end with a linalg_ext.yield i1!");
  }
  return success();
}

LogicalResult
TopkOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

MutableOperandRange TopkOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getIndices() ? 2 : 1,
                             /*numInits=*/2);
}

//===----------------------------------------------------------------------===//
// TopkV2Op
//===----------------------------------------------------------------------===//

LogicalResult TopkV2Op::verify() {
  auto inputValuesType = getInputType();
  auto outputValuesType = cast<ShapedType>(getOutputValues().getType());
  uint64_t dim = getDimension();

  if (dim >= static_cast<uint64_t>(getInputRank())) {
    return emitOpError("dimension exceeds rank");
  }
  if (inputValuesType.getElementType() != outputValuesType.getElementType()) {
    return emitOpError("expected input/output value types to be identical");
  }
  if (inputValuesType.getRank() != outputValuesType.getRank()) {
    return emitOpError("expected input/output to have the same rank");
  }

  if (Value inputIndices = getInputIndices()) {
    if (!getOutputIndices()) {
      return emitOpError(
          "input indices require output indices to carry provenance");
    }
    auto inputIndicesType = cast<ShapedType>(inputIndices.getType());
    if (!isa<IntegerType>(inputIndicesType.getElementType())) {
      return emitOpError("expected input indices to be integer type");
    }
    if (failed(verifyCompatibleShape(inputValuesType, inputIndicesType))) {
      return emitOpError("input values/indices shape must match");
    }
    auto outputIndicesType = cast<ShapedType>(getOutputIndices().getType());
    if (inputIndicesType.getElementType() !=
        outputIndicesType.getElementType()) {
      return emitOpError(
          "expected input/output indices element types to be identical");
    }
  }

  if (Value outputIndices = getOutputIndices()) {
    auto outputIndicesType = cast<ShapedType>(outputIndices.getType());
    if (!isa<IntegerType>(outputIndicesType.getElementType())) {
      return emitOpError("expected output indices to be integer type");
    }
    if (failed(verifyCompatibleShape(outputValuesType, outputIndicesType))) {
      return emitOpError("output values/indices shape must match");
    }
  }

  // All dimensions except the sort dimension must match.
  for (auto [idx, inDim, outDim] : llvm::enumerate(
           inputValuesType.getShape(), outputValuesType.getShape())) {
    if (idx == dim) {
      continue;
    }
    if (ShapedType::isStatic(inDim) && ShapedType::isStatic(outDim) &&
        inDim != outDim) {
      return emitOpError("incompatible input/output shapes at dimension ")
             << idx;
    }
  }

  // Validate that output K does not exceed input along the sort dimension.
  int64_t inputDimSize = inputValuesType.getDimSize(dim);
  int64_t outputDimSize = outputValuesType.getDimSize(dim);
  if (ShapedType::isStatic(inputDimSize) &&
      ShapedType::isStatic(outputDimSize)) {
    if (outputDimSize == 0) {
      return emitOpError("output dimension must be positive");
    }
    if (outputDimSize > inputDimSize) {
      return emitOpError("output dimension must not exceed input, got ")
             << outputDimSize << " > " << inputDimSize;
    }
  }

  return success();
}

LogicalResult TopkV2Op::verifyRegions() {
  auto inputValuesType = getInputType();
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return emitOpError("region block should have 2 arguments");
  }
  if (block.getArgument(0).getType() != inputValuesType.getElementType() ||
      block.getArgument(1).getType() != inputValuesType.getElementType()) {
    return emitOpError("region block types must match input value type");
  }
  auto terminatorOp = cast<YieldOp>(block.getTerminator());
  if (terminatorOp.getNumOperands() != 1 ||
      !terminatorOp.getOperand(0).getType().isInteger(1)) {
    return emitOpError("region block must end with a linalg_ext.yield i1");
  }
  return success();
}

LogicalResult
TopkV2Op::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

MutableOperandRange TopkV2Op::getDpsInitsMutable() {
  // Operands order: values, [input_indices], output_values, [output_indices]
  unsigned numInputs = 1 + (getInputIndices() ? 1 : 0);
  unsigned numInits = 1 + (getOutputIndices() ? 1 : 0);
  return MutableOperandRange(*this, numInputs, numInits);
}

//===----------------------------------------------------------------------===//
// ArgCompareOp
//===----------------------------------------------------------------------===//

LogicalResult ArgCompareOp::verify() {
  ShapedType inputValueType = getInputType();
  Type inputValueElemType = inputValueType.getElementType();

  ShapedType outputValueType = getOutputValueType();
  ShapedType outputIndexType = getOutputIndexType();
  Type outputIndexElemType = getOutputIndexElementType();

  if (hasExplicitIndexInput()) {
    ShapedType inputIndexType = getInputIndexType();
    Type inputIndexElemType = getInputIndexElementType();

    if (inputValueType.getShape() != inputIndexType.getShape()) {
      return emitOpError(
                 "explicit-index mode: value and index inputs must have "
                 "the same shape. ")
             << "Value shape: "
             << llvm::interleaved_array(inputValueType.getShape())
             << ", index shape: "
             << llvm::interleaved_array(inputIndexType.getShape());
    }

    if (!isa<IntegerType, IndexType>(inputIndexElemType)) {
      return emitOpError(
                 "explicit-index mode: index input must have integer or index "
                 "element type, but got ")
             << inputIndexElemType;
    }

    if (inputIndexElemType != outputIndexElemType) {
      return emitOpError(
                 "explicit-index mode: input and output index element types "
                 "must match. ")
             << "Input index type: " << inputIndexElemType
             << ", output index type: " << outputIndexElemType;
    }

    if (getIndexBase()) {
      return emitOpError("index_base must not be used with explicit indices");
    }
  }

  Type outputValueElemType = outputValueType.getElementType();
  if (inputValueElemType != outputValueElemType) {
    return emitOpError("input and output value element types must match. ")
           << "Input type: " << inputValueElemType
           << ", output value type: " << outputValueElemType;
  }

  if (!isa<IntegerType, IndexType>(outputIndexElemType)) {
    return emitOpError(
               "output index must have integer or index element type, but got ")
           << outputIndexElemType;
  }

  if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
    return emitOpError("output indices/values shape must match. ")
           << "Output value shape: "
           << llvm::interleaved_array(outputValueType.getShape())
           << ", output index shape: "
           << llvm::interleaved_array(outputIndexType.getShape());
  }

  int64_t dim = getDimension();
  int64_t rank = getInputRank();
  if (dim >= rank) {
    return emitOpError("reduction dimension exceeds or equals input rank. ")
           << "got dimension: " << dim << ", but input rank is: " << rank;
  }

  SmallVector<int64_t> expectedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      expectedShape.push_back(inputValueType.getDimSize(i));
    }
  }
  if (!llvm::equal(expectedShape, outputValueType.getShape())) {
    return emitOpError("output shape must match input shape with reduction "
                       "dimension removed. ")
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: "
           << llvm::interleaved_array(outputValueType.getShape());
  }

  return success();
}

LogicalResult ArgCompareOp::verifyRegions() {
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return emitOpError("region block should have 2 arguments, but got ")
           << block.getNumArguments();
  }

  Type inputValueElemType = getInputType().getElementType();
  Type arg0Type = block.getArgument(0).getType();
  Type arg1Type = block.getArgument(1).getType();
  if (arg0Type != inputValueElemType || arg1Type != inputValueElemType) {
    return emitOpError(
               "comparator arguments must match input value element type. ")
           << "Expected: " << inputValueElemType << ", but got: " << arg0Type
           << " and " << arg1Type;
  }

  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(block.getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return emitOpError(
               "expected linalg_ext.yield to return 1 operand, but got ")
           << yieldOp->getNumOperands();
  }

  if (!yieldOp.getOperand(0).getType().isInteger(1)) {
    return emitOpError(
               "region block must end with a linalg_ext.yield i1, but got: ")
           << yieldOp.getOperand(0).getType();
  }
  return success();
}

LogicalResult ArgCompareOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> IREE::LinalgExt::ArgCompareOp::getIndexingMapsArray() {
  MLIRContext *ctx = getContext();

  const int64_t rank = getInputRank();
  const int64_t redDim = static_cast<int64_t>(getDimension());

  Builder b(ctx);
  AffineMap inputMap = b.getMultiDimIdentityMap(rank);

  SmallVector<AffineExpr> proj;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == redDim) {
      continue;
    }
    proj.push_back(getAffineDimExpr(i, ctx));
  }
  AffineMap resultMap = AffineMap::get(rank, 0, proj, ctx);

  if (hasExplicitIndexInput()) {
    return {inputMap, inputMap, resultMap, resultMap};
  }

  return {inputMap, resultMap, resultMap};
}

SmallVector<AffineMap>
IREE::LinalgExt::ArgCompareOp::getIndexingMapsForOperands() {
  SmallVector<AffineMap> maps = getIndexingMapsArray();
  maps.resize(getNumDpsInputs() + getNumDpsInits());
  return maps;
}

SmallVector<AffineMap>
IREE::LinalgExt::ArgCompareOp::getIndexingMapsForResults() {
  return llvm::to_vector_of<AffineMap>(
      llvm::drop_begin(getIndexingMapsArray(), getNumDpsInputs()));
}

SmallVector<int64_t> IREE::LinalgExt::ArgCompareOp::getStaticLoopRanges() {
  return llvm::to_vector(getInputType().getShape());
}

ArrayAttr IREE::LinalgExt::ArgCompareOp::getIndexingMaps() {
  SmallVector<AffineMap> maps = getIndexingMapsArray();
  return Builder(getContext()).getAffineMapArrayAttr(maps);
}

AffineMap
IREE::LinalgExt::ArgCompareOp::getMatchingIndexingMap(OpOperand *operand) {
  SmallVector<AffineMap> maps = getIndexingMapsArray();
  unsigned idx = operand->getOperandNumber();
  assert(idx < maps.size() &&
         "operand does not have an indexing map (e.g. index_base)");
  return maps[idx];
}

MutableOperandRange ArgCompareOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getInputIndex() ? 2 : 1,
                             /*numInits=*/2);
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static SmallVector<OpFoldResult> getMixedValues(MLIRContext *context,
                                                ArrayRef<int64_t> staticValues,
                                                OperandRange dynamicValues) {
  OpBuilder b(context);
  return mlir::getMixedValues(staticValues, dynamicValues, b);
}

//===----------------------------------------------------------------------===//
// WinogradInputTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradInputTransformOp::verify() {
  Operation *op = getOperation();
  auto inputType = getInputType();
  auto outputType = getOutputType();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 4) {
    return op->emitOpError("expected input operand to have rank either 2 or 4");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    if ((!inputType.isDynamicDim(0) &&
         inputType.getDimSize(0) > getInputTileSize()) ||
        (inputType.isDynamicDim(1) &&
         inputType.getDimSize(1) > getInputTileSize())) {
      return op->emitOpError("expected input dims not greater than input tile "
                             "size if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getInputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError(
          "expected output dims equal to inputTileSize if input is of rank 2");
    }
    return success();
  }

  if (getOutputRank() != getInputRank() + 2) {
    return op->emitOpError(
        "expected output rank to be equal to input rank + 2");
  }
  ArrayRef<int64_t> imageDims = getImageDimensions();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  if (imageDims.size() != 2) {
    return op->emitOpError("expected only 2 image dimensions");
  }
  if (!isNchw() && !isNhwc()) {
    return op->emitOpError(
        "expect image dimensions to be either [1, 2] or [2, 3]");
  }
  SmallVector<int64_t> expectedOutputShape(getOutputRank(), getInputTileSize());
  int outputIndex;
  ArrayRef<int64_t> inputShape = inputType.getShape();
  for (int i = 0; i < inputShape.size(); i++) {
    outputIndex = i + imageDims.size();
    if (ShapedType::isDynamic(inputShape[i])) {
      expectedOutputShape[outputIndex] = inputShape[i];
      continue;
    }
    if (!imageDimsSet.contains(i)) {
      expectedOutputShape[outputIndex] = inputShape[i];
    } else {
      expectedOutputShape[outputIndex] =
          std::ceil(static_cast<float>(inputShape[i] - getKernelSize() + 1) /
                    getOutputTileSize());
    }
  }
  if (isNchw()) {
    permute<Permutation::TTNCHW_TO_TTNHWC>(expectedOutputShape);
  }
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradInputTransformOp::fold(FoldAdaptor,
                                             SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradInputTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WinogradFilterTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradFilterTransformOp::verify() {
  Operation *op = getOperation();
  auto inputType = getInputType();
  auto outputType = getOutputType();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 4) {
    return op->emitOpError("expected input operand to have rank either 2 or 4");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    SmallVector<int64_t> expectedInputShape(2, getKernelSize());
    if (failed(
            verifyCompatibleShape(expectedInputShape, inputType.getShape()))) {
      return op->emitOpError("expected input dims to be equal to kernel size "
                             "if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getInputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError("expected output dims equal to input tile size if "
                             "input is of rank 2");
    }
    return success();
  }

  if (getOutputRank() != getInputRank()) {
    return op->emitOpError("expected output rank to be equal to input rank");
  }
  const ArrayRef<int64_t> kernelDims = getKernelDimensions();
  if (kernelDims.size() != 2) {
    return op->emitOpError("expected only 2 kernel dimensions");
  }
  if (!isHwcf() && !isFchw()) {
    return op->emitOpError(
        "expect kernel dimensions to be either [0, 1] or [2, 3]");
  }
  const int64_t kernelSize = getKernelSize();
  for (auto kernelDim : kernelDims) {
    if (inputType.getDimSize(kernelDim) != kernelSize) {
      return op->emitOpError(
          "expect all kernel dimensions to have the kernel size");
    }
  }
  const int64_t inputTileSize = getInputTileSize();
  SmallVector<int64_t> expectedOutputShape(kernelDims.size(), inputTileSize);
  llvm::SmallSetVector<int64_t, 2> kernelDimsSet(kernelDims.begin(),
                                                 kernelDims.end());
  for (int i = 0; i < inputType.getRank(); i++) {
    if (!kernelDimsSet.contains(i)) {
      expectedOutputShape.push_back(inputType.getDimSize(i));
    }
  }
  if (isFchw()) {
    permute<Permutation::TTFC_TO_TTCF>(expectedOutputShape);
  }
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradFilterTransformOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradFilterTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WinogradOutputTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradOutputTransformOp::verify() {
  Operation *op = getOperation();
  auto inputType = getInputType();
  auto outputType = getOutputType();
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 6) {
    return op->emitOpError("expected input operand to have rank either 2 or 6");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    SmallVector<int64_t> expectedInputShape(2, getInputTileSize());
    if (failed(
            verifyCompatibleShape(expectedInputShape, inputType.getShape()))) {
      return op->emitOpError("expected input dims to be equal to input tile "
                             "size if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getOutputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError("expected output dims equal to output tile size "
                             "if input is of rank 2");
    }
    return success();
  }
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (outputRank != inputRank - 2) {
    return op->emitOpError(
        "expected output rank to be equal to input rank - 2");
  }
  ArrayRef<int64_t> imageDims = getImageDimensions();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  if (imageDims.size() != 2) {
    return op->emitOpError("expected only 2 image dimensions");
  }
  if (!isNchw() && !isNhwc()) {
    return op->emitOpError(
        "expect image dimensions to be either [1, 2] or [2, 3]");
  }
  SmallVector<int64_t> inputShape(inputType.getShape());
  if (isNchw()) {
    permute<Permutation::TTNHWC_TO_TTNCHW>(inputShape);
  }
  SmallVector<int64_t> expectedOutputShape(getOutputRank(), 1);
  int outputIndex;
  for (int i = imageDims.size(); i < inputShape.size(); i++) {
    outputIndex = i - imageDims.size();
    if (ShapedType::isDynamic(inputShape[i])) {
      expectedOutputShape[outputIndex] = inputShape[i];
      continue;
    }
    if (!imageDimsSet.contains(outputIndex)) {
      expectedOutputShape[outputIndex] = inputShape[i];
    } else {
      expectedOutputShape[outputIndex] = getOutputTileSize() * inputShape[i];
    }
  }
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradOutputTransformOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradOutputTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, Value query, Value key, Value value,
                        Value scale, Value output, ArrayAttr indexingMaps,
                        std::optional<Value> mask) {
  Value maskIn = mask.value_or(Value());
  build(odsBuilder, odsState, results, query, key, value, scale, maskIn, output,
        indexingMaps, DictionaryAttr());
}

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, ValueRange inputOperands,
                        ValueRange initOperands, ArrayAttr indexingMaps) {
  assert(inputOperands.size() < 6);
  assert(initOperands.size() == 1);
  Value mask = inputOperands.size() > 4 ? inputOperands[4] : Value();
  build(odsBuilder, odsState, results, inputOperands[0], inputOperands[1],
        inputOperands[2], inputOperands[3], mask, initOperands[0], indexingMaps,
        DictionaryAttr());
}

LogicalResult AttentionOp::verify() {
  AttentionOp attnOp = *this;

  // Check if indexing maps can represent attention.
  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();
  if (indexingMaps.size() != getOperation()->getNumOperands()) {
    return attnOp->emitOpError("expected an indexing map for each operand");
  }
  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  if (failed(maybeOpInfo)) {
    return attnOp->emitOpError("failed to verify op's indexing maps");
  }

  FloatType scaleElementType = dyn_cast<FloatType>(getScale().getType());
  if (!scaleElementType) {
    return attnOp->emitOpError("expected scale to be of floating point type");
  }

  // Check shape compatibility based on indexing maps.
  SmallVector<int64_t> shape(getIterationDomainRank());
  SmallVector<bool> foundDims(getIterationDomainRank(), false);
  auto checkShape = [&shape, &foundDims,
                     &attnOp](StringRef operandName, ArrayRef<int64_t> valShape,
                              AffineMap indexingMap) -> LogicalResult {
    if (indexingMap.getNumResults() != valShape.size()) {
      return attnOp->emitError("Rank Mismatch for ")
             << operandName << ". Expected: " << indexingMap.getNumResults()
             << " Got: " << valShape.size();
    }
    for (auto [i, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      AffineDimExpr dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (ShapedType::isDynamic(valShape[i])) {
        continue;
      }
      if (!foundDims[pos]) {
        foundDims[pos] = true;
        shape[pos] = valShape[i];
      }
      if (shape[pos] != valShape[i]) {
        return attnOp->emitError("Shape Mismatch for ")
               << operandName << " at position " << i
               << ". Expected: " << shape[pos] << " Got: " << valShape[i];
      }
    }
    return success();
  };

  if (failed(checkShape("Query", getQuery().getType().getShape(),
                        getQueryMap())) ||
      failed(checkShape("Key", getKey().getType().getShape(), getKeyMap())) ||
      failed(checkShape("Value", getValue().getType().getShape(),
                        getValueMap())) ||
      failed(checkShape("Output", getOutput().getType().getShape(),
                        getOutputMap()))) {
    return failure();
  }

  // Additional check case if mask exists
  if (auto maskMap = getMaskMap()) {
    if (failed(checkShape("Mask", getMask().getType().getShape(), *maskMap))) {
      return failure();
    }
  }

  int expectedSymbols = getQueryMap().getNumInputs();
  auto checkDomain =
      [&attnOp, &expectedSymbols](StringRef operandName,
                                  AffineMap indexingMap) -> LogicalResult {
    if (expectedSymbols != indexingMap.getNumInputs()) {
      return attnOp->emitError("Mismatched map domain for ")
             << operandName << ". Expected: " << expectedSymbols
             << " Got: " << indexingMap.getNumInputs();
    }
    return success();
  };

  if (failed(checkDomain("Query", getQueryMap())) ||
      failed(checkDomain("Key", getKeyMap())) ||
      failed(checkDomain("Value", getValueMap())) ||
      failed(checkDomain("Scale", getScaleMap())) ||
      failed(checkDomain("Output", getOutputMap()))) {
    return failure();
  }

  // Additional check case if mask exists
  if (auto maskMap = getMaskMap()) {
    if (failed(checkDomain("Mask", *maskMap))) {
      return failure();
    }
  }

  auto &block = getRegion().front();
  auto blockTys = block.getArgumentTypes();
  if (!isa<FloatType>(blockTys[0])) {
    return attnOp->emitOpError("block argument 0 should be float");
  }

  auto yieldOp = dyn_cast<IREE::LinalgExt::YieldOp>(block.getTerminator());
  if (!yieldOp) {
    return attnOp->emitOpError("expected linalg_ext.yield");
  }

  if (yieldOp->getNumOperands() != 1) {
    return emitOpError("expected only one return");
  }

  return success();
}

MutableOperandRange AttentionOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getMask() ? 5 : 4,
                             /*numInits=*/1);
}

LogicalResult AttentionOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

SmallVector<int64_t> AttentionOp::getStaticLoopRanges() {
  SmallVector<int64_t> bounds(getIterationDomainRank());
  SmallVector<bool> dimsFound(getIterationDomainRank(), false);

  // batch(s), m, k1
  ArrayRef<int64_t> queryShape = getQuery().getType().getShape();
  ArrayRef<AffineExpr> queryDims = getQueryMap().getResults();
  // batch(s), k2, n
  ArrayRef<int64_t> valueShape = getValue().getType().getShape();
  ArrayRef<AffineExpr> valueDims = getValueMap().getResults();

  auto fillSizes = [&](ArrayRef<int64_t> sizes, ArrayRef<AffineExpr> dims) {
    for (auto [size, dim] : llvm::zip_equal(sizes, dims)) {
      int pos = cast<AffineDimExpr>(dim).getPosition();
      if (dimsFound[pos]) {
        continue;
      }
      bounds[pos] = size;
      dimsFound[pos] = true;
    }
  };
  fillSizes(queryShape, queryDims);
  fillSizes(valueShape, valueDims);
  return bounds;
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsForOperands() {
  auto maps = getIndexingMapsArray();
  maps.resize(getNumDpsInputs());
  return maps;
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsForResults() {
  return llvm::to_vector_of<AffineMap>(
      llvm::drop_begin(getIndexingMapsArray(), getNumDpsInputs()));
}

void AttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.insert<StaticizeLinalgExtOp<AttentionOp>>(ctx);
}

AffineMap AttentionOp::getMatchingIndexingMap(OpOperand *operand) {
  return *(getIndexingMaps().getAsValueRange<AffineMapAttr>().begin() +
           operand->getOperandNumber());
}

//===----------------------------------------------------------------------===//
// OnlineAttentionOp
//===----------------------------------------------------------------------===//

void OnlineAttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              TypeRange results, Value query, Value key,
                              Value value, Value scale, Value output, Value max,
                              Value sum, ArrayAttr indexingMaps,
                              std::optional<Value> mask) {
  Value maskIn = mask.value_or(Value());
  build(odsBuilder, odsState, results, query, key, value, scale, maskIn, output,
        max, sum, indexingMaps, DictionaryAttr());
}

void OnlineAttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              TypeRange results, ValueRange inputOperands,
                              ValueRange initOperands, ArrayAttr indexingMaps) {
  assert(inputOperands.size() < 6);
  assert(initOperands.size() == 3);
  Value mask = inputOperands.size() > 4 ? inputOperands[4] : Value();
  build(odsBuilder, odsState, results, inputOperands[0], inputOperands[1],
        inputOperands[2], inputOperands[3], mask, initOperands[0],
        initOperands[1], initOperands[2], indexingMaps, DictionaryAttr());
}

LogicalResult OnlineAttentionOp::verify() {
  OnlineAttentionOp attnOp = *this;

  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();

  // Check if indexing maps can represent attention.
  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());

  // Check shape compatibility based on indexing maps.
  SmallVector<int64_t> shape(getIterationDomainRank());
  SmallVector<bool> foundDims(getIterationDomainRank(), false);
  auto checkShape = [&shape, &foundDims,
                     &attnOp](StringRef operandName, ArrayRef<int64_t> valShape,
                              AffineMap indexingMap) -> LogicalResult {
    if (indexingMap.getNumResults() != valShape.size()) {
      return attnOp->emitError("Rank Mismatch for ")
             << operandName << ". Expected: " << indexingMap.getNumResults()
             << " Got: " << valShape.size();
    }
    for (auto [i, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      AffineDimExpr dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (ShapedType::isDynamic(valShape[i])) {
        continue;
      }
      if (!foundDims[pos]) {
        foundDims[pos] = true;
        shape[pos] = valShape[i];
      }
      if (shape[pos] != valShape[i]) {
        return attnOp->emitError("Shape Mismatch for ")
               << operandName << ". Expected: " << shape[pos]
               << " Got: " << valShape[i];
      }
    }
    return success();
  };

  if (failed(checkShape("Query", getQuery().getType().getShape(),
                        getQueryMap())) ||
      failed(checkShape("Key", getKey().getType().getShape(), getKeyMap())) ||
      failed(checkShape("Value", getValue().getType().getShape(),
                        getValueMap())) ||
      failed(checkShape("Output", getOutput().getType().getShape(),
                        getOutputMap())) ||
      failed(checkShape("Max", getMax().getType().getShape(), getMaxMap())) ||
      failed(checkShape("Sum", getSum().getType().getShape(), getSumMap()))) {
    return failure();
  }

  // Additional check case if mask exists
  if (auto maskMap = getMaskMap()) {
    if (failed(checkShape("Mask", getMask().getType().getShape(), *maskMap))) {
      return failure();
    }
  }

  int expectedSymbols = getQueryMap().getNumInputs();
  auto checkDomain =
      [&attnOp, &expectedSymbols](StringRef operandName,
                                  AffineMap indexingMap) -> LogicalResult {
    if (expectedSymbols != indexingMap.getNumInputs()) {
      return attnOp->emitError("Mismatched map domain for ")
             << operandName << ". Expected: " << expectedSymbols
             << " Got: " << indexingMap.getNumInputs();
    }
    return success();
  };

  if (failed(checkDomain("Query", getQueryMap())) ||
      failed(checkDomain("Key", getKeyMap())) ||
      failed(checkDomain("Value", getValueMap())) ||
      failed(checkDomain("Scale", getScaleMap())) ||
      failed(checkDomain("Output", getOutputMap())) ||
      failed(checkDomain("Max", getMaxMap())) ||
      failed(checkDomain("Sum", getSumMap()))) {
    return failure();
  }

  // Additional check case if mask exists
  if (auto maskMap = getMaskMap()) {
    if (failed(checkDomain("Mask", *maskMap))) {
      return failure();
    }
  }

  Block &block = attnOp.getRegion().front();
  auto blockTys = block.getArgumentTypes();
  if (blockTys.size() != 1) {
    return attnOp->emitOpError("expects single block argument for score");
  }

  if (!isa<FloatType>(blockTys[0])) {
    return attnOp->emitOpError("block argument 0 should be float");
  }

  auto yieldOp = dyn_cast<IREE::LinalgExt::YieldOp>(block.getTerminator());
  if (!yieldOp) {
    return attnOp->emitOpError("expected linalg_ext.yield");
  }

  if (yieldOp->getNumOperands() != 1) {
    return emitOpError("expected only one return");
  }

  return success();
}

MutableOperandRange OnlineAttentionOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getMask() ? 5 : 4,
                             /*numInits=*/3);
}

LogicalResult OnlineAttentionOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

SmallVector<AffineMap> OnlineAttentionOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

SmallVector<int64_t> OnlineAttentionOp::getStaticLoopRanges() {
  SmallVector<int64_t> bounds(getIterationDomainRank());
  SmallVector<bool> dimsFound(getIterationDomainRank(), false);

  ArrayRef<int64_t> queryShape = getQuery().getType().getShape();
  ArrayRef<AffineExpr> queryDims = getQueryMap().getResults();
  ArrayRef<int64_t> valueShape = getValue().getType().getShape();
  ArrayRef<AffineExpr> valueDims = getValueMap().getResults();

  auto fillSizes = [&](ArrayRef<int64_t> sizes, ArrayRef<AffineExpr> dims) {
    for (auto [size, dim] : llvm::zip_equal(sizes, dims)) {
      int pos = cast<AffineDimExpr>(dim).getPosition();
      if (dimsFound[pos]) {
        continue;
      }
      bounds[pos] = size;
      dimsFound[pos] = true;
    }
  };
  fillSizes(queryShape, queryDims);
  fillSizes(valueShape, valueDims);
  return bounds;
}

SmallVector<AffineMap> OnlineAttentionOp::getIndexingMapsForOperands() {
  auto maps = getIndexingMapsArray();
  maps.resize(getNumDpsInputs());
  return maps;
}

SmallVector<AffineMap> OnlineAttentionOp::getIndexingMapsForResults() {
  return llvm::to_vector_of<AffineMap>(
      llvm::drop_begin(getIndexingMapsArray(), getNumDpsInputs()));
}

void OnlineAttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx) {
  patterns.insert<StaticizeLinalgExtOp<OnlineAttentionOp>>(ctx);
}

//===----------------------------------------------------------------------===//
// ExpReductionOp
//===----------------------------------------------------------------------===//

LogicalResult ExpReductionOp::verify() {
  Operation *op = getOperation();

  for (int64_t reducedOperand : getExpReducedOperands()) {
    if (reducedOperand == 0) {
      return op->emitOpError(
          "operand index in exp_reduced_operands cannot be the 0th operand, it "
          "always contains the maximum value");
    }
    if (reducedOperand >= getNumDpsInits()) {
      return op->emitOpError("operand index in exp_reduced_operands must index "
                             "the outs operands ("
                             "outs has ")
             << getNumDpsInits() << " operands)";
    }
  }

  if (!allIndexingsAreProjectedPermutation(*this)) {
    return op->emitOpError("all indexing maps must be projected permutations");
  }

  if (!getBody()->getOps<linalg::IndexOp>().empty()) {
    return op->emitOpError("linalg.index is not supported in body");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Im2colOp
//===----------------------------------------------------------------------===//

// Custom printer for the nested dynamic index list used by output_sizes.
// Prints format: [[2], [%oh, %ow], [3, 3, 640]]. Dynamic positions use SSA
// values; static positions print integer literals. ShapedType::kDynamic in
// the static arrays marks dynamic positions. MLIR's parseDynamicIndexList
// handles only flat lists, not the nested [[...], [...]] structure here.
static void printNestedDynamicIndexList(OpAsmPrinter &p, Operation *op,
                                        OperandRange dynamicValues,
                                        ArrayAttr staticOutputSizes) {
  int64_t dynamicIdx = 0;
  p << "[";
  llvm::interleaveComma(staticOutputSizes, p, [&](Attribute innerAttr) {
    auto innerArray = cast<DenseI64ArrayAttr>(innerAttr);
    p << "[";
    llvm::interleaveComma(innerArray.asArrayRef(), p, [&](int64_t val) {
      if (ShapedType::isDynamic(val)) {
        p << dynamicValues[dynamicIdx++];
      } else {
        p << val;
      }
    });
    p << "]";
  });
  p << "]";
}

// Custom parser for nested dynamic index lists.
// Parses format: [[2], [%oh, %ow], [3, 3, 640]]
// Returns the flat list of dynamic SSA values and the ArrayAttr of
// DenseI64ArrayAttr with ShapedType::kDynamic as sentinel for dynamic
// positions.
static ParseResult parseNestedDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicValues,
    ArrayAttr &staticOutputSizes) {
  SmallVector<Attribute> innerArrayAttrs;
  if (parser.parseLSquare()) {
    return failure();
  }

  auto parseInnerList = [&]() -> ParseResult {
    SmallVector<int64_t> staticVals;
    if (parser.parseLSquare()) {
      return failure();
    }

    auto parseElement = [&]() -> ParseResult {
      // Try to parse an SSA value (dynamic).
      OpAsmParser::UnresolvedOperand operand;
      auto res = parser.parseOptionalOperand(operand);
      if (res.has_value()) {
        if (failed(res.value())) {
          return failure();
        }
        dynamicValues.push_back(operand);
        staticVals.push_back(ShapedType::kDynamic);
        return success();
      }
      // Otherwise parse a static integer.
      int64_t val;
      if (parser.parseInteger(val)) {
        return failure();
      }
      staticVals.push_back(val);
      return success();
    };

    if (parser.parseCommaSeparatedList(parseElement)) {
      return failure();
    }

    if (parser.parseRSquare()) {
      return failure();
    }
    innerArrayAttrs.push_back(
        DenseI64ArrayAttr::get(parser.getContext(), staticVals));
    return success();
  };

  if (parser.parseCommaSeparatedList(parseInnerList)) {
    return failure();
  }

  if (parser.parseRSquare()) {
    return failure();
  }
  staticOutputSizes = ArrayAttr::get(parser.getContext(), innerArrayAttrs);
  return success();
}

/// Return all static and dynamic kernel_size as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedKernelSize() {
  return LinalgExt::getMixedValues(getContext(), getStaticKernelSize(),
                                   getKernelSize());
}

/// Return all static and dynamic offsets as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedOffsets() {
  return LinalgExt::getMixedValues(getContext(), getStaticOffsets(),
                                   getOffsets());
}

/// Return the nested output_sizes as a vector of vectors of OpFoldResults.
SmallVector<SmallVector<OpFoldResult>> Im2colOp::getMixedOutputSizes() {
  SmallVector<SmallVector<OpFoldResult>> result;
  ArrayAttr sizesAttr = getStaticOutputSizes();
  auto dynamicVals = getOutputSizes();
  int64_t dynamicIdx = 0;
  for (Attribute innerAttr : sizesAttr) {
    auto innerArray = cast<DenseI64ArrayAttr>(innerAttr);
    SmallVector<OpFoldResult> innerResult;
    for (int64_t val : innerArray.asArrayRef()) {
      if (ShapedType::isDynamic(val)) {
        innerResult.push_back(dynamicVals[dynamicIdx++]);
      } else {
        innerResult.push_back(
            IntegerAttr::get(IndexType::get(getContext()), val));
      }
    }
    result.push_back(std::move(innerResult));
  }
  return result;
}

/// Return all static and dynamic input_pad_low as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedInputPadLow() {
  return LinalgExt::getMixedValues(getContext(), getStaticInputPadLow(),
                                   getInputPadLow());
}

/// Return all static and dynamic input_pad_high as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedInputPadHigh() {
  return LinalgExt::getMixedValues(getContext(), getStaticInputPadHigh(),
                                   getInputPadHigh());
}

/// Return all static and dynamic output_pad_low as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedOutputPadLow() {
  return LinalgExt::getMixedValues(getContext(), getStaticOutputPadLow(),
                                   getOutputPadLow());
}

/// Return all static and dynamic output_pad_high as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedOutputPadHigh() {
  return LinalgExt::getMixedValues(getContext(), getStaticOutputPadHigh(),
                                   getOutputPadHigh());
}

void Im2colOp::setMixedOffsets(SmallVector<OpFoldResult> newOffsets) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(newOffsets, dynamicOffsets, staticOffsets);
  setStaticOffsets(staticOffsets);
  getOffsetsMutable().assign(dynamicOffsets);
}

static std::pair<SmallVector<Attribute>, SmallVector<Value>>
dispatchNestedOutputSizes(MLIRContext *ctx,
                          ArrayRef<SmallVector<OpFoldResult>> outputSizes) {
  SmallVector<Attribute> innerArrayAttrs;
  SmallVector<Value> dynamicValues;
  for (const auto &innerSizes : outputSizes) {
    SmallVector<int64_t> staticVals;
    for (auto ofr : innerSizes) {
      if (auto val = getConstantIntValue(ofr)) {
        staticVals.push_back(*val);
      } else {
        staticVals.push_back(ShapedType::kDynamic);
        dynamicValues.push_back(cast<Value>(ofr));
      }
    }
    innerArrayAttrs.push_back(DenseI64ArrayAttr::get(ctx, staticVals));
  }
  return {innerArrayAttrs, dynamicValues};
}

void Im2colOp::setMixedOutputSizes(
    ArrayRef<SmallVector<OpFoldResult>> outputSizes) {
  auto [innerArrayAttrs, dynamicValues] =
      dispatchNestedOutputSizes(getContext(), outputSizes);
  setStaticOutputSizesAttr(ArrayAttr::get(getContext(), innerArrayAttrs));
  getOutputSizesMutable().assign(dynamicValues);
}

void Im2colOp::setMixedInputPadLow(SmallVector<OpFoldResult> padLow) {
  SmallVector<int64_t> staticPadLow;
  SmallVector<Value> dynamicPadLow;
  dispatchIndexOpFoldResults(padLow, dynamicPadLow, staticPadLow);
  setStaticInputPadLow(staticPadLow);
  getInputPadLowMutable().assign(dynamicPadLow);
}

void Im2colOp::setMixedInputPadHigh(SmallVector<OpFoldResult> padHigh) {
  SmallVector<int64_t> staticPadHigh;
  SmallVector<Value> dynamicPadHigh;
  dispatchIndexOpFoldResults(padHigh, dynamicPadHigh, staticPadHigh);
  setStaticInputPadHigh(staticPadHigh);
  getInputPadHighMutable().assign(dynamicPadHigh);
}

void Im2colOp::setMixedOutputPadLow(SmallVector<OpFoldResult> padLow) {
  SmallVector<int64_t> staticPadLow;
  SmallVector<Value> dynamicPadLow;
  dispatchIndexOpFoldResults(padLow, dynamicPadLow, staticPadLow);
  setStaticOutputPadLow(staticPadLow);
  getOutputPadLowMutable().assign(dynamicPadLow);
}

void Im2colOp::setMixedOutputPadHigh(SmallVector<OpFoldResult> padHigh) {
  SmallVector<int64_t> staticPadHigh;
  SmallVector<Value> dynamicPadHigh;
  dispatchIndexOpFoldResults(padHigh, dynamicPadHigh, staticPadHigh);
  setStaticOutputPadHigh(staticPadHigh);
  getOutputPadHighMutable().assign(dynamicPadHigh);
}

bool Im2colOp::hasOutputPadding() {
  SmallVector<OpFoldResult> outPadLow = getMixedOutputPadLow();
  SmallVector<OpFoldResult> outPadHigh = getMixedOutputPadHigh();
  if (outPadLow.empty()) {
    return false;
  }
  for (auto pad : outPadLow) {
    if (!isConstantIntValue(pad, 0)) {
      return true;
    }
  }
  for (auto pad : outPadHigh) {
    if (!isConstantIntValue(pad, 0)) {
      return true;
    }
  }
  return false;
}

SmallVector<int64_t> Im2colOp::getBatchOutputDims() {
  SmallVector<int64_t> inverseOutPerm =
      invertPermutationVector(getOutputPerm());
  return llvm::map_to_vector(llvm::seq<int64_t>(0, getBatchPos().size()),
                             [&](int64_t dim) { return inverseOutPerm[dim]; });
}

int64_t Im2colOp::getNumMOutputDims() {
  ArrayAttr sizesAttr = getStaticOutputSizes();
  int64_t batchSize = getBatchPos().size();
  int64_t mTarget = getMPos().size();
  int64_t accumulated = 0;
  int64_t numDims = static_cast<int64_t>(sizesAttr.size());
  for (int64_t i = batchSize; i < numDims; ++i) {
    auto innerSizes = cast<DenseI64ArrayAttr>(sizesAttr[i]);
    accumulated += innerSizes.size();
    if (accumulated == mTarget) {
      return i - batchSize + 1;
    }
  }
  assert(false &&
         "M/K boundary not found in output_sizes; verifier should have caught "
         "this");
  return 0;
}

SmallVector<int64_t> Im2colOp::getMOutputDims() {
  int64_t begin = getBatchPos().size();
  int64_t end = begin + getNumMOutputDims();
  SmallVector<int64_t> inverseOutPerm =
      invertPermutationVector(getOutputPerm());
  return llvm::map_to_vector(llvm::seq<int64_t>(begin, end),
                             [&](int64_t dim) { return inverseOutPerm[dim]; });
}

SmallVector<int64_t> Im2colOp::getKOutputDims() {
  int64_t begin = getBatchPos().size() + getNumMOutputDims();
  int64_t end = getOutputRank();
  SmallVector<int64_t> inverseOutPerm =
      invertPermutationVector(getOutputPerm());
  return llvm::map_to_vector(llvm::seq<int64_t>(begin, end),
                             [&](int64_t dim) { return inverseOutPerm[dim]; });
}

SmallVector<SmallVector<int64_t>>
Im2colOp::getInputToOutputDimVectorizationMap() {
  SetVector<int64_t> batchInputDims(llvm::from_range, getBatchPos());
  SetVector<int64_t> mInputDims(llvm::from_range, getMPos());
  SetVector<int64_t> kInputDims(llvm::from_range, getKPos());
  SmallVector<SmallVector<int64_t>> vectorizationMap;
  vectorizationMap.resize(getInputRank());
  SmallVector<int64_t> batchOutputDims = getBatchOutputDims();
  if (batchInputDims.size() == batchOutputDims.size()) {
    for (auto [batchInputDim, batchOutputDim] :
         llvm::zip_equal(batchInputDims, batchOutputDims)) {
      vectorizationMap[batchInputDim] = {batchOutputDim};
    }
  } else {
    vectorizationMap[batchInputDims.back()].push_back(batchOutputDims.back());
  }
  SmallVector<int64_t> mOutputDims = getMOutputDims();
  if (mInputDims.size() == mOutputDims.size()) {
    for (auto [mInputDim, mOutputDim] :
         llvm::zip_equal(mInputDims, mOutputDims)) {
      vectorizationMap[mInputDim] = {mOutputDim};
    }
  } else {
    vectorizationMap[mInputDims.back()].push_back(mOutputDims.back());
  }
  SmallVector<int64_t> kOutputDims = getKOutputDims();
  // Compute the input dimensions captured by the K output dimensions, in the
  // order that they appear in kOutputDims. The canonical order for the input
  // K dims is just the order that they appear in the input tensor. Get the
  // dims in this order, and then apply the input_k_perm to get the dims in the
  // order that they are represented in the output tensor.
  SmallVector<int64_t> orderedKInputDims;
  for (int64_t dim = 0; dim < getInputRank(); ++dim) {
    if (batchInputDims.contains(dim)) {
      continue;
    }
    if (kInputDims.contains(dim)) {
      orderedKInputDims.push_back(dim);
      continue;
    }
    orderedKInputDims.push_back(dim);
  }
  applyPermutationToVector(orderedKInputDims, getInputKPerm());
  if (orderedKInputDims.size() == kOutputDims.size()) {
    for (auto [kInputDim, kOutputDim] :
         llvm::zip_equal(orderedKInputDims, kOutputDims)) {
      vectorizationMap[kInputDim].push_back(kOutputDim);
    }
  } else {
    vectorizationMap[orderedKInputDims.back()].push_back(kOutputDims.back());
  }
  return vectorizationMap;
}

/// Custom builder methods for im2col op.
void Im2colOp::build(
    OpBuilder &builder, OperationState &state, Value input, Value output,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
    ArrayRef<OpFoldResult> kernelSize, ArrayRef<OpFoldResult> offsets,
    ArrayRef<SmallVector<OpFoldResult>> outputSizes, ArrayRef<int64_t> batchPos,
    ArrayRef<int64_t> mPos, ArrayRef<int64_t> kPos,
    ArrayRef<int64_t> inputKPerm, ArrayRef<int64_t> outputPerm,
    ArrayRef<OpFoldResult> inputPadLow, ArrayRef<OpFoldResult> inputPadHigh,
    ArrayRef<OpFoldResult> outputPadLow, ArrayRef<OpFoldResult> outputPadHigh,
    Value padValue) {
  assert(strides.size() == kernelSize.size() &&
         dilations.size() == kernelSize.size() &&
         mPos.size() == kernelSize.size() &&
         "strides, dilations, m_pos, and kernel expected to be the same rank");
  SmallVector<int64_t> staticKernelSize, staticOffsets;
  SmallVector<Value> dynamicKernelSize, dynamicOffsets;
  dispatchIndexOpFoldResults(kernelSize, dynamicKernelSize, staticKernelSize);
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  // Build the nested ArrayAttr of DenseI64ArrayAttr for output_sizes.
  auto [innerArrayAttrs, dynamicOutputSizes] =
      dispatchNestedOutputSizes(builder.getContext(), outputSizes);
  ArrayAttr staticOutputSizesAttr =
      ArrayAttr::get(builder.getContext(), innerArrayAttrs);

  SmallVector<int64_t> staticInputPadLow, staticInputPadHigh;
  SmallVector<Value> dynamicInputPadLow, dynamicInputPadHigh;
  dispatchIndexOpFoldResults(inputPadLow, dynamicInputPadLow,
                             staticInputPadLow);
  dispatchIndexOpFoldResults(inputPadHigh, dynamicInputPadHigh,
                             staticInputPadHigh);

  SmallVector<int64_t> staticOutputPadLow, staticOutputPadHigh;
  SmallVector<Value> dynamicOutputPadLow, dynamicOutputPadHigh;
  dispatchIndexOpFoldResults(outputPadLow, dynamicOutputPadLow,
                             staticOutputPadLow);
  dispatchIndexOpFoldResults(outputPadHigh, dynamicOutputPadHigh,
                             staticOutputPadHigh);

  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, input, output,
        builder.getDenseI64ArrayAttr(strides),
        builder.getDenseI64ArrayAttr(dilations), dynamicKernelSize,
        builder.getDenseI64ArrayAttr(staticKernelSize), dynamicOffsets,
        builder.getDenseI64ArrayAttr(staticOffsets), dynamicOutputSizes,
        staticOutputSizesAttr, builder.getDenseI64ArrayAttr(batchPos),
        builder.getDenseI64ArrayAttr(mPos), builder.getDenseI64ArrayAttr(kPos),
        builder.getDenseI64ArrayAttr(inputKPerm),
        builder.getDenseI64ArrayAttr(outputPerm), dynamicInputPadLow,
        builder.getDenseI64ArrayAttr(staticInputPadLow), dynamicInputPadHigh,
        builder.getDenseI64ArrayAttr(staticInputPadHigh), dynamicOutputPadLow,
        builder.getDenseI64ArrayAttr(staticOutputPadLow), dynamicOutputPadHigh,
        builder.getDenseI64ArrayAttr(staticOutputPadHigh), padValue);
}

LogicalResult Im2colOp::verify() {
  Operation *op = getOperation();
  if (llvm::count_if(getDpsInputs(), [](Value v) {
        return isa<ShapedType>(v.getType());
      }) != 1) {
    return op->emitOpError("expected only one ShapedType operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }

  // Verify operand ranks and dim position sizes.
  auto inputType = getInputType();
  unsigned inputRank = inputType.getRank();
  ArrayRef<int64_t> batchPos = getBatchPos();
  ArrayRef<int64_t> mPos = getMPos();
  ArrayRef<int64_t> kPos = getKPos();
  if (inputRank != batchPos.size() + mPos.size() + kPos.size()) {
    return op->emitOpError(
        "expected input rank to be the sum of batch, m, and k ranks");
  }
  auto outputType = getOutputType();
  unsigned outputRank = outputType.getRank();

  // Verify offsets and output_sizes.
  SmallVector<OpFoldResult> mixedOffsets = getMixedOffsets();
  ArrayAttr sizesAttr = getStaticOutputSizes();
  if (mixedOffsets.size() != outputRank) {
    return op->emitOpError("expected offsets size (")
           << mixedOffsets.size() << ") to match output rank (" << outputRank
           << ")";
  }
  if (static_cast<unsigned>(sizesAttr.size()) != outputRank) {
    return op->emitOpError("expected output_sizes outer size (")
           << sizesAttr.size() << ") to match output rank (" << outputRank
           << ")";
  }

  // Verify inner sizes for each output dim type (Batch, M, K) separately.
  // Output dims in canonical order are: [Batch..., M..., K...].
  //
  // Batch output dims: each produces 1 coordinate (batch index).
  //   -> total inner sizes across all batch dims = batchPos.size()
  //
  // M output dims: collectively produce mPos.size() coordinates
  //   (spatial output positions, one per spatial dimension).
  //   -> total inner sizes across all M dims = mPos.size()
  //
  // K output dims: collectively produce (mPos.size() + kPos.size())
  //   coordinates (kernel window offsets + channel positions,
  //   indexed by input_k_perm).
  //   -> total inner sizes across all K dims = mPos.size() + kPos.size()
  int64_t numBatchOutputDims = batchPos.size();
  int64_t expectedBatchInner = batchPos.size();
  int64_t expectedMInner = mPos.size();
  int64_t expectedKInner = mPos.size() + kPos.size();

  // Count batch inner dims.
  int64_t batchInnerTotal = 0;
  for (int64_t i = 0; i < numBatchOutputDims; ++i) {
    batchInnerTotal += cast<DenseI64ArrayAttr>(sizesAttr[i]).size();
  }
  if (batchInnerTotal != expectedBatchInner) {
    return op->emitOpError(
        "expected sum of batch output_sizes inner dimensions to equal "
        "batch_pos size");
  }

  // Find M/K boundary and count M inner dims.
  int64_t mInnerTotal = 0;
  int64_t numMOutputDims = 0;
  for (int64_t i = numBatchOutputDims;
       i < static_cast<int64_t>(sizesAttr.size()); ++i) {
    mInnerTotal += cast<DenseI64ArrayAttr>(sizesAttr[i]).size();
    if (mInnerTotal == expectedMInner) {
      numMOutputDims = i - numBatchOutputDims + 1;
      break;
    }
  }
  if (numMOutputDims == 0) {
    return op->emitOpError(
        "M/K boundary not found: accumulated output_sizes inner dimensions "
        "must equal m_pos size at some output dim boundary");
  }

  // Count K inner dims.
  int64_t kInnerTotal = 0;
  for (int64_t i = numBatchOutputDims + numMOutputDims;
       i < static_cast<int64_t>(sizesAttr.size()); ++i) {
    kInnerTotal += cast<DenseI64ArrayAttr>(sizesAttr[i]).size();
  }
  if (kInnerTotal != expectedKInner) {
    return op->emitOpError("expected sum of K output_sizes inner dimensions (")
           << kInnerTotal << ") to equal m_pos.size() + k_pos.size() ("
           << expectedKInner << ")";
  }

  // Verify convolution metadata.
  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> dilations = getDilations();
  SmallVector<OpFoldResult> kernelSize = getMixedKernelSize();
  ArrayRef<int64_t> inputKPerm = getInputKPerm();
  if (kernelSize.size() != mPos.size()) {
    return op->emitOpError(
        "expected kernel rank to be equal to the m_pos rank");
  }
  if (strides.size() != kernelSize.size()) {
    return op->emitOpError(
        "expected strides rank to be equal to the kernel rank");
  }
  if (dilations.size() != kernelSize.size()) {
    return op->emitOpError(
        "expected dilations rank to be equal to the kernel rank");
  }

  int64_t sharedRank = mPos.size() + kPos.size();
  if (inputKPerm.size() != sharedRank) {
    return op->emitOpError("expected input_k_perm size (")
           << inputKPerm.size()
           << ") to match the number of shared dimensions (m_pos + k_pos = "
           << sharedRank << ")";
  }
  SmallVector<int64_t> permVec(inputKPerm);
  llvm::sort(permVec);
  for (int64_t i = 0; i < static_cast<int64_t>(sharedRank); ++i) {
    if (permVec[i] != i) {
      return op->emitOpError(
                 "expected input_k_perm to be a permutation of [0, ")
             << sharedRank << ")";
    }
  }

  // Verify input and output shapes.
  ArrayRef<int64_t> outputShape = outputType.getShape();
  ArrayRef<int64_t> outputPerm = getOutputPerm();
  if (!isPermutationVector(outputPerm)) {
    return op->emitOpError("expected output_perm to be a permutation");
  }
  if (outputPerm.size() != outputShape.size()) {
    return op->emitOpError(
        "expected output_perm to have the same rank as the result");
  }

  // Verify padding attributes. This must happen before the batch-dim shape
  // check below, which indexes into padLow/padHigh.
  SmallVector<OpFoldResult> padLow = getMixedInputPadLow();
  SmallVector<OpFoldResult> padHigh = getMixedInputPadHigh();
  Value padVal = getPadValue();

  // pad_low and pad_high must have the same size.
  if (padLow.size() != padHigh.size()) {
    return op->emitOpError(
        "expected input_pad_low and input_pad_high to have the same size");
  }

  // If either pad_low or pad_high is non-empty, they must match input rank.
  if (!padLow.empty() && padLow.size() != inputRank) {
    return op->emitOpError("expected input_pad_low size (")
           << padLow.size() << ") to match input rank (" << inputRank << ")";
  }

  // If padding sizes are non-empty, pad_value must be present.
  if (!padLow.empty() && !padVal) {
    return op->emitOpError(
        "expected pad_value when input_pad_low/input_pad_high are specified");
  }

  // pad_value can exist without input padding. It indicates that some type
  // of padding is present (input or output), and specifies the value to use
  // for out-of-bounds positions.

  // Verify output padding attributes.
  SmallVector<OpFoldResult> outPadLow = getMixedOutputPadLow();
  SmallVector<OpFoldResult> outPadHigh = getMixedOutputPadHigh();

  // output_pad_low and output_pad_high must have the same size.
  if (outPadLow.size() != outPadHigh.size()) {
    return op->emitOpError(
        "expected output_pad_low and output_pad_high to have the same size");
  }

  // If either is non-empty, they must match the output rank.
  if (!outPadLow.empty() &&
      outPadLow.size() != static_cast<size_t>(outputRank)) {
    return op->emitOpError("expected output_pad_low size (")
           << outPadLow.size() << ") to match output rank (" << outputRank
           << ")";
  }

  // If output padding is non-empty, pad_value must be present.
  if (!outPadLow.empty() && !padVal) {
    return op->emitOpError(
        "expected pad_value when output_pad_low/output_pad_high are specified");
  }

  // Note: batch dim shapes between input and output are intentionally NOT
  // verified. With offset-based batch indexing, the output batch dim can be:
  //   - smaller (tiled: output is a tile, input is the full tensor)
  //   - equal (untiled case)
  //   - larger (output padding for alignment, e.g. tile=64, actual=56)
  // The output_sizes attribute encodes the valid batch region; padding and
  // bounds checking are handled by computeIm2colValidSize.

  return success();
}

namespace {

/// Fold tensor.pad on the input of im2col into the im2col's padding
/// attributes.
///
/// %padded = tensor.pad %input low[...] high[...] { yield %cst }
/// %result = im2col ins(%padded) outs(%out) ...
/// -->
/// %result = im2col ins(%input) outs(%out) ...
///     input_pad_low=[...] input_pad_high=[...] pad_value(%cst : type)
struct FoldInputPadIntoIm2col final : public OpRewritePattern<Im2colOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = im2colOp.getInput().getDefiningOp<tensor::PadOp>();
    if (!padOp) {
      return rewriter.notifyMatchFailure(im2colOp,
                                         "input not produced by tensor.pad");
    }

    // Only fold constant padding values.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      return rewriter.notifyMatchFailure(im2colOp,
                                         "pad value is not a constant");
    }

    // If the im2col already has padding, the pad values must be compatible
    // (same constant value) for the fold to be valid, since we can only have
    // one pad_value on the im2col.
    if (im2colOp.hasPadding()) {
      auto existingConst =
          im2colOp.getPadValue().getDefiningOp<arith::ConstantOp>();
      auto newConst = padValue.getDefiningOp<arith::ConstantOp>();
      if (!existingConst || !newConst ||
          existingConst.getValue() != newConst.getValue()) {
        return rewriter.notifyMatchFailure(
            im2colOp, "pad values are not compatible constants");
      }
      padValue = im2colOp.getPadValue();
    }

    Location loc = im2colOp.getLoc();
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();

    // If im2col already has input padding, compose by adding element-wise.
    SmallVector<OpFoldResult> existingLow = im2colOp.getMixedInputPadLow();
    SmallVector<OpFoldResult> existingHigh = im2colOp.getMixedInputPadHigh();
    if (!existingLow.empty()) {
      for (auto [i, e] : llvm::enumerate(existingLow)) {
        lowPad[i] = addOfrs(rewriter, loc, e, lowPad[i]);
      }
      for (auto [i, e] : llvm::enumerate(existingHigh)) {
        highPad[i] = addOfrs(rewriter, loc, e, highPad[i]);
      }
    }

    auto newIm2col = Im2colOp::create(
        rewriter, loc, padOp.getSource(), im2colOp.getOutput(),
        im2colOp.getStrides(), im2colOp.getDilations(),
        im2colOp.getMixedKernelSize(), im2colOp.getMixedOffsets(),
        im2colOp.getMixedOutputSizes(), im2colOp.getBatchPos(),
        im2colOp.getMPos(), im2colOp.getKPos(), im2colOp.getInputKPerm(),
        im2colOp.getOutputPerm(), lowPad, highPad,
        im2colOp.getMixedOutputPadLow(), im2colOp.getMixedOutputPadHigh(),
        padValue);

    rewriter.replaceOp(im2colOp, newIm2col->getResults());
    return success();
  }
};

/// Fold tensor.pad on the output of im2col into the im2col by expanding the
/// output tensor.
///
/// %out = tensor.empty(...)
/// %result = im2col ins(%input) outs(%out) ...
/// %padded = tensor.pad %result low[...] high[...] { yield %cst }
/// -->
/// %bigger_out = tensor.empty(padded_shape)
/// %result = im2col ins(%input) outs(%bigger_out) ...
struct FoldOutputPadIntoIm2col final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto im2colOp = padOp.getSource().getDefiningOp<Im2colOp>();
    if (!im2colOp) {
      return rewriter.notifyMatchFailure(padOp,
                                         "source not produced by im2col");
    }

    // The im2col must have a single use (this pad).
    if (!im2colOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(padOp, "im2col has multiple uses");
    }

    // The im2col output must come from a tensor.empty.
    auto emptyOp = im2colOp.getOutput().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) {
      return rewriter.notifyMatchFailure(
          padOp, "im2col output not produced by tensor.empty");
    }

    // Only fold constant padding values.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      return rewriter.notifyMatchFailure(padOp, "pad value is not a constant");
    }

    // The padding value must be compatible with the im2col op's existing
    // pad_value. If the im2col has no pad_value yet, adopt the pad's value.
    // If it has one, the values must match.
    if (im2colOp.hasPadding()) {
      auto existingConst =
          im2colOp.getPadValue().getDefiningOp<arith::ConstantOp>();
      auto newConst = padValue.getDefiningOp<arith::ConstantOp>();
      if (!existingConst || !newConst ||
          existingConst.getValue() != newConst.getValue()) {
        return rewriter.notifyMatchFailure(
            padOp, "pad values are not compatible constants");
      }
      padValue = im2colOp.getPadValue();
    }

    Location loc = padOp.getLoc();
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();

    // This fold is safe because the pad_value is verified to be the same
    // constant above, so padded positions in the larger output match what
    // the downstream consumer (e.g., GEMM) expects.
    auto outputType = cast<RankedTensorType>(padOp.getResultType());
    int64_t outputRank = outputType.getRank();

    SmallVector<OpFoldResult> newOutputShape;
    SmallVector<OpFoldResult> oldOutputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getOutput());
    AffineExpr d0, d1, d2;
    bindDims(rewriter.getContext(), d0, d1, d2);
    for (int64_t i = 0; i < outputRank; ++i) {
      newOutputShape.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, d0 + d1 + d2,
          {oldOutputSizes[i], lowPad[i], highPad[i]}));
    }

    auto newEmptyOp = tensor::EmptyOp::create(rewriter, loc, newOutputShape,
                                              outputType.getElementType());

    // Compose output padding from the pad op with any existing output padding.
    SmallVector<OpFoldResult> existingOutPadLow =
        im2colOp.getMixedOutputPadLow();
    SmallVector<OpFoldResult> existingOutPadHigh =
        im2colOp.getMixedOutputPadHigh();
    SmallVector<OpFoldResult> newOutPadLow(lowPad);
    SmallVector<OpFoldResult> newOutPadHigh(highPad);
    if (!existingOutPadLow.empty()) {
      for (int64_t i = 0; i < outputRank; ++i) {
        newOutPadLow[i] =
            addOfrs(rewriter, loc, existingOutPadLow[i], newOutPadLow[i]);
        newOutPadHigh[i] =
            addOfrs(rewriter, loc, existingOutPadHigh[i], newOutPadHigh[i]);
      }
    }

    auto newIm2col = Im2colOp::create(
        rewriter, loc, im2colOp.getInput(), newEmptyOp.getResult(),
        im2colOp.getStrides(), im2colOp.getDilations(),
        im2colOp.getMixedKernelSize(), im2colOp.getMixedOffsets(),
        im2colOp.getMixedOutputSizes(), im2colOp.getBatchPos(),
        im2colOp.getMPos(), im2colOp.getKPos(), im2colOp.getInputKPerm(),
        im2colOp.getOutputPerm(), im2colOp.getMixedInputPadLow(),
        im2colOp.getMixedInputPadHigh(), newOutPadLow, newOutPadHigh, padValue);

    rewriter.replaceOp(padOp, newIm2col->getResults());
    return success();
  }
};

} // namespace

void Im2colOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<FoldInputPadIntoIm2col, FoldOutputPadIntoIm2col>(context);
}

LogicalResult Im2colOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
Im2colOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===---------------------------------------------------------------------===//
// Custom Op
//===---------------------------------------------------------------------===//

unsigned CustomOp::getNumLoops() { return getIteratorTypesAttr().size(); }

int64_t CustomOp::getRank(Value v) {
  Type type = v.getType();
  if (type.isIntOrIndexOrFloat()) {
    return 0;
  }
  return cast<RankedTensorType>(type).getRank();
}

unsigned CustomOp::getNumNonLoopDimensions() {
  for (auto map : getIndexingMaps().getAsValueRange<AffineMapAttr>()) {
    if (map.isEmpty()) {
      continue;
    }
    return map.getNumSymbols();
  }
  return 0;
}

LogicalResult CustomOp::verify() {
  // All inputs/outputs must have indexing maps.
  if (static_cast<int64_t>(getIndexingMapsAttr().size()) != getNumOperands()) {
    return emitOpError("expected number of indexing maps (")
           << getIndexingMapsAttr().size()
           << ") to be same as the "
              "number of input/output operands ("
           << getNumOperands() << ")";
  }

  // Check the form of the indexing maps.
  std::optional<unsigned> numSymbolDims;
  for (auto [index, indexingMapAttr, operand] :
       llvm::enumerate(getIndexingMapsAttr(), getOperands())) {
    auto indexingMap = cast<AffineMapAttr>(indexingMapAttr).getValue();
    if (indexingMap.isEmpty()) {
      continue;
    }

    // Domain must be consistent.
    unsigned numLoops = getNumLoops();
    if (indexingMap.getNumDims() != numLoops) {
      return emitOpError("expected indexing_map #")
             << index << " to have " << numLoops
             << " dim(s) to match the number of loops or be zero";
    }

    // Check that number of symbols is consistent.
    if (numSymbolDims) {
      if (indexingMap.getNumSymbols() != numSymbolDims.value()) {
        return emitOpError(
                   "inconsistent number of symbol dimensions in indexing_map #")
               << index << ", expected " << numSymbolDims.value()
               << " instead of " << indexingMap.getNumSymbols();
      }
    } else {
      numSymbolDims = indexingMap.getNumSymbols();
    }

    // Range must match the rank of the operands.
    int64_t rank = getRank(operand);
    if (indexingMap.getNumResults() != rank) {
      return emitOpError("expected operand rank(")
             << rank << ") to match the result rank of indexing map #" << index;
    }
  }

  // Check that number of basic block arguments is same as number of operands
  Block *body = getBody();
  if (body->getNumArguments() != getNumOperands()) {
    return emitOpError("expected as many basic block arguments (")
           << body->getNumArguments() << ") as the number of operands ("
           << getNumOperands() << ")";
  }

  // Check that type of the basic block argument matches the type of the
  // operands.
  for (auto [index, bbArg, operand] :
       llvm::enumerate(body->getArguments(), getOperands())) {
    Type operandType = operand.getType();
    Type bbArgType = bbArg.getType();

    if (operandType.isIntOrIndexOrFloat()) {
      if (operandType != bbArgType) {
        return emitOpError("for (scalar) operand #")
               << index
               << " expected corresponding basic block argument to be of the "
                  "same type";
      }
      continue;
    }

    auto operandTensorType = cast<RankedTensorType>(operandType);
    auto bbArgTensorType = dyn_cast<RankedTensorType>(bbArgType);
    if (!bbArgTensorType) {
      return emitOpError("for (tensor) operand #")
             << index
             << " expected corresponding basic block argument to be tensor as "
                "well";
    }

    // Check that the basic block arg has same rank/element type, but all shapes
    // dynamic.
    auto expectedBBArgType = RankedTensorType::get(
        SmallVector<int64_t>(operandTensorType.getRank(), ShapedType::kDynamic),
        operandTensorType.getElementType(), operandTensorType.getEncoding());
    if (bbArgTensorType != expectedBBArgType) {
      return emitOpError("expected basic block argument corresponding to "
                         "(tensor) operand #")
             << index << " to be " << expectedBBArgType << " instead of "
             << bbArgTensorType;
    }
  }

  // Check yield operation operand types.
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != getOutputs().size()) {
    return emitOpError(
        "expected as many yields as the numbers of `outs` operand");
  }

  for (auto [index, yieldVal, bbOperand] :
       llvm::enumerate(yieldOp.getOperands(),
                       body->getArguments().take_back(getOutputs().size()))) {
    if (yieldVal.getType() != bbOperand.getType()) {
      return emitOpError("expected type of ")
             << index
             << "-th operand of yield to match the corresponding output basic "
                "block argument";
    }
  }

  return success();
}

/// Start `LinalgFusionInterface` implementation.

SmallVector<AffineMap> CustomOp::getIndexingMapsForOperands() {
  return llvm::map_to_vector(
      getIndexingMaps().getValue().take_front(getNumDpsInputs()),
      [](Attribute attr) { return cast<AffineMapAttr>(attr).getValue(); });
}

SmallVector<AffineMap> CustomOp::getIndexingMapsForResults() {
  return llvm::map_to_vector(
      getIndexingMaps().getValue().take_back(getNumDpsInits()),
      [](Attribute attr) { return cast<AffineMapAttr>(attr).getValue(); });
}

/// End `LinalgFusionInterface` implementation

/// Start `ReifyRankedShapedTypeOpInterface` implementation

LogicalResult
CustomOp::reifyResultShapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  for (auto init : getOutputs()) {
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(builder, getLoc(), init);
    reifiedReturnShapes.emplace_back(std::move(sizes));
  }
  return success();
}

/// End `ReifyRankedShapedTypeOpInterface` implementation

//===---------------------------------------------------------------------===//
// IndexOp
//===---------------------------------------------------------------------===//

LogicalResult IREE::LinalgExt::IndexOp::verify() {
  auto parentOp = getOperation()->getParentOp();
  if (!isa<CustomOp, AttentionOp, OnlineAttentionOp>(parentOp)) {
    return emitOpError(
        "expected parent op to be one of `iree_linalg_ext.custom_op`, "
        "`iree_linalg_ext.attention`, `iree_linalg_ext.online_attention`");
  }
  int64_t numLoops;
  if (auto customOp = dyn_cast<CustomOp>(parentOp)) {
    numLoops = customOp.getNumLoops();
  } else if (auto attentionOp = dyn_cast<AttentionOp>(parentOp)) {
    numLoops = attentionOp.getIterationDomainRank();
  } else {
    numLoops = cast<OnlineAttentionOp>(parentOp).getIterationDomainRank();
  }
  if (numLoops <= getDim()) {
    return emitOpError("expected dim (")
           << getDim() << ") to be lower than the number of loops (" << numLoops
           << ") of the enclosing operation";
  }
  return success();
}

//===---------------------------------------------------------------------===//
// End operation definitions
//===---------------------------------------------------------------------===//

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    getEffectsImpl(effects, getDpsInputOperands(), getDpsInitsMutable());      \
  }

DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(GatherOp)
DEFINE_OP_GET_EFFECTS(MapStoreOp)
DEFINE_OP_GET_EFFECTS(MapLoadOp)
DEFINE_OP_GET_EFFECTS(SortOp)
DEFINE_OP_GET_EFFECTS(FftOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(TopkOp)
DEFINE_OP_GET_EFFECTS(TopkV2Op)
DEFINE_OP_GET_EFFECTS(ArgCompareOp)
DEFINE_OP_GET_EFFECTS(WinogradInputTransformOp)
DEFINE_OP_GET_EFFECTS(WinogradFilterTransformOp)
DEFINE_OP_GET_EFFECTS(WinogradOutputTransformOp)
DEFINE_OP_GET_EFFECTS(AttentionOp)
DEFINE_OP_GET_EFFECTS(OnlineAttentionOp)
DEFINE_OP_GET_EFFECTS(ExpReductionOp)
DEFINE_OP_GET_EFFECTS(Im2colOp)
DEFINE_OP_GET_EFFECTS(CustomOp)

} // namespace mlir::iree_compiler::IREE::LinalgExt

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc" // IWYU pragma: keep

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtPureOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
