// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
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

/// Helper function to verify both `scatter` and `gather`. Since both ops share
/// the same sementics, we can use the same function to verify them. Note: this
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
struct StaticizeLinalgExtOp : public OpRewritePattern<OpTy> {
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
  return {builder.getMultiDimIdentityMap(getUpdateType().getRank()),
          builder.getMultiDimIdentityMap(getIndicesType().getRank()),
          /*output=*/AffineMap(nullptr)};
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
  return SmallVector<AffineMap>{
      AffineMap(nullptr),
      builder.getMultiDimIdentityMap(getIndicesType().getRank()),
      builder.getMultiDimIdentityMap(getOutputType().getRank())};
}

SmallVector<AffineMap> GatherOp::getIndexingMapsForResults() {
  Builder builder(getContext());
  return SmallVector<AffineMap>{
      builder.getMultiDimIdentityMap(getOutputType().getRank())};
}

namespace {
struct ConvertGatherToExtract
    : public OpRewritePattern<IREE::LinalgExt::GatherOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::LinalgExt::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support memref case.
    if (!gatherOp.hasPureTensorSemantics()) {
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
struct ConvertIdentityMapLoadStoreToCopy : public OpRewritePattern<OpTy> {
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
  SmallVector<Type> indexTypes(numOutputIndices, builder.getIndexType());
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
  SmallVector<Type> indexTypes(outputType.getRank(), builder.getIndexType());
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
  SmallVector<Type> indexTypes(inputType.getRank(), builder.getIndexType());
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
  SmallVector<Type> indexTypes(numSourceIndices, builder.getIndexType());
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
/// operands as well as it's comparitor block arguments. So, to remove unused
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
/// arguments are used within the comparitor because that's needed for op
/// functionality.
struct RemoveUnusedSortOpResults
    : public OpRewritePattern<IREE::LinalgExt::SortOp> {
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
  // Input indicies and values must have the same shape.
  if (Value inputIndices = getIndices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices.getType());
    if (failed(verifyCompatibleShape(inputValuesType, inputIndicesType))) {
      return op->emitOpError("input indices/values shape must match");
    }
  }
  // Output indicies and values must have the same shape.
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
// ArgCompareOp
//===----------------------------------------------------------------------===//

LogicalResult ArgCompareOp::verify() {
  Operation *op = getOperation();

  ShapedType inputValueType = getInputType();
  Type inputValueElemType = inputValueType.getElementType();

  ShapedType outputValueType = getOutputValueType();
  ShapedType outputIndexType = getOutputIndexType();
  Type outputIndexElemType = getOutputIndexElementType();

  if (hasExplicitIndexInput()) {
    ShapedType inputIndexType = getInputIndexType();
    Type inputIndexElemType = getInputIndexElementType();

    if (inputValueType.getShape() != inputIndexType.getShape()) {
      return op->emitOpError(
                 "explicit-index mode: value and index inputs must have "
                 "the same shape. ")
             << "Value shape: "
             << llvm::interleaved_array(inputValueType.getShape())
             << ", index shape: "
             << llvm::interleaved_array(inputIndexType.getShape());
    }

    if (!isa<IntegerType, IndexType>(inputIndexElemType)) {
      return op->emitOpError(
                 "explicit-index mode: index input must have integer or index "
                 "element type, but got ")
             << inputIndexElemType;
    }

    if (inputIndexElemType != outputIndexElemType) {
      return op->emitOpError(
                 "explicit-index mode: input and output index element types "
                 "must match. ")
             << "Input index type: " << inputIndexElemType
             << ", output index type: " << outputIndexElemType;
    }

    if (getIndexBase()) {
      return op->emitOpError(
          "index_base must not be used with explicit indices");
    }
  }

  Type outputValueElemType = outputValueType.getElementType();
  if (inputValueElemType != outputValueElemType) {
    return op->emitOpError("input and output value element types must match. ")
           << "Input type: " << inputValueElemType
           << ", output value type: " << outputValueElemType;
  }

  if (!isa<IntegerType, IndexType>(outputIndexElemType)) {
    return op->emitOpError(
               "output index must have integer or index element type, but got ")
           << outputIndexElemType;
  }

  if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
    return op->emitOpError("output indices/values shape must match. ")
           << "Output value shape: "
           << llvm::interleaved_array(outputValueType.getShape())
           << ", output index shape: "
           << llvm::interleaved_array(outputIndexType.getShape());
  }

  uint64_t dim = getDimension();
  int64_t rank = getInputRank();
  if (dim >= rank) {
    return op->emitOpError("reduction dimension exceeds or equals input rank. ")
           << "got dimension: " << dim << ", but input rank is: " << rank;
  }

  SmallVector<int64_t> expectedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      expectedShape.push_back(inputValueType.getDimSize(i));
    }
  }
  if (!llvm::equal(expectedShape, outputValueType.getShape())) {
    return op->emitOpError("output shape must match input shape with reduction "
                           "dimension removed. ")
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: "
           << llvm::interleaved_array(outputValueType.getShape());
  }

  Region &region = getRegion();
  Block &block = region.front();
  unsigned numArgs = block.getNumArguments();
  if (numArgs != 2) {
    return op->emitOpError("region block should have 2 arguments, but got ")
           << numArgs;
  }

  Type arg0Type = block.getArgument(0).getType();
  Type arg1Type = block.getArgument(1).getType();

  if (arg0Type != inputValueElemType || arg1Type != inputValueElemType) {
    return op->emitOpError(
               "comparator arguments must match input value element type. ")
           << "Expected: " << inputValueElemType << ", but got: " << arg0Type
           << " and " << arg1Type;
  }

  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(block.getTerminator());
  unsigned numOperands = yieldOp->getNumOperands();
  if (numOperands != 1) {
    return op->emitOpError(
               "expected linalg_ext.yield to return 1 operand, but got ")
           << numOperands;
  }

  Type yieldType = yieldOp.getOperand(0).getType();
  if (!yieldType.isInteger(1)) {
    return op->emitOpError(
               "region block must end with a linalg_ext.yield i1, but got: ")
           << yieldType;
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

MutableOperandRange ArgCompareOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getInputIndex() ? 2 : 1,
                             /*numInits=*/2);
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp utils
//===----------------------------------------------------------------------===//

/// Return true if at least one element in `tiles` is zero.
static bool hasZeros(ArrayRef<OpFoldResult> tiles) {
  return llvm::any_of(tiles, isZeroInteger);
}

/// Check if we have enough static information to catch undefined behavior when
/// the tile size does not divide perfectly the dimension of the input tensor.
static bool
areNotFullTiles(ArrayRef<int64_t> inputShape,
                DenseMap<int64_t, OpFoldResult> const &dimAndTileMapping) {
  int64_t rank = inputShape.size();
  for (int64_t dim = 0; dim < rank; dim++) {
    if (ShapedType::isDynamic(inputShape[dim])) {
      continue;
    }
    auto it = dimAndTileMapping.find(dim);
    if (it != dimAndTileMapping.end()) {
      std::optional<int64_t> constantTile = getConstantIntValue(it->second);
      if (!constantTile) {
        continue;
      }
      if (inputShape[dim] % (*constantTile) != 0) {
        return true;
      }
    }
  }
  return false;
}

static SmallVector<OpFoldResult> getMixedValues(MLIRContext *context,
                                                ArrayRef<int64_t> staticValues,
                                                OperandRange dynamicValues) {
  OpBuilder b(context);
  return mlir::getMixedValues(staticValues, dynamicValues, b);
}

static SmallVector<int64_t>
getStaticValues(SmallVector<OpFoldResult> mixedValues) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(mixedValues, dynamicTiles, staticTiles);
  return staticTiles;
}

/// Utility function shared between Pack and UnPack to get the tile sizes as
/// OpFoldResults.
// TODO: interface or base class in .td
template <typename OpTy>
static SmallVector<OpFoldResult> getMixedTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  return LinalgExt::getMixedValues(op.getContext(), op.getStaticInnerTiles(),
                                   op.getInnerTiles());
}

/// Return the tile sizes as `int64_t`. If a tile size is dynamic a sentinel
/// `kDynamic` is introduced at that position in the returned vector.
template <typename OpTy>
static SmallVector<int64_t> getStaticTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  return getStaticValues(op.getMixedTiles());
}

/// Utility function shared between Pack and UnPack to get a map between
/// `dim_pos` and `inner_tiles`.
// TODO: interface or base class in .td
template <typename OpTy>
static DenseMap<int64_t, OpFoldResult> getDimAndTileMapping(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
  ArrayRef<int64_t> dimsToBlock = op.getInnerDimsPos();
  SmallVector<OpFoldResult> tiles = op.getMixedTiles();
  assert(tiles.size() == dimsToBlock.size() &&
         "tiles must match indices of dimension to block");
  // bind the dimension with the tile factor.
  for (auto i : llvm::seq<int64_t>(0, dimsToBlock.size())) {
    dimAndTileMapping[dimsToBlock[i]] = tiles[i];
  }
  return dimAndTileMapping;
}

/// Common verifier for `PackOp` and `UnPackOp`.
template <typename OpTy>
static LogicalResult commonVerifierPackAndUnPackOp(OpTy packOrUnPack) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Operation *op = packOrUnPack.getOperation();
  ShapedType unpackedType = (std::is_same<OpTy, PackOp>::value)
                                ? packOrUnPack.getInputType()
                                : packOrUnPack.getOutputType();
  int64_t unpackedRank = unpackedType.getRank();
  ArrayRef<int64_t> innerDimsPos = packOrUnPack.getInnerDimsPos();
  ArrayRef<int64_t> outerDimPerm = packOrUnPack.getOuterDimsPerm();
  // Verify tiles. Make sure each provided tile is non-zero.
  SmallVector<OpFoldResult> mixedTiles = packOrUnPack.getMixedTiles();
  if (hasZeros(mixedTiles)) {
    return op->emitError("invalid tile factor");
  }
  if (isInvalid(innerDimsPos, unpackedRank)) {
    return op->emitError("invalid inner_dims_pos vector");
  }
  if (isInvalid(outerDimPerm, unpackedRank)) {
    return op->emitError("invalid outer_dims_perm vector");
  }
  if (mixedTiles.size() != innerDimsPos.size()) {
    return op->emitError(
        "blocking factors must equal the number of dimensions to block");
  }

  // Blocking factors must be less or equal than the input rank, and must
  // match the number of `dims_pos`.
  if (mixedTiles.size() > unpackedRank) {
    return op->emitError(
        "blocking factors must be less or equal than the input rank");
  }

  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? packOrUnPack.getOutputType()
                              : packOrUnPack.getInputType();
  int64_t packedRank = packedType.getRank();
  // Require output rank to match input rank + number of blocking factors.
  if (unpackedRank + mixedTiles.size() != packedRank) {
    return op->emitError(
        "packed rank must equal unpacked rank + blocking factors");
  }

  // Verify result shape is greater than the minimum expected
  // by the pack operation, and that the output shape
  // represents full tiles.
  ShapedType expectedPackedType = PackOp::getPackedType(
      unpackedType, packOrUnPack.getStaticTiles(), innerDimsPos, outerDimPerm);
  if (!isSmallerThan(expectedPackedType.getShape(), packedType.getShape())) {
    return op->emitError("the shape of output is not large enough to hold the "
                         "packed data. Expected at least ")
           << expectedPackedType << ", got " << packedType;
  }
  if (!llvm::all_of(
          llvm::zip_equal(packedType.getShape().take_back(mixedTiles.size()),
                          mixedTiles),
          [](std::tuple<int64_t, OpFoldResult> it) {
            std::optional<int64_t> constTileSize =
                getConstantIntValue(std::get<1>(it));
            int64_t shape = std::get<0>(it);
            if (!constTileSize) {
              // If specified tile size is dynamic, output shape should
              // be dynamic too.
              return ShapedType::isDynamic(shape);
            } else {
              if (ShapedType::isDynamic(shape)) {
                // For the shape being dynamic when tile size is
                // specified, return true. In canonical form a constant
                // tile size should lead to constant shape of the tiled
                // dimension, but not needed for verification.
                return true;
              }
              return shape == constTileSize.value();
            }
          })) {
    return op->emitError("mismatch in inner tile sizes specified and shaped of "
                         "tiled dimension in the packed type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

/// Custom builder methods for pack ops.
void PackOp::build(OpBuilder &builder, OperationState &state, Value source,
                   Value output, ArrayRef<int64_t> innerDimsPos,
                   ArrayRef<OpFoldResult> innerTiles,
                   std::optional<Value> paddingValue,
                   ArrayRef<int64_t> outerDimsPerm) {
  assert(innerDimsPos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, source, output,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes),
        (paddingValue ? paddingValue.value() : nullptr));
}

LogicalResult PackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }

  // Bail out if the tile does not divide the dimension fully. In the case of
  // dynamic tile factors or dimensions, having a partial tile is undefined
  // behavior.
  auto dimAndTileMapping = getDimAndTileMapping();
  if (!getPaddingValue() &&
      areNotFullTiles(getInputShape(), dimAndTileMapping)) {
    return emitOpError("invalid tile factor provided. Only full tiles are "
                       "supported when padding_value is not set");
  }

  if (auto paddingValue = getPaddingValue()) {
    if (paddingValue.getType() != getInputType().getElementType()) {
      return emitOpError("expected padding_value has ")
             << getInputType().getElementType()
             << " but got: " << paddingValue.getType();
    }
  }
  return success();
}

SmallVector<OpFoldResult> PackOp::getMixedTiles() {
  return LinalgExt::getMixedTiles(*this);
}

SmallVector<int64_t> PackOp::getStaticTiles() {
  return LinalgExt::getStaticTiles(*this);
}

// Helper for PackOp::{getResultShape,getPackedType}. Returns the shape of the
// packed type. Having a shared helper helps implement these two methods in a
// way that ensures that they agree on which dimensions are dynamic.
static SmallVector<int64_t> getPackOpResultTypeShape(
    ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> innerTileSizes,
    ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultShape = llvm::to_vector(sourceShape);
  for (auto [idx, tiledDim] : llvm::enumerate(innerDimsPos)) {
    if (ShapedType::isDynamic(resultShape[tiledDim])) {
      continue;
    }
    if (ShapedType::isDynamic(innerTileSizes[idx])) {
      resultShape[tiledDim] = ShapedType::kDynamic;
      continue;
    }
    resultShape[tiledDim] =
        llvm::divideCeil(resultShape[tiledDim], innerTileSizes[idx]);
  }

  // Swap tile loops if outer_dims_perm is available.
  resultShape = interchange<int64_t>(resultShape, outerDimsPerm, /*offset=*/0);

  // Append the inner tile dimensions.
  resultShape.append(innerTileSizes.begin(), innerTileSizes.end());
  return resultShape;
}

SmallVector<OpFoldResult> PackOp::getResultShape(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> sourceDims,
    ArrayRef<OpFoldResult> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<OpFoldResult> resultDims = llvm::to_vector(sourceDims);

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);
  for (auto [idx, tiledDim] : llvm::enumerate(innerDimsPos)) {
    resultDims[tiledDim] = affine::makeComposedFoldedAffineApply(
        builder, loc, ceilDivExpr, {resultDims[tiledDim], innerTileSizes[idx]});
  }
  if (!outerDimsPerm.empty()) {
    resultDims =
        interchange<OpFoldResult>(resultDims, outerDimsPerm, /*offset=*/0);
  }
  resultDims.append(innerTileSizes.begin(), innerTileSizes.end());

  SmallVector<int64_t> resultTypeShape =
      getPackOpResultTypeShape(asShapeWithAnyValueAsDynamic(sourceDims),
                               asShapeWithAnyValueAsDynamic(innerTileSizes),
                               innerDimsPos, outerDimsPerm);

  // Fix-up `resultDims` to ensure that they are Value's if and only if the
  // result type shape says it's a dynamic dim. This is needed as callers may
  // use dispatchIndexOpFoldResults on the result, and rely on exact number of
  // dynamic dims returned by that.
  for (unsigned i = 0; i < resultDims.size(); ++i) {
    if (ShapedType::isStatic(resultTypeShape[i])) {
      continue;
    }
    resultDims[i] =
        getValueOrCreateConstantIndexOp(builder, loc, resultDims[i]);
  }

  return resultDims;
}

SmallVector<OpFoldResult> PackOp::getResultShape(OpBuilder &builder) {
  return tensor::getMixedSizes(builder, getLoc(), getOutput());
}

ShapedType PackOp::getPackedType(ShapedType sourceType,
                                 ArrayRef<int64_t> innerTileSizes,
                                 ArrayRef<int64_t> innerDimsPos,
                                 ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultTypeShape = getPackOpResultTypeShape(
      sourceType.getShape(), innerTileSizes, innerDimsPos, outerDimsPerm);

  return TypeSwitch<ShapedType, ShapedType>(sourceType)
      .Case([&](RankedTensorType shapedType) {
        return RankedTensorType::get(resultTypeShape,
                                     shapedType.getElementType());
      })
      .Case([&](MemRefType shapedType) {
        return MemRefType::get(resultTypeShape, shapedType.getElementType());
      })
      .Default([&](Type t) {
        assert(false && "unexpected type");
        return nullptr;
      });
}

DenseMap<int64_t, OpFoldResult> PackOp::getDimAndTileMapping() {
  return LinalgExt::getDimAndTileMapping(*this);
}

LogicalResult
PackOp::reifyResultShapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

MutableOperandRange PackOp::getDpsInitsMutable() { return getOutputMutable(); }

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

/// Custom builder methods for unpack ops.
void UnPackOp::build(OpBuilder &builder, OperationState &state, Value source,
                     Value output, ArrayRef<int64_t> innerDimsPos,
                     ArrayRef<OpFoldResult> innerTiles,
                     ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, source, output,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

SmallVector<OpFoldResult> UnPackOp::getMixedTiles() {
  return LinalgExt::getMixedTiles(*this);
}

SmallVector<int64_t> UnPackOp::getStaticTiles() {
  return LinalgExt::getStaticTiles(*this);
}

DenseMap<int64_t, OpFoldResult> UnPackOp::getDimAndTileMapping() {
  return LinalgExt::getDimAndTileMapping(*this);
}

LogicalResult
UnPackOp::reifyResultShapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

LogicalResult UnPackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }
  return success();
}

MutableOperandRange UnPackOp::getDpsInitsMutable() {
  return getOutputMutable();
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
  build(odsBuilder, odsState, results, query, key, value, maskIn, scale, output,
        max, sum, indexingMaps, DictionaryAttr());
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

/// Return all static and dynamic kernel_size as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedKernelSize() {
  return LinalgExt::getMixedValues(getContext(), getStaticKernelSize(),
                                   getKernelSize());
}

/// Return all static and dynamic k_offset as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedKOffset() {
  return LinalgExt::getMixedValues(getContext(), getStaticKOffset(),
                                   getKOffset());
}

/// Return all static and dynamic m_offset as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedMOffset() {
  return LinalgExt::getMixedValues(getContext(), getStaticMOffset(),
                                   getMOffset());
}

/// Return all static and dynamic k_strides as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedKStrides() {
  return LinalgExt::getMixedValues(getContext(), getStaticKStrides(),
                                   getKStrides());
}

/// Return all static and dynamic m_strides as OpFoldResults.
SmallVector<OpFoldResult> Im2colOp::getMixedMStrides() {
  return LinalgExt::getMixedValues(getContext(), getStaticMStrides(),
                                   getMStrides());
}

void Im2colOp::setMixedKOffset(SmallVector<OpFoldResult> kOffset) {
  SmallVector<int64_t> staticKOffset;
  SmallVector<Value> dynamicKOffset;
  dispatchIndexOpFoldResults(kOffset, dynamicKOffset, staticKOffset);
  setStaticKOffset(staticKOffset);
  getKOffsetMutable().assign(dynamicKOffset);
}

void Im2colOp::setMixedMOffset(SmallVector<OpFoldResult> mOffset) {
  SmallVector<int64_t> staticMOffset;
  SmallVector<Value> dynamicMOffset;
  dispatchIndexOpFoldResults(mOffset, dynamicMOffset, staticMOffset);
  setStaticMOffset(staticMOffset);
  getMOffsetMutable().assign(dynamicMOffset);
}

void Im2colOp::setMixedKStrides(SmallVector<OpFoldResult> kStrides) {
  SmallVector<int64_t> staticKStrides;
  SmallVector<Value> dynamicKStrides;
  dispatchIndexOpFoldResults(kStrides, dynamicKStrides, staticKStrides);
  setStaticKStrides(staticKStrides);
  getKStridesMutable().assign(dynamicKStrides);
}

void Im2colOp::setMixedMStrides(SmallVector<OpFoldResult> mStrides) {
  SmallVector<int64_t> staticMStrides;
  SmallVector<Value> dynamicMStrides;
  dispatchIndexOpFoldResults(mStrides, dynamicMStrides, staticMStrides);
  setStaticMStrides(staticMStrides);
  getMStridesMutable().assign(dynamicMStrides);
}

SmallVector<int64_t> Im2colOp::getBatchOutputDims() {
  SmallVector<int64_t> inverseOutPerm =
      invertPermutationVector(getOutputPerm());
  return llvm::map_to_vector(llvm::seq<int64_t>(0, getBatchPos().size()),
                             [&](int64_t dim) { return inverseOutPerm[dim]; });
}

SmallVector<int64_t> Im2colOp::getMOutputDims() {
  int64_t begin = getBatchPos().size();
  int64_t end = begin + getMixedMOffset().size();
  SmallVector<int64_t> inverseOutPerm =
      invertPermutationVector(getOutputPerm());
  return llvm::map_to_vector(llvm::seq<int64_t>(begin, end),
                             [&](int64_t dim) { return inverseOutPerm[dim]; });
}

SmallVector<int64_t> Im2colOp::getKOutputDims() {
  int64_t begin = getBatchPos().size() + getMixedMOffset().size();
  int64_t end = begin + getMixedKOffset().size();
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
    ArrayRef<OpFoldResult> kernelSize, ArrayRef<OpFoldResult> mOffset,
    ArrayRef<OpFoldResult> mStrides, ArrayRef<OpFoldResult> kOffset,
    ArrayRef<OpFoldResult> kStrides, ArrayRef<int64_t> batchPos,
    ArrayRef<int64_t> mPos, ArrayRef<int64_t> kPos,
    ArrayRef<int64_t> inputKPerm, ArrayRef<int64_t> outputPerm) {
  assert(strides.size() == kernelSize.size() &&
         dilations.size() == kernelSize.size() &&
         mPos.size() == kernelSize.size() &&
         "strides, dilations, m_pos, and kernel expected to be the same rank");
  SmallVector<int64_t> staticKernelSize, staticMOffset, staticKOffset,
      staticMStrides, staticKStrides;
  SmallVector<Value> dynamicKernelSize, dynamicMOffset, dynamicKOffset,
      dynamicMStrides, dynamicKStrides;
  dispatchIndexOpFoldResults(kernelSize, dynamicKernelSize, staticKernelSize);
  dispatchIndexOpFoldResults(mOffset, dynamicMOffset, staticMOffset);
  dispatchIndexOpFoldResults(mStrides, dynamicMStrides, staticMStrides);
  dispatchIndexOpFoldResults(kOffset, dynamicKOffset, staticKOffset);
  dispatchIndexOpFoldResults(kStrides, dynamicKStrides, staticKStrides);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (isa<RankedTensorType>(outputType)) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, input, output,
        builder.getDenseI64ArrayAttr(strides),
        builder.getDenseI64ArrayAttr(dilations), dynamicKernelSize,
        builder.getDenseI64ArrayAttr(staticKernelSize), dynamicMOffset,
        builder.getDenseI64ArrayAttr(staticMOffset), dynamicMStrides,
        builder.getDenseI64ArrayAttr(staticMStrides), dynamicKOffset,
        builder.getDenseI64ArrayAttr(staticKOffset), dynamicKStrides,
        builder.getDenseI64ArrayAttr(staticKStrides),
        builder.getDenseI64ArrayAttr(batchPos),
        builder.getDenseI64ArrayAttr(mPos), builder.getDenseI64ArrayAttr(kPos),
        builder.getDenseI64ArrayAttr(inputKPerm),
        builder.getDenseI64ArrayAttr(outputPerm));
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

  // Verify offsets and strides
  SmallVector<OpFoldResult> kOffset = getMixedKOffset();
  SmallVector<OpFoldResult> mOffset = getMixedMOffset();
  SmallVector<OpFoldResult> kStrides = getMixedKStrides();
  SmallVector<OpFoldResult> mStrides = getMixedMStrides();
  if (kOffset.size() < 1) {
    return op->emitOpError("expected at least one k_offset");
  }
  if (mOffset.size() < 1) {
    return op->emitOpError("expected at least one m_offset");
  }
  if (kOffset.size() != kStrides.size()) {
    return op->emitOpError("expected the same size k_offset and k_strides");
  }
  if (mOffset.size() != mStrides.size()) {
    return op->emitOpError("expected the same size m_offset and m_strides");
  }
  std::optional<int64_t> constInnerKStrides =
      getConstantIntValue(kStrides.back());
  if (!constInnerKStrides.has_value() || constInnerKStrides.value() != 1) {
    return op->emitOpError("expected inner k_strides to be 1");
  }
  std::optional<int64_t> constInnerMStrides =
      getConstantIntValue(mStrides.back());
  if (!constInnerMStrides.has_value() || constInnerMStrides.value() != 1) {
    return op->emitOpError("expected inner m_strides to be 1");
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
  if (outputRank != batchPos.size() + kOffset.size() + mOffset.size()) {
    return op->emitOpError("expected output rank to be the sum of "
                           "batch_pos, k_offset, and m_offset ranks");
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

  size_t sharedRank = mPos.size() + kPos.size();
  if (inputKPerm.size() != sharedRank) {
    return op->emitOpError("expected input_k_perm size (")
           << inputKPerm.size()
           << ") to match the number of shared dimensions (m_Pos + k_pos = "
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
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  ArrayRef<int64_t> outputPerm = getOutputPerm();
  if (!isPermutationVector(outputPerm)) {
    return op->emitOpError("expected output_perm to be a permutation");
  }
  if (outputPerm.size() != outputShape.size()) {
    return op->emitOpError(
        "expected output_perm to have the same rank as the result");
  }

  // When the op is tiled, the m and k dimensions of the output are tiled, but
  // they are not tiled in the input, so we cannot verify the output size of
  // these dimensions. Only verify the shape of the batch dimensions.
  SmallVector<int64_t> expectedOutputShape(outputShape);
  SmallVector<int64_t> inverseOutputPerm = invertPermutationVector(outputPerm);
  for (auto [idx, pos] : llvm::enumerate(batchPos)) {
    expectedOutputShape[inverseOutputPerm[idx]] = inputShape[pos];
  }
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
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
  if (!isa<CustomOp, AttentionOp>(parentOp)) {
    return emitOpError(
        "expected parent op to be one of `iree_linalg_ext.custom_op`, "
        "`iree_linalg_ext.attention`");
  }
  auto customOp = dyn_cast<CustomOp>(parentOp);
  auto attentionOp = dyn_cast<AttentionOp>(parentOp);
  int64_t numLoops =
      customOp ? customOp.getNumLoops() : attentionOp.getNumLoops();
  if (numLoops <= getDim()) {
    return emitOpError("expected dim (")
           << getDim() << ") to be lower than the number of loops (" << numLoops
           << ") of the enclosing CustomOp/AttentionOp";
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
DEFINE_OP_GET_EFFECTS(ArgCompareOp)
DEFINE_OP_GET_EFFECTS(PackOp)
DEFINE_OP_GET_EFFECTS(UnPackOp)
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
