// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {

namespace {

// -----------------------------------------------------------------------------
// Broadcasting utilities
// -----------------------------------------------------------------------------

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<StringRef, 3> getParallelAndReductionIterators(int nLoops,
                                                           int nReduction) {
  SmallVector<StringRef, 3> res(nLoops - nReduction,
                                getParallelIteratorTypeName());
  res.append(nReduction, getReductionIteratorTypeName());
  return res;
}

SmallVector<StringRef, 3> getNParallelLoopsAttrs(int nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

// Holds a static extent or Value for dynamic extents.
class Extent {
 public:
  Extent() {}
  Extent(int64_t extent) : extent(extent) {}
  Extent(Value value) : value(value) {}

  bool isStatic() { return !value; }
  bool isUnitExtent() { return isStatic() && getStatic() == 1; }
  int64_t getStatic() {
    assert(isStatic());
    return extent;
  }
  Value getValue() {
    assert(!isStatic());
    return value;
  }

  Value convertToValue(OpBuilder &builder, Location loc) {
    if (!isStatic()) return getValue();
    return builder.create<ConstantIndexOp>(loc, getStatic());
  }

 private:
  int64_t extent;
  Value value;
};

Value broadcast(OpBuilder &builder, Location loc, Value operand,
                SmallVectorImpl<Extent> &resultExtents,
                SmallVectorImpl<bool> &isExpansion) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  SmallVector<int64_t> resultShape;
  SmallVector<Value> dynDims;
  for (Extent &dim : resultExtents) {
    if (dim.isStatic()) {
      resultShape.push_back(dim.getStatic());
    } else {
      resultShape.push_back(-1);
      dynDims.push_back(dim.getValue());
    }
  }

  // Traverse the right aligned operand dimensions and form expressions.
  // We keep 1-dims in place instead of reshaping them away, relying on the
  // DropUnitDims pass to run later.
  SmallVector<AffineExpr> dimExprs;
  dimExprs.reserve(operandType.getRank());
  for (int i = resultExtents.size() - operandType.getRank();
       i < resultExtents.size(); ++i) {
    if (isExpansion[i]) {
      dimExprs.push_back(builder.getAffineConstantExpr(0));
    } else {
      dimExprs.push_back(builder.getAffineDimExpr(i));
    }
  }

  int nloops = resultExtents.size();
  Value init = builder.create<linalg::InitTensorOp>(
      loc, dynDims, resultShape, operandType.getElementType());
  auto generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{init.getType()}, ValueRange{operand},
      /*outputBuffers=*/ValueRange{init},
      llvm::makeArrayRef({
          AffineMap::get(/*dimCount=*/nloops, /*symbolCount=*/0, dimExprs,
                         builder.getContext()),
          builder.getMultiDimIdentityMap(nloops),
      }),
      getNParallelLoopsAttrs(nloops),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
      });
  return generic.getResult(0);
}

Value broadcastScalar(OpBuilder &builder, Location loc, Value scalarValue,
                      SmallVectorImpl<Extent> &resultExtents) {
  SmallVector<bool> isExpansion(resultExtents.size());
  for (int i = 0, e = resultExtents.size(); i < e; ++i) {
    isExpansion[i] = true;
  }
  return broadcast(builder, loc, scalarValue, resultExtents, isExpansion);
}

Optional<Extent> computeResultExtent(OpBuilder &builder, Location loc,
                                     Extent &lhsDim, Extent &rhsDim,
                                     bool &isLhsExpansion,
                                     bool &isRhsExpansion) {
  if (lhsDim.isStatic() && rhsDim.isStatic()) {
    // Both are static. Just check.
    if (lhsDim.getStatic() != rhsDim.getStatic() &&
        !(lhsDim.getStatic() == 1 || rhsDim.getStatic() == 1)) {
      // Statically illegal.
      emitError(loc) << "cannot broadcast extents of differing size unless "
                        "if one of them is 1 (got "
                     << lhsDim.getStatic() << ", " << rhsDim.getStatic() << ")";
      return llvm::None;
    }

    // Static expansions.
    if (lhsDim.isUnitExtent() && rhsDim.isUnitExtent()) {
      // For the fully static case, we can trivially check the 1-equality,
      // and know we are not expanding.
      isLhsExpansion = false;
      isRhsExpansion = false;
    } else {
      // Otherwise, mark the dim as expanding if it is 1.
      isLhsExpansion = lhsDim.isUnitExtent();
      isRhsExpansion = rhsDim.isUnitExtent();
    }
    return Extent(std::max(lhsDim.getStatic(), rhsDim.getStatic()));
  }

  // At least one of them is dynamic.
  // Branch on whether one of them is a static-1, which is the only case
  // we allow for dynamic expansion.
  if (lhsDim.isUnitExtent() || rhsDim.isUnitExtent()) {
    if (lhsDim.isUnitExtent()) {
      isLhsExpansion = true;
      isRhsExpansion = false;
      return rhsDim;
    } else {
      isLhsExpansion = false;
      isRhsExpansion = true;
      return lhsDim;
    }
  }

  // At least one is dynamic and neither are a static 1.
  // In this case, we do not allow either to be an expanding dim and
  // error if this is the case at runtime.
  isLhsExpansion = false;
  isRhsExpansion = false;
  Value lhsExtentValue = lhsDim.convertToValue(builder, loc);
  Value rhsExtentValue = rhsDim.convertToValue(builder, loc);

  Value isEqual = builder.create<CmpIOp>(loc, CmpIPredicate::eq, lhsExtentValue,
                                         rhsExtentValue);
  builder.create<AssertOp>(
      loc, isEqual,
      builder.getStringAttr("mismatched dynamic broadcast extents"));

  // Here, if one of them is static, that has to be the result extent
  // (because we checked the error condition above).
  if (lhsDim.isStatic()) {
    return Extent(lhsDim.getStatic());
  } else if (rhsDim.isStatic()) {
    return Extent(rhsDim.getStatic());
  }

  // Both are dynamic. Compute the max.
  Value lhsIsGreater = builder.create<CmpIOp>(loc, CmpIPredicate::sge,
                                              lhsExtentValue, rhsExtentValue);
  Value resultExtent = builder.create<SelectOp>(loc, lhsIsGreater,
                                                lhsExtentValue, rhsExtentValue);
  return Extent(resultExtent);
}

void padExtents(SmallVectorImpl<Extent> &extents, int size) {
  for (int i = 0; i < size; ++i) {
    extents.push_back({1});
  }
}

void appendExtents(OpBuilder &builder, Location loc,
                   SmallVectorImpl<Extent> &extents, Value v,
                   RankedTensorType t) {
  for (int i = 0; i < t.getRank(); ++i) {
    if (t.isDynamicDim(i)) {
      // Emit a dim op.
      Value dim = builder.create<memref::DimOp>(loc, v, i);
      extents.push_back(dim);
    } else {
      // Static dim.
      extents.push_back({t.getDimSize(i)});
    }
  }
}

// -----------------------------------------------------------------------------
// Structural op conversions
// -----------------------------------------------------------------------------

struct ConvertConstantLikeOp
    : public OpConversionPattern<chlo::ConstantLikeOp> {
  using OpConversionPattern<chlo::ConstantLikeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      chlo::ConstantLikeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType().cast<RankedTensorType>();
    if (!resultTy.hasRank())
      return rewriter.notifyMatchFailure(op, "only supports ranked");
    // Lower to MHLO constant if statically shaped.
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ConstOp>(
          op, DenseElementsAttr::get(resultTy, op.value()));
      return success();
    }

    chlo::ConstantLikeOpAdaptor transformed(operands);
    Location loc = op.getLoc();

    int resultRank = resultTy.getRank();
    SmallVector<Extent> resultExtents;
    resultExtents.reserve(resultRank);
    appendExtents(rewriter, loc, resultExtents, transformed.operand(),
                  resultTy);

    auto resultTy0D = RankedTensorType::get({}, resultTy.getElementType());
    Value scalarConst = rewriter.create<mhlo::ConstOp>(
        loc, DenseElementsAttr::get(resultTy0D, op.value()));
    Value broadcasted =
        broadcastScalar(rewriter, loc, scalarConst, resultExtents);
    rewriter.replaceOp(op, {broadcasted});
    return success();
  }
};

// -----------------------------------------------------------------------------
// Binary broadcasting op conversions
// -----------------------------------------------------------------------------

// Adapter base class for adapting binary elementwise broadcasting ops
// via generic patterns. Implemented as a virtual class in order to reduce
// high fanout template instantiations.
struct BinaryBroadcastingAdaptor {
  using BroadcastValues = std::pair<Value, Value>;
  virtual ~BinaryBroadcastingAdaptor() = default;
  virtual StringRef getFromOperationName() = 0;
  virtual LogicalResult verifyBroadcastCompatibility(
      Operation *op, ArrayRef<Value> operands) = 0;
  virtual BroadcastValues getFromBroadcastValues(Operation *op,
                                                 ArrayRef<Value> operands) = 0;
  virtual Operation *createTargetOperation(Location loc, Operation *op,
                                           Type resultType,
                                           ArrayRef<Value> operands,
                                           BroadcastValues broadcastValues,
                                           OpBuilder &builder) = 0;
};

// Adaptor for simple binary elementwise operations which have exactly two
// operands and are matched from src -> target by name.
template <typename FromOpTy, typename ToOpTy>
struct SimpleBinaryBroadcastingAdaptor : public BinaryBroadcastingAdaptor {
  static BinaryBroadcastingAdaptor &getInstance() {
    static SimpleBinaryBroadcastingAdaptor<FromOpTy, ToOpTy> instance;
    return instance;
  }
  StringRef getFromOperationName() override {
    return FromOpTy::getOperationName();
  }
  LogicalResult verifyBroadcastCompatibility(
      Operation *op, ArrayRef<Value> operands) override {
    auto broadcastDimensions = llvm::cast<FromOpTy>(op).broadcast_dimensions();
    if (broadcastDimensions &&
        !hlo::IsLegalNumpyRankedBroadcast(operands[0], operands[1],
                                          *broadcastDimensions)) {
      return failure();
    }
    return success();
  }
  BroadcastValues getFromBroadcastValues(Operation *op,
                                         ArrayRef<Value> operands) override {
    assert(operands.size() == 2);
    return std::make_pair(operands[0], operands[1]);
  }
  Operation *createTargetOperation(Location loc, Operation *op, Type resultType,
                                   ArrayRef<Value> operands,
                                   BroadcastValues broadcastValues,
                                   OpBuilder &builder) override {
    return builder.create<ToOpTy>(loc, resultType, broadcastValues.first,
                                  broadcastValues.second);
  }
};

struct CompareBinaryBroadcastingAdaptor : public BinaryBroadcastingAdaptor {
  static BinaryBroadcastingAdaptor &getInstance() {
    static CompareBinaryBroadcastingAdaptor instance;
    return instance;
  }
  StringRef getFromOperationName() override {
    return chlo::BroadcastCompareOp::getOperationName();
  }
  LogicalResult verifyBroadcastCompatibility(
      Operation *op, ArrayRef<Value> operands) override {
    auto broadcastDimensions =
        llvm::cast<chlo::BroadcastCompareOp>(op).broadcast_dimensions();
    if (broadcastDimensions &&
        !hlo::IsLegalNumpyRankedBroadcast(operands[0], operands[1],
                                          *broadcastDimensions)) {
      return failure();
    }
    return success();
  }
  BroadcastValues getFromBroadcastValues(Operation *op,
                                         ArrayRef<Value> operands) override {
    chlo::BroadcastCompareOpAdaptor adaptor(operands, op->getAttrDictionary());
    return std::make_pair(adaptor.lhs(), adaptor.rhs());
  }
  Operation *createTargetOperation(Location loc, Operation *op, Type resultType,
                                   ArrayRef<Value> operands,
                                   BroadcastValues broadcastValues,
                                   OpBuilder &builder) override {
    chlo::BroadcastCompareOpAdaptor adaptor(operands, op->getAttrDictionary());
    return builder.create<mhlo::CompareOp>(
        loc, resultType, broadcastValues.first, broadcastValues.second,
        adaptor.comparison_direction(), adaptor.compare_type());
  }
};

struct ConvertRankedBroadcastBinaryOp : public ConversionPattern {
  ConvertRankedBroadcastBinaryOp(MLIRContext *context,
                                 TypeConverter &typeConverter,
                                 PatternBenefit benefit,
                                 BinaryBroadcastingAdaptor &bcastAdaptor)
      : ConversionPattern(typeConverter, bcastAdaptor.getFromOperationName(),
                          benefit, context),
        bcastAdaptor(bcastAdaptor) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Only rewrite for statically determinable non-broadcasting cases.
    auto bcastOperands = bcastAdaptor.getFromBroadcastValues(op, operands);
    Value lhs = bcastOperands.first;
    Value rhs = bcastOperands.second;
    auto lhsType = lhs.getType().template dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "not ranked tensors");

    if (failed(bcastAdaptor.verifyBroadcastCompatibility(op, operands))) {
      return rewriter.notifyMatchFailure(op, "not legal broadcasting");
    }

    // Extract the original extents.
    SmallVector<Extent> lhsOrigExtents;
    lhsOrigExtents.reserve(lhsType.getRank());
    appendExtents(rewriter, loc, lhsOrigExtents, lhs, lhsType);
    SmallVector<Extent> rhsOrigExtents;
    rhsOrigExtents.reserve(rhsType.getRank());
    appendExtents(rewriter, loc, rhsOrigExtents, rhs, rhsType);

    // Left pad with 1-extents to the result rank.
    int resultRank = std::max(lhsType.getRank(), rhsType.getRank());
    SmallVector<Extent> lhsBcastExtents;
    lhsBcastExtents.reserve(resultRank);
    SmallVector<Extent> rhsBcastExtents;
    rhsBcastExtents.reserve(resultRank);
    padExtents(lhsBcastExtents, resultRank - lhsType.getRank());
    lhsBcastExtents.append(lhsOrigExtents);
    padExtents(rhsBcastExtents, resultRank - rhsType.getRank());
    rhsBcastExtents.append(rhsOrigExtents);

    // Compute the result extents.
    SmallVector<Extent> resultExtents(resultRank);
    SmallVector<bool> isLhsExpansion(resultRank);
    SmallVector<bool> isRhsExpansion(resultRank);
    bool lhsNeedsBroadcast = resultRank != lhsType.getRank();
    bool rhsNeedsBroadcast = resultRank != rhsType.getRank();
    for (int i = 0; i < resultRank; i++) {
      auto resultExtent = computeResultExtent(
          rewriter, loc, lhsBcastExtents[i], rhsBcastExtents[i],
          isLhsExpansion[i], isRhsExpansion[i]);
      if (!resultExtent)
        return rewriter.notifyMatchFailure(op,
                                           "could not compute result extent");
      resultExtents[i] = *resultExtent;
      if (isLhsExpansion[i]) lhsNeedsBroadcast = true;
      if (isRhsExpansion[i]) rhsNeedsBroadcast = true;
    }

    // Broadcast the operands.
    Value lhsBcast =
        lhsNeedsBroadcast
            ? broadcast(rewriter, loc, lhs, resultExtents, isLhsExpansion)
            : lhs;
    Value rhsBcast =
        rhsNeedsBroadcast
            ? broadcast(rewriter, loc, rhs, resultExtents, isRhsExpansion)
            : rhs;

    // TODO: Don't do this result type change.
    rewriter.replaceOp(op,
                       bcastAdaptor
                           .createTargetOperation(
                               loc, op, op->getResult(0).getType(), operands,
                               std::make_pair(lhsBcast, rhsBcast), rewriter)
                           ->getResults());
    return success();
  }

  BinaryBroadcastingAdaptor &bcastAdaptor;
};

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding mhlo non-broadcasting op.
struct ConvertTrivialNonBroadcastBinaryOp : public ConversionPattern {
  ConvertTrivialNonBroadcastBinaryOp(MLIRContext *context,
                                     TypeConverter &typeConverter,
                                     PatternBenefit benefit,
                                     BinaryBroadcastingAdaptor &bcastAdaptor)
      : ConversionPattern(typeConverter, bcastAdaptor.getFromOperationName(),
                          benefit, context),
        bcastAdaptor(bcastAdaptor) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto bcastOperands = bcastAdaptor.getFromBroadcastValues(op, operands);
    auto lhs_type =
        bcastOperands.first.getType().template dyn_cast<RankedTensorType>();
    auto rhs_type =
        bcastOperands.second.getType().template dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type)
      return rewriter.notifyMatchFailure(op, "not ranked tensors");

    // Requires rank broadcast.
    if (lhs_type.getRank() != rhs_type.getRank())
      return rewriter.notifyMatchFailure(op, "not same rank");
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhs_type.hasStaticShape() || !rhs_type.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "not static shapes");

    for (auto extents : llvm::zip(lhs_type.getShape(), rhs_type.getShape())) {
      auto lhs_extent = std::get<0>(extents);
      auto rhs_extent = std::get<1>(extents);
      if (lhs_extent != rhs_extent) {
        return rewriter.notifyMatchFailure(op, "not equal extents");
      }
    }

    if (failed(bcastAdaptor.verifyBroadcastCompatibility(op, operands))) {
      return rewriter.notifyMatchFailure(op, "not legal broadcasting");
    }

    rewriter.replaceOp(op, bcastAdaptor
                               .createTargetOperation(
                                   op->getLoc(), op, op->getResult(0).getType(),
                                   operands, bcastOperands, rewriter)
                               ->getResults());
    return success();
  }

  BinaryBroadcastingAdaptor &bcastAdaptor;
};

// -----------------------------------------------------------------------------
// Ternary broadcasting op conversions
// -----------------------------------------------------------------------------

// Sepecial case conversion for the BroadcastSelectOp into primitives.
// Note that the "specification" for this op is totally self-contradictory and
// no one seems to know what its broadcasting semantics actually are.
// The most canonical documentation
// (https://www.tensorflow.org/xla/operation_semantics#select) has a completely
// different set of constraints expressed than the (minimal) descriptions
// of both the BroadcastSelectOp and the SelectOp, the original conversions
// from BroadcastSelectOp, and the XlaBuilder implementation. The implementation
// in XlaBuilder::TernaryOp is taken as authoritative, since that is the oldest
// code. Note that in that code, pred=lhs, onTrue=rhs, onFalse=ehs.
// In that implementation, there can only be one non-scalar shape in
// {pred, onTrue, onFalse} and any of them can be scalar (in violation of the
// specification). Since they are all assumed to be the same shape, the
// result shape is the first non-scalar of {pred, onTrue, onFalse}. Then
// any scalars are broadcast to that shape.
struct ConvertSelectOp : public OpConversionPattern<chlo::BroadcastSelectOp> {
  using OpConversionPattern<chlo::BroadcastSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      chlo::BroadcastSelectOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    chlo::BroadcastSelectOp::Adaptor transformed(operands);
    Location loc = op.getLoc();

    // Only support ranked operands.
    Value pred = transformed.pred();
    Value onTrue = transformed.on_true();
    Value onFalse = transformed.on_false();
    auto predType = pred.getType().dyn_cast<RankedTensorType>();
    auto onTrueType = onTrue.getType().dyn_cast<RankedTensorType>();
    auto onFalseType = onFalse.getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!predType || !onTrueType || !onFalseType || !resultType) {
      return rewriter.notifyMatchFailure(op, "cannot convert unranked tensors");
    }

    // Short-circuit if all types are statically equal.
    if (predType == onTrueType && predType == onFalseType) {
      // No broadcasting. This includes the 0d -> 0d case.
      rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, resultType, pred, onTrue,
                                                  onFalse);
      return success();
    }

    // Determine which component will be taken as the result shape.
    SmallVector<Value, 3> resultCandidates = {pred, onTrue, onFalse};
    Value nonScalarResult;
    for (Value resultCandidate : resultCandidates) {
      auto t = resultCandidate.getType().cast<RankedTensorType>();
      if (t.getRank() > 0) {
        if (nonScalarResult &&
            nonScalarResult.getType().cast<RankedTensorType>().getShape() !=
                t.getShape()) {
          // Since the spec is ill-defined on this point, don't trust the
          // verifier and make sure to avoid the situation in all builds.
          return rewriter.notifyMatchFailure(op, "mismatched select shapes");
        }
        nonScalarResult = resultCandidate;
      }
    }
    // Must be true per the equality early-exit above.
    assert(nonScalarResult && "must have a non-scalar result");
    auto nonScalarResultType =
        nonScalarResult.getType().cast<RankedTensorType>();

    // Compute result extents.
    int resultRank = nonScalarResultType.getRank();
    SmallVector<Extent> resultExtents;
    resultExtents.reserve(resultRank);
    appendExtents(rewriter, loc, resultExtents, nonScalarResult,
                  nonScalarResultType);

    // Broadcast any scalars.
    if (predType.getRank() == 0) {
      pred = broadcastScalar(rewriter, loc, pred, resultExtents);
    }
    if (onTrueType.getRank() == 0) {
      onTrue = broadcastScalar(rewriter, loc, onTrue, resultExtents);
    }
    if (onFalseType.getRank() == 0) {
      onFalse = broadcastScalar(rewriter, loc, onFalse, resultExtents);
    }

    rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, resultType, pred, onTrue,
                                                onFalse);
    return success();
  }
};

}  // namespace

}  // namespace iree_compiler
}  // namespace mlir

void mlir::iree_compiler::populateHLOBroadcastingToLinalgPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
#define POPULATE_SIMPLE_BCAST(ChloOp, HloOp)                          \
  patterns.insert<ConvertTrivialNonBroadcastBinaryOp>(                \
      context, typeConverter, 10,                                     \
      SimpleBinaryBroadcastingAdaptor<ChloOp, HloOp>::getInstance()); \
  patterns.insert<ConvertRankedBroadcastBinaryOp>(                    \
      context, typeConverter, 5,                                      \
      SimpleBinaryBroadcastingAdaptor<ChloOp, HloOp>::getInstance());

  POPULATE_SIMPLE_BCAST(chlo::BroadcastAddOp, mhlo::AddOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastAndOp, mhlo::AndOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastAtan2Op, mhlo::Atan2Op);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastComplexOp, mhlo::ComplexOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastDivOp, mhlo::DivOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastMaxOp, mhlo::MaxOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastMinOp, mhlo::MinOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastMulOp, mhlo::MulOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastOrOp, mhlo::OrOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastPolygammaOp, chlo::PolygammaOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastPowOp, mhlo::PowOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastRemOp, mhlo::RemOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastShiftLeftOp, mhlo::ShiftLeftOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastShiftRightArithmeticOp,
                        mhlo::ShiftRightArithmeticOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastShiftRightLogicalOp,
                        mhlo::ShiftRightLogicalOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastSubOp, mhlo::SubOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastXorOp, mhlo::XorOp);
  POPULATE_SIMPLE_BCAST(chlo::BroadcastZetaOp, chlo::ZetaOp);

  // Special case for Compare (not a simple signature).
  patterns.insert<ConvertTrivialNonBroadcastBinaryOp>(
      context, typeConverter, 10,
      CompareBinaryBroadcastingAdaptor::getInstance());
  patterns.insert<ConvertRankedBroadcastBinaryOp>(
      context, typeConverter, 5,
      CompareBinaryBroadcastingAdaptor::getInstance());

  // Other ops.
  // TODO: Remove the benefit after it is removed upstream.
  patterns.insert<ConvertSelectOp>(typeConverter, context, 1000);
  patterns.insert<ConvertConstantLikeOp>(typeConverter, context);

  // Make mixed scalar broadcasting of Clamp explicit.
  // NOTE: Because we are doing a full conversion out of HLO, we do not use
  // the corresponding setup legality, since that explicitly marks clamp as
  // conditionally legal.
  // TODO: Rename this upstream or find a better place to shove it.
  mhlo::PopulateMaterializeBroadcastsPatterns(context, &patterns);
}
