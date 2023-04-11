// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Patterns for lowering from dynamic-shape sensitive CHLO/MHLO ops. This
// primarily involves broadcasting ops but also includes other ops that have
// an impact on dynamic shape conversions.

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_chlo_to_hlo_op.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {

// -----------------------------------------------------------------------------
// Broadcasting utilities
// -----------------------------------------------------------------------------

/// Whether an element type is legal for codegen via linalg on IREE.
bool isElementTypeLegalForCodegen(Type t) { return !t.isa<ComplexType>(); }

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
    int nLoops, int nReduction) {
  SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                          utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(int nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

// Holds a static extent or Value for dynamic extents.
class Extent {
 public:
  Extent() {}
  Extent(int64_t extent) : extent(extent) {}
  Extent(Value value) : value(value) {}

  bool isStatic() const { return !value; }
  bool isUnitExtent() const { return isStatic() && getStatic() == 1; }
  int64_t getStatic() const {
    assert(isStatic());
    return extent;
  }
  Value getValue() const {
    assert(!isStatic());
    return value;
  }

  Value convertToValue(OpBuilder &builder, Location loc) {
    if (!isStatic()) return getValue();
    return builder.create<arith::ConstantIndexOp>(loc, getStatic());
  }

 private:
  int64_t extent;
  Value value;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Extent &extent) {
  if (extent.isStatic()) {
    os << "DIM[" << extent.getStatic() << "]";
  } else {
    os << "DIM[" << extent.getValue() << "]";
  }
  return os;
}

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
      resultShape.push_back(ShapedType::kDynamic);
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
  Value init = builder.create<tensor::EmptyOp>(
      loc, resultShape, operandType.getElementType(), dynDims);
  auto generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{init.getType()}, ValueRange{operand},
      /*outputBuffers=*/ValueRange{init},
      llvm::ArrayRef({
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

std::optional<Extent> computeBinaryResultExtent(OpBuilder &builder,
                                                Location loc, Extent &lhsDim,
                                                Extent &rhsDim,
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
      return std::nullopt;
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

  Value isEqual = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                lhsExtentValue, rhsExtentValue);
  builder.create<cf::AssertOp>(
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
  return Extent(lhsExtentValue);
}

std::optional<Extent> computeTernaryResultExtent(OpBuilder &builder,
                                                 Location loc, Extent &aValue,
                                                 Extent &bValue, Extent &cValue,
                                                 bool &isAExpansion,
                                                 bool &isBExpansion,
                                                 bool &isCExpansion) {
  // Collect non unit extents (which includes, implicitly, dynamic dims).
  SmallVector<Extent> nonUnitExtents;
  if (!aValue.isUnitExtent()) nonUnitExtents.push_back(aValue);
  if (!bValue.isUnitExtent()) nonUnitExtents.push_back(bValue);
  if (!cValue.isUnitExtent()) nonUnitExtents.push_back(cValue);

  // Early exit if all unit extents.
  if (nonUnitExtents.empty()) {
    isAExpansion = false;
    isBExpansion = false;
    isCExpansion = false;
    return aValue;
  }

  // Are any a unit?
  bool hasUnitExtent = false;
  if (aValue.isUnitExtent()) hasUnitExtent = true;
  if (bValue.isUnitExtent()) hasUnitExtent = true;
  if (cValue.isUnitExtent()) hasUnitExtent = true;

  // Mark expansion for any unit.
  if (hasUnitExtent) {
    if (aValue.isUnitExtent()) isAExpansion = true;
    if (bValue.isUnitExtent()) isBExpansion = true;
    if (cValue.isUnitExtent()) isCExpansion = true;
  }

  // By default, compare against the first non unit extent; however, prefer
  // a static extent if present.
  int nonUnitCompareExtentIndex = 0;
  for (int i = 0, e = nonUnitExtents.size(); i < e; i++) {
    if (nonUnitExtents[i].isStatic()) nonUnitCompareExtentIndex = i;
  }

  // Generate checks for each non unit extent.
  for (int i = 0, e = nonUnitExtents.size(); i < e; i++) {
    if (i == nonUnitCompareExtentIndex) continue;
    Extent &cmpLhs = nonUnitExtents[nonUnitCompareExtentIndex];
    Extent &cmpRhs = nonUnitExtents[i];
    // Static check.
    if (cmpLhs.isStatic() && cmpRhs.isStatic()) {
      if (cmpLhs.getStatic() != cmpRhs.getStatic()) {
        // Statically illegal.
        emitError(loc) << "cannot broadcast extents of differing size unless "
                          "if one of them is 1 (got "
                       << cmpLhs.getStatic() << ", " << cmpRhs.getStatic()
                       << ")";
        return std::nullopt;
      }
      continue;
    }
    // Dynamic check.
    Value cmpLhsValue = cmpLhs.convertToValue(builder, loc);
    Value cmpRhsValue = cmpRhs.convertToValue(builder, loc);
    Value isEqual = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  cmpLhsValue, cmpRhsValue);
    builder.create<cf::AssertOp>(
        loc, isEqual,
        builder.getStringAttr("mismatched dynamic broadcast extents"));
  }

  // The result must be one of the non unit extents. Just take the one
  // used for comparison.
  return nonUnitExtents[nonUnitCompareExtentIndex];
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
      Value dim = builder.create<tensor::DimOp>(loc, v, i);
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
      chlo::ConstantLikeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType().cast<RankedTensorType>();
    if (!resultTy.hasRank())
      return rewriter.notifyMatchFailure(op, "only supports ranked");
    // Lower to MHLO constant if statically shaped.
    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(
          op, DenseElementsAttr::get(resultTy, op.getValue()));
      return success();
    }

    Location loc = op.getLoc();

    int resultRank = resultTy.getRank();
    SmallVector<Extent> resultExtents;
    resultExtents.reserve(resultRank);
    appendExtents(rewriter, loc, resultExtents, adaptor.getOperand(), resultTy);

    auto resultTy0D = RankedTensorType::get({}, resultTy.getElementType());
    Value scalarConst = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(resultTy0D, op.getValue()));
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
    auto broadcastDimensions =
        llvm::cast<FromOpTy>(op).getBroadcastDimensions();
    if (broadcastDimensions &&
        !hlo::isLegalNumpyRankedBroadcast(operands[0], operands[1],
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
        llvm::cast<chlo::BroadcastCompareOp>(op).getBroadcastDimensions();
    if (broadcastDimensions &&
        !hlo::isLegalNumpyRankedBroadcast(operands[0], operands[1],
                                          *broadcastDimensions)) {
      return failure();
    }
    return success();
  }
  BroadcastValues getFromBroadcastValues(Operation *op,
                                         ArrayRef<Value> operands) override {
    chlo::BroadcastCompareOpAdaptor adaptor(operands, op->getAttrDictionary());
    return std::make_pair(adaptor.getLhs(), adaptor.getRhs());
  }
  Operation *createTargetOperation(Location loc, Operation *op, Type resultType,
                                   ArrayRef<Value> operands,
                                   BroadcastValues broadcastValues,
                                   OpBuilder &builder) override {
    chlo::BroadcastCompareOpAdaptor adaptor(operands, op->getAttrDictionary());
    std::optional<chlo::ComparisonType> chloCmpType = adaptor.getCompareType();
    mhlo::ComparisonTypeAttr mhloCmpType;
    if (chloCmpType)
      mhloCmpType = mhlo::ComparisonTypeAttr::get(
          builder.getContext(), *chlo::mhloComparisonType(*chloCmpType));
    return builder.create<mhlo::CompareOp>(
        loc, resultType, broadcastValues.first, broadcastValues.second,
        *chlo::mhloComparisonDirection(adaptor.getComparisonDirection()),
        mhloCmpType);
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
    if (!isElementTypeLegalForCodegen(lhsType.getElementType()) ||
        !isElementTypeLegalForCodegen(rhsType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "not legal element type for codegen");
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
      auto resultExtent = computeBinaryResultExtent(
          rewriter, loc, lhsBcastExtents[i], rhsBcastExtents[i],
          isLhsExpansion[i], isRhsExpansion[i]);
      if (!resultExtent) {
        return rewriter.notifyMatchFailure(op,
                                           "could not compute result extent");
      }
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
    auto lhsType =
        bcastOperands.first.getType().template dyn_cast<RankedTensorType>();
    auto rhsType =
        bcastOperands.second.getType().template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "not ranked tensors");
    if (!isElementTypeLegalForCodegen(lhsType.getElementType()) ||
        !isElementTypeLegalForCodegen(rhsType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "not legal element type for codegen");
    }

    // Requires rank broadcast.
    if (lhsType.getRank() != rhsType.getRank())
      return rewriter.notifyMatchFailure(op, "not same rank");
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "not static shapes");

    for (auto [lhsExtent, rhsExtent] :
         llvm::zip_equal(lhsType.getShape(), rhsType.getShape())) {
      if (lhsExtent != rhsExtent) {
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
// This follows the new convention of SelectV2, which allows a true ternary
// select (whereas the original definition only supported one broadcasting
// value).
struct ConvertSelectOp : public OpConversionPattern<chlo::BroadcastSelectOp> {
  using OpConversionPattern<chlo::BroadcastSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      chlo::BroadcastSelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Only support ranked operands.
    Value pred = adaptor.getPred();
    Value thenValue = adaptor.getOnTrue();
    Value elseValue = adaptor.getOnFalse();
    auto predType = pred.getType().dyn_cast<RankedTensorType>();
    auto thenType = thenValue.getType().dyn_cast<RankedTensorType>();
    auto elseType = elseValue.getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!predType || !thenType || !elseType || !resultType) {
      return rewriter.notifyMatchFailure(op, "cannot convert unranked tensors");
    }
    if (!isElementTypeLegalForCodegen(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "not legal element type for codegen");
    }

    // Short-circuit if all types are statically equal.
    if (predType == thenType && predType == elseType) {
      // No broadcasting. This includes the 0d -> 0d case.
      rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, resultType, pred,
                                                  thenValue, elseValue);
      return success();
    }

    // Full ternary broadcast. See ConvertBroadcastBinaryOp for the
    // simplified version.
    // Extract the original extents.
    SmallVector<Extent> predOrigExtents;
    predOrigExtents.reserve(predType.getRank());
    appendExtents(rewriter, loc, predOrigExtents, pred, predType);
    SmallVector<Extent> thenOrigExtents;
    thenOrigExtents.reserve(thenType.getRank());
    appendExtents(rewriter, loc, thenOrigExtents, thenValue, thenType);
    SmallVector<Extent> elseOrigExtents;
    elseOrigExtents.reserve(elseType.getRank());
    appendExtents(rewriter, loc, elseOrigExtents, elseValue, elseType);

    // Left pad with 1-extents to the result rank.
    int resultRank = std::max(std::max(predType.getRank(), thenType.getRank()),
                              elseType.getRank());
    SmallVector<Extent> predBcastExtents;
    predBcastExtents.reserve(resultRank);
    padExtents(predBcastExtents, resultRank - predType.getRank());
    predBcastExtents.append(predOrigExtents);

    SmallVector<Extent> thenBcastExtents;
    thenBcastExtents.reserve(resultRank);
    padExtents(thenBcastExtents, resultRank - thenType.getRank());
    thenBcastExtents.append(thenOrigExtents);

    SmallVector<Extent> elseBcastExtents;
    elseBcastExtents.reserve(resultRank);
    padExtents(elseBcastExtents, resultRank - elseType.getRank());
    elseBcastExtents.append(elseOrigExtents);

    // Compute the result extents.
    SmallVector<Extent> resultExtents(resultRank);
    SmallVector<bool> isPredExpansion(resultRank);
    SmallVector<bool> isThenExpansion(resultRank);
    SmallVector<bool> isElseExpansion(resultRank);
    bool predNeedsBroadcast = resultRank != predType.getRank();
    bool thenNeedsBroadcast = resultRank != thenType.getRank();
    bool elseNeedsBroadcast = resultRank != elseType.getRank();
    for (int i = 0; i < resultRank; i++) {
      auto resultExtent = computeTernaryResultExtent(
          rewriter, loc, predBcastExtents[i], thenBcastExtents[i],
          elseBcastExtents[i], isPredExpansion[i], isThenExpansion[i],
          isElseExpansion[i]);
      if (!resultExtent) {
        return rewriter.notifyMatchFailure(op,
                                           "could not compute result extent");
      }
      resultExtents[i] = *resultExtent;
      if (isPredExpansion[i]) predNeedsBroadcast = true;
      if (isThenExpansion[i]) thenNeedsBroadcast = true;
      if (isElseExpansion[i]) elseNeedsBroadcast = true;
    }

    // Broadcast all.
    Value predBcast =
        predNeedsBroadcast
            ? broadcast(rewriter, loc, pred, resultExtents, isPredExpansion)
            : pred;
    Value thenBcast = thenNeedsBroadcast
                          ? broadcast(rewriter, loc, thenValue, resultExtents,
                                      isThenExpansion)
                          : thenValue;
    Value elseBcast = elseNeedsBroadcast
                          ? broadcast(rewriter, loc, elseValue, resultExtents,
                                      isElseExpansion)
                          : elseValue;

    rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, resultType, predBcast,
                                                thenBcast, elseBcast);
    return success();
  }
};

// Fallback conversion of mhlo.dynamic_reshape to flow.tensor.reshape.
// This is not the most optimal way to lower most reshapes, and higher
// benefit patterns should match more specific ops and lower them to
// Linalg expanding and contracting reshapes.
//
// Note that as a low-level op, it is assumed that invariants have been
// satisfied externally in some fashion and further checks are not inserted
// at this time. This may need to be re-evaluated as more user-driven
// reshapes are permitted.
struct ConvertDynamicReshapeOp
    : public OpConversionPattern<mhlo::DynamicReshapeOp> {
  using OpConversionPattern<mhlo::DynamicReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getOperand();
    Value outputShape = adaptor.getOutputShape();
    auto outputShapeType = outputShape.getType().dyn_cast<RankedTensorType>();
    auto resultType = typeConverter->convertType(op.getType())
                          .dyn_cast_or_null<RankedTensorType>();
    if (!outputShapeType || !resultType) {
      return rewriter.notifyMatchFailure(op, "not ranked");
    }
    SmallVector<Value> targetDims;
    assert(resultType.getRank() == outputShapeType.getNumElements() &&
           "mismatched rank");
    for (int i = 0, e = resultType.getRank(); i < e; ++i) {
      if (resultType.isDynamicDim(i)) {
        Value index = rewriter.create<arith::ConstantIndexOp>(loc, i);
        targetDims.push_back(
            rewriter.create<tensor::ExtractOp>(loc, outputShape, index));
      }
    }

    SmallVector<Value> castedTargetDims;
    for (Value dim : targetDims) {
      if (dim.getType().isa<IntegerType>()) {
        dim = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                  dim);
      }
      castedTargetDims.push_back(dim);
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, resultType, input, castedTargetDims);
    return success();
  }
};

}  // namespace

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir

void mlir::iree_compiler::MHLO::populateMHLOBroadcastingToLinalgPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet &patterns) {
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
  POPULATE_SIMPLE_BCAST(chlo::BroadcastSubOp, mhlo::SubtractOp);
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
  patterns.insert<ConvertDynamicReshapeOp>(typeConverter, context);

  // Make mixed scalar broadcasting of Clamp explicit.
  // NOTE: Because we are doing a full conversion out of HLO, we do not use
  // the corresponding setup legality, since that explicitly marks clamp as
  // conditionally legal.
  // TODO: Rename this upstream or find a better place to shove it.
  mhlo::populateMaterializeBroadcastsPatterns(context, &patterns);
}
