// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <cassert>

namespace mlir::iree_compiler::IREE::Encoding {

AffineMap getRowMajorStorageElementOffsetMap(RankedTensorType type,
                                             int64_t storageBits,
                                             ArrayRef<int64_t> padding) {
  if (!padding.empty() &&
      (padding.size() != type.getRank() ||
       llvm::any_of(padding, [](int64_t value) { return value < 0; }))) {
    return {};
  }
  MLIRContext *ctx = type.getContext();
  AffineExpr offset = getAffineConstantExpr(0, ctx);
  int64_t dynamicIndex = 0;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    AffineExpr extent = type.isDynamicDim(i)
                            ? getAffineSymbolExpr(dynamicIndex++, ctx)
                            : getAffineConstantExpr(type.getDimSize(i), ctx);
    if (!padding.empty()) {
      extent = extent + padding[i];
    }
    offset = offset * extent + getAffineDimExpr(i, ctx);
  }
  return AffineMap::get(type.getRank(), dynamicIndex,
                        (offset * storageBits).floorDiv(8));
}

// Storage offsets use unsigned arithmetic without signed-overflow assumptions.
Value buildStorageElementByteOffset(Location loc, OpBuilder &builder,
                                    AffineExpr expr, ValueRange indices,
                                    ValueRange dynamicDims) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    return indices[dim.getPosition()];
  }
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    return dynamicDims[symbol.getPosition()];
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
    return builder.createOrFold<arith::ConstantIndexOp>(loc,
                                                        constant.getValue());
  }
  auto binary = cast<AffineBinaryOpExpr>(expr);
  Value lhs = buildStorageElementByteOffset(loc, builder, binary.getLHS(),
                                            indices, dynamicDims);
  Value rhs = buildStorageElementByteOffset(loc, builder, binary.getRHS(),
                                            indices, dynamicDims);
  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return builder.createOrFold<arith::AddIOp>(loc, lhs, rhs);
  case AffineExprKind::Mul:
    return builder.createOrFold<arith::MulIOp>(loc, lhs, rhs);
  case AffineExprKind::FloorDiv:
    return builder.createOrFold<arith::DivUIOp>(loc, lhs, rhs);
  case AffineExprKind::CeilDiv:
    return builder.createOrFold<arith::CeilDivUIOp>(loc, lhs, rhs);
  case AffineExprKind::Mod:
    return builder.createOrFold<arith::RemUIOp>(loc, lhs, rhs);
  default:
    llvm_unreachable("expected an affine binary expression");
  }
}

Value SerializableAttr::calculateStorageElementOffsetInBytes(
    Location loc, OpBuilder &builder, RankedTensorType type,
    ValueRange dynamicDims, ValueRange indices) {
  AffineMap map = getStorageElementOffsetMap(type);
  if (!map) {
    return {};
  }
  return buildStorageElementByteOffset(loc, builder, map.getResult(0), indices,
                                       dynamicDims);
}

// static
bool SerializableAttr::areCompatible(Attribute lhs, Attribute rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhsEncoding = dyn_cast_if_present<SerializableAttr>(lhs);
  auto rhsEncoding = dyn_cast_if_present<SerializableAttr>(rhs);

  return (!lhsEncoding || lhsEncoding.isCompatibleWith(rhsEncoding)) &&
         (!rhsEncoding || rhsEncoding.isCompatibleWith(lhsEncoding));
}

/// Given a LinalgOp and one of its OpOperands, return the element type,
/// inferring unsignedness from the body of the LinalgOp.
static Type getContractionInputTypeWithSignedness(OpBuilder &builder,
                                                  linalg::LinalgOp linalgOp,
                                                  OpOperand *operand) {
  auto elemType = getElementTypeOrSelf(operand->get().getType());
  // Infer if unsigned from body ops
  Value blockArg = linalgOp.getMatchingBlockArgument(operand);
  for (auto bodyCastOp : blockArg.getParentBlock()->getOps<arith::ExtUIOp>()) {
    if (bodyCastOp->getOperand(0) == blockArg) {
      return builder.getIntegerType(elemType.getIntOrFloatBitWidth(),
                                    /*isSigned=*/false);
    }
  }
  return elemType;
}

/// Extract dynamic dimension values for a linalg operation in loop order.
/// Returns values only for dimensions that are dynamic.
static SmallVector<Value> getDynamicLoopDims(OpBuilder &builder,
                                             linalg::LinalgOp linalgOp) {
  SmallVector<Range> loopRanges =
      linalgOp.createLoopRanges(builder, linalgOp.getLoc());

  SmallVector<Value> dynamicDims;
  for (Range &range : loopRanges) {
    // OpFoldResult is either Value (dynamic) or Attribute (static).
    if (auto value = dyn_cast<Value>(range.size)) {
      dynamicDims.push_back(value);
    }
  }
  return dynamicDims;
}

FailureOr<OpEncodingProperties>
SerializableAttr::getEncodingProperties(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return failure();
  }

  MLIRContext *ctx = op->getContext();
  OpBuilder builder(op);

  OpEncodingProperties props;
  SmallVector<Type> elemTypes;
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  SmallVector<int64_t> iterationSizes = linalgOp.getStaticLoopRanges();
  EncodingOpType opType;

  // Extract dynamic loop dimensions.
  SmallVector<Value> dynamicDims = getDynamicLoopDims(builder, linalgOp);

  auto addEncoding = [&](int64_t operandIndex) {
    return EncodingProperties{
        EncodingAttr::get(ctx, operandIndex, opType, elemTypes,
                          /*originalElementType=*/{}, maps, iterationSizes),
        dynamicDims};
  };

  // Return encoding properties for contraction operations.
  if (linalg::isaContractionOpInterface(linalgOp)) {
    // Get element types with signedness inference.
    Type lhsElemType = getContractionInputTypeWithSignedness(
        builder, linalgOp, linalgOp.getDpsInputOperand(0));
    Type rhsElemType = getContractionInputTypeWithSignedness(
        builder, linalgOp, linalgOp.getDpsInputOperand(1));
    Type outElemType = getContractionInputTypeWithSignedness(
        builder, linalgOp, linalgOp.getDpsInitOperand(0));

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }

    elemTypes = {lhsElemType, rhsElemType, outElemType};
    opType = EncodingOpType::matmul;

    props.operands.push_back(addEncoding(MATMUL_LHS));
    props.operands.push_back(addEncoding(MATMUL_RHS));
    props.inits.push_back(addEncoding(MATMUL_RESULT));

    return props;
  }

  // Return encoding properties for scaled contraction operations.
  if (IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
    // Get element types for scaled matmul operands.
    Type lhsElemType =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get().getType());
    Type rhsElemType =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(1)->get().getType());
    Type lhsScalesElemType =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(2)->get().getType());
    Type rhsScalesElemType =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(3)->get().getType());
    Type outElemType =
        getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());

    elemTypes = {lhsElemType, rhsElemType, lhsScalesElemType, rhsScalesElemType,
                 outElemType};
    opType = EncodingOpType::scaled_matmul;

    props.operands.push_back(addEncoding(SCALED_MATMUL_LHS));
    props.operands.push_back(addEncoding(SCALED_MATMUL_RHS));
    props.operands.push_back(addEncoding(SCALED_MATMUL_LHS_SCALES));
    props.operands.push_back(addEncoding(SCALED_MATMUL_RHS_SCALES));
    props.inits.push_back(addEncoding(SCALED_MATMUL_RESULT));

    return props;
  }

  // Return encoding properties for convolutions operations.
  if (linalg::isaConvolutionOpInterface(linalgOp)) {
    Type inputElemType =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get().getType());
    Type filterElemType =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(1)->get().getType());
    Type outputElemType =
        getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
    elemTypes = {inputElemType, filterElemType, outputElemType};
    opType = EncodingOpType::conv;
    props.operands.push_back(addEncoding(CONV_IN));
    props.operands.push_back(addEncoding(CONV_FILTER));
    props.inits.push_back(addEncoding(CONV_OUT));
    return props;
  }

  // Return failure for unsupported operations.
  return failure();
}

static std::string stringifyMatmulOperandIndex(IntegerAttr valueAttr) {
  uint64_t value = valueAttr.getValue().getZExtValue();
  switch (value) {
  case MATMUL_LHS:
    return "LHS";
  case MATMUL_RHS:
    return "RHS";
  case MATMUL_RESULT:
    return "RESULT";
  default:
    assert(false && "invalid index");
    return "";
  }
}

static std::string stringifyScaledMatmulOperandIndex(IntegerAttr valueAttr) {
  uint64_t value = valueAttr.getValue().getZExtValue();
  switch (value) {
  case SCALED_MATMUL_LHS:
    return "LHS";
  case SCALED_MATMUL_LHS_SCALES:
    return "LHS_SCALES";
  case SCALED_MATMUL_RHS:
    return "RHS";
  case SCALED_MATMUL_RHS_SCALES:
    return "RHS_SCALES";
  case SCALED_MATMUL_RESULT:
    return "RESULT";
  default:
    assert(false && "invalid index");
    return "";
  }
}

static std::string stringifyConvOperandIndex(IntegerAttr valueAttr) {
  uint64_t value = valueAttr.getValue().getZExtValue();
  switch (value) {
  case CONV_IN:
    return "CONV_IN";
  case CONV_FILTER:
    return "CONV_FILTER";
  case CONV_OUT:
    return "CONV_OUT";
  default:
    assert(false && "invalid index");
    return "";
  }
}

std::string stringifyOperandIndex(EncodingOpType opType,
                                  IntegerAttr valueAttr) {
  switch (opType) {
  case EncodingOpType::matmul:
    return stringifyMatmulOperandIndex(valueAttr);
  case EncodingOpType::scaled_matmul:
    return stringifyScaledMatmulOperandIndex(valueAttr);
  case EncodingOpType::conv:
    return stringifyConvOperandIndex(valueAttr);
  default:
    assert(false && "invalid opType");
    return "";
  }
}

MatmulNarrowDim getMatmulNarrowDim(linalg::LinalgOp linalgOp,
                                   int narrowThreshold) {
  linalg::ContractionDimensions cDims =
      linalg::inferContractionDims(linalgOp).value();
  AffineMap map = linalgOp.getIndexingMapsArray().back();
  auto outType = cast<ShapedType>(linalgOp.getDpsInits()[0].getType());
  auto getOutputSizeAtDimPos = [=](unsigned dimPos) -> int64_t {
    return outType.getDimSize(
        map.getResultPosition(getAffineDimExpr(dimPos, linalgOp->getContext()))
            .value());
  };
  // M or N can be empty instead of having an explicit dim size of 1 for matvec
  // and vecmat, so set to 1 if empty.
  int64_t mSize = cDims.m.empty() ? 1 : getOutputSizeAtDimPos(cDims.m[0]);
  int64_t nSize = cDims.n.empty() ? 1 : getOutputSizeAtDimPos(cDims.n[0]);

  MatmulNarrowDim narrowM, narrowN;
  if (ShapedType::isStatic(mSize) && mSize < narrowThreshold) {
    narrowM = {/*dim=*/MatmulNarrowDim::Dim::M, /*size=*/mSize};
  }
  if (ShapedType::isStatic(nSize) && nSize < narrowThreshold) {
    narrowN = {/*dim=*/MatmulNarrowDim::Dim::N, /*size=*/nSize};
  }

  return (narrowM && (!narrowN || mSize <= nSize)) ? narrowM : narrowN;
}

} // namespace mlir::iree_compiler::IREE::Encoding
