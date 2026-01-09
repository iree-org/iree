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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <cassert>

namespace mlir::iree_compiler::IREE::Encoding {

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

/// Extract only the dynamic dimension values for a contraction operation.
/// Returns values only for dimensions that are dynamic in iteration_sizes.
/// The returned values are ordered to match iteration_sizes: Batch, M, N, K.
/// This matches the standard loop dimension ordering for matmul-like ops.
static SmallVector<Value> getContractionDynamicDims(OpBuilder &builder,
                                                    linalg::LinalgOp linalgOp) {
  Location loc = linalgOp.getLoc();

  // Infer contraction dimensions (B, M, N, K).
  auto cDims = linalg::inferContractionDims(linalgOp);
  if (failed(cDims)) {
    return {};
  }

  // Get static loop ranges to determine which dimensions are dynamic.
  SmallVector<int64_t> iterationSizes = linalgOp.getStaticLoopRanges();

  // Get the output tensor to extract B, M and N from.
  Value output = linalgOp.getDpsInits()[0];
  AffineMap outputMap = linalgOp.getIndexingMapsArray().back();

  // Helper to get dimension value from a tensor given a loop dimension.
  // Note: This is only called for dimensions that are dynamic in
  // iteration_sizes. For standard matmul/batch_matmul, all dynamic dimensions
  // should be mappable to tensor dimensions.
  auto getDimValue = [&](Value tensor, AffineMap map,
                         unsigned loopDim) -> Value {
    auto dimPos = map.getResultPosition(
        getAffineDimExpr(loopDim, linalgOp->getContext()));
    if (!dimPos) {
      // This dimension is not present in this operand. This is unexpected for
      // dynamic dimensions in standard contractions, but we handle it
      // gracefully by returning constant 1.
      return arith::ConstantIndexOp::create(builder, loc, 1);
    }
    return tensor::DimOp::create(builder, loc, tensor, *dimPos);
  };

  // Extract dynamic dimension values in order: B, M, N, K.
  // This matches the iteration_sizes order for standard matmul-like ops.
  SmallVector<Value> dynamicDims;

  // Process Batch dimensions (come first in iteration order).
  for (unsigned batchDim : cDims->batch) {
    if (batchDim < iterationSizes.size() &&
        ShapedType::isDynamic(iterationSizes[batchDim])) {
      dynamicDims.push_back(getDimValue(output, outputMap, batchDim));
    }
  }

  // Process M dimensions.
  for (unsigned mDim : cDims->m) {
    if (mDim < iterationSizes.size() &&
        ShapedType::isDynamic(iterationSizes[mDim])) {
      dynamicDims.push_back(getDimValue(output, outputMap, mDim));
    }
  }

  // Process N dimensions.
  for (unsigned nDim : cDims->n) {
    if (nDim < iterationSizes.size() &&
        ShapedType::isDynamic(iterationSizes[nDim])) {
      dynamicDims.push_back(getDimValue(output, outputMap, nDim));
    }
  }

  // Process K dimensions (extracted from LHS).
  Value lhs = linalgOp.getDpsInputs()[0];
  AffineMap lhsMap = linalgOp.getIndexingMapsArray()[0];
  for (unsigned kDim : cDims->k) {
    if (kDim < iterationSizes.size() &&
        ShapedType::isDynamic(iterationSizes[kDim])) {
      dynamicDims.push_back(getDimValue(lhs, lhsMap, kDim));
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

  // Extract M, N, K dynamic dimensions (shared by all contraction types).
  SmallVector<Value> dynamicDims = getContractionDynamicDims(builder, linalgOp);

  auto addEncoding = [&](int64_t operandIndex) {
    return EncodingProperties{EncodingAttr::get(ctx, operandIndex, opType,
                                                elemTypes, maps,
                                                iterationSizes),
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

  // Return failure for unsupported operations.
  return failure();
}

std::string stringifyOperandIndex(IntegerAttr valueAttr) {
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
