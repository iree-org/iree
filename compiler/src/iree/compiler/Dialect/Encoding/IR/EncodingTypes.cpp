// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <cassert>

namespace mlir::iree_compiler::IREE::Encoding {

// static
bool SerializableAttr::areCompatible(Attribute lhs, Attribute rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhsEncoding = llvm::dyn_cast_or_null<SerializableAttr>(lhs);
  auto rhsEncoding = llvm::dyn_cast_or_null<SerializableAttr>(rhs);
  if (!lhsEncoding || !rhsEncoding) {
    return false;
  }
  return lhsEncoding.isCompatibleWith(rhsEncoding) &&
         rhsEncoding.isCompatibleWith(lhsEncoding);
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
  auto outType = llvm::cast<ShapedType>(linalgOp.getDpsInits()[0].getType());
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
  if (!ShapedType::isDynamic(mSize) && mSize < narrowThreshold) {
    narrowM = {/*dim=*/MatmulNarrowDim::Dim::M, /*size=*/mSize};
  }
  if (!ShapedType::isDynamic(nSize) && nSize < narrowThreshold) {
    narrowN = {/*dim=*/MatmulNarrowDim::Dim::N, /*size=*/nSize};
  }

  return (narrowM && (!narrowN || mSize <= nSize)) ? narrowM : narrowN;
}

} // namespace mlir::iree_compiler::IREE::Encoding
