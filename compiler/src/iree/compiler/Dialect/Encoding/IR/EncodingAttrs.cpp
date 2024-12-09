// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Encoding {

EncodingAttr EncodingAttr::get(MLIRContext *ctx, int64_t operandIndex,
                               EncodingOpType opType, ArrayRef<Type> elemTypes,
                               ArrayRef<AffineMap> maps,
                               std::optional<AffineMap> bcastMap,
                               ArrayRef<int64_t> roundDimsTo,
                               ArrayRef<Attribute> layouts) {
  Builder b(ctx);
  auto opTypeAttr = EncodingOpTypeAttr::get(ctx, opType);
  auto roundDimsToAttr = roundDimsTo.empty()
                             ? DenseI64ArrayAttr()
                             : b.getDenseI64ArrayAttr(roundDimsTo);
  auto bcastMapAttr = bcastMap.has_value()
                          ? AffineMapAttr::get(bcastMap.value())
                          : AffineMapAttr();
  auto layoutsAttr = layouts.empty() ? ArrayAttr() : b.getArrayAttr(layouts);
  return get(ctx, b.getIndexAttr(operandIndex), opTypeAttr,
             b.getTypeArrayAttr(elemTypes), b.getAffineMapArrayAttr(maps),
             bcastMapAttr, roundDimsToAttr, layoutsAttr);
}

AffineMap EncodingAttr::getMapForOperandIndex() const {
  auto index = getOperandIndex().getValue().getZExtValue();
  switch (index) {
  case MATMUL_LHS:
  case MATMUL_RHS:
  case MATMUL_RESULT: {
    auto indexingMap =
        llvm::cast<AffineMapAttr>(getUserIndexingMaps()[index]).getAffineMap();
    if (auto bcastMap = getBcastMap()) {
      indexingMap = bcastMap.getAffineMap().compose(indexingMap);
    }
    return indexingMap;
  }
  default:
    return AffineMap();
  }
}

std::optional<unsigned>
EncodingAttr::mapDimToOperandIndex(int64_t dimPos) const {
  return getMapForOperandIndex().getResultPosition(
      getAffineDimExpr(dimPos, getContext()));
}

MatmulNarrowDim getMatmulNarrowDim(linalg::LinalgOp linalgOp,
                                   int narrowThreshold) {
  linalg::ContractionDimensions cDims =
      linalg::inferContractionDims(linalgOp).value();
  auto map = linalgOp.getIndexingMapsArray().back();
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

ArrayRef<int64_t> EncodingAttr::getRoundDimsToArray() const {
  auto roundDimsTo = getRoundDimsTo();
  if (!roundDimsTo) {
    return {};
  }
  return llvm::cast<DenseI64ArrayAttr>(roundDimsTo).asArrayRef();
}

SmallVector<Type> EncodingAttr::getElementTypesArray() {
  return llvm::map_to_vector(getElementTypes().getValue(), [](Attribute a) {
    return llvm::cast<TypeAttr>(a).getValue();
  });
}

EncodingAttr EncodingAttr::clone(AffineMap bcastMap) {
  return get(bcastMap.getContext(), getOperandIndex(), getOpType(),
             getElementTypes(), getUserIndexingMaps(),
             AffineMapAttr::get(bcastMap), getRoundDimsTo(), getLayouts());
}

Value EncodingAttr::calculateStorageSizeInBytes(Location loc,
                                                OpBuilder &builder,
                                                RankedTensorType type,
                                                ValueRange dynamicDims) const {
  SmallVector<OpFoldResult> shape =
      getMixedValues(type.getShape(), dynamicDims, builder);
  ArrayRef<int64_t> roundDimsTo = getRoundDimsToArray();
  FailureOr<linalg::ContractionDimensions> cDims =
      getEncodingContractionDims(*this);
  auto pad = [&](int dim, int value) {
    std::optional<unsigned> maybeMappedDim = mapDimToOperandIndex(dim);
    if (!maybeMappedDim) {
      return;
    }
    unsigned mappedDim = maybeMappedDim.value();
    AffineExpr expr = builder.getAffineDimExpr(0);
    shape[mappedDim] = affine::makeComposedFoldedAffineApply(
        builder, loc, expr.ceilDiv(value) * value, {shape[mappedDim]});
  };
  for (auto m : cDims->m) {
    pad(m, roundDimsTo[0]);
  }
  for (auto n : cDims->n) {
    pad(n, roundDimsTo[1]);
  }
  for (auto k : cDims->k) {
    pad(k, roundDimsTo[2]);
  }

  constexpr int64_t kNumBitsInByte = 8;
  unsigned elementBits = IREE::Util::getTypeBitWidth(type.getElementType());
  int64_t numBytesPerElem = 1;
  if (elementBits > kNumBitsInByte) {
    numBytesPerElem *=
        IREE::Util::getRoundedElementByteWidth(type.getElementType());
  }
  OpFoldResult res = builder.getIndexAttr(numBytesPerElem);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  for (auto dimSize : shape) {
    res = affine::makeComposedFoldedAffineApply(builder, loc, d0 * d1,
                                                {res, dimSize});
  }
  // Always pack the elements back-to-back for subtypes.
  if (elementBits < kNumBitsInByte) {
    if (kNumBitsInByte % elementBits) {
      assert(false && "unsupported subtype");
      return Value();
    }
    unsigned divisor = kNumBitsInByte / elementBits;
    AffineExpr expr = builder.getAffineDimExpr(0);
    res = affine::makeComposedFoldedAffineApply(
        builder, loc, expr.ceilDiv(divisor) * divisor, {res});
  }
  return getValueOrCreateConstantIntOp(builder, loc, res);
}

MatmulNarrowDim getMatmulNarrowDim(EncodingAttr encoding) {
  if (encoding.getOpType().getValue() != EncodingOpType::matmul) {
    return {};
  }
  ArrayRef<int64_t> roundDimsTo = encoding.getRoundDimsToArray();
  if (roundDimsTo.empty()) {
    return {};
  }
  int m = roundDimsTo[0];
  int n = roundDimsTo[1];
  if (m < n) {
    return {MatmulNarrowDim::Dim::M, m};
  }
  if (n < m) {
    return {MatmulNarrowDim::Dim::N, n};
  }
  return {};
}

bool isNarrowNResult(EncodingAttr encoding) {
  if (encoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RESULT) {
    return false;
  }

  return IREE::Encoding::getMatmulNarrowDim(encoding).isN();
}

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return dyn_cast_or_null<EncodingAttr>(type.getEncoding());
}

FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  auto indexingMapsAttr = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
      indexingMapsAttr.getValue(), [](Attribute m) -> AffineMap {
        return cast<AffineMapAttr>(m).getAffineMap();
      });
  return linalg::inferContractionDims(indexingMaps);
}

std::string stringifyOperandIndex(IntegerAttr valueAttr) {
  auto value = valueAttr.getValue().getZExtValue();
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

} // namespace mlir::iree_compiler::IREE::Encoding
