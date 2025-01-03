// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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

EncodingAttr EncodingAttr::cloneWithLayouts(ArrayRef<Attribute> layouts) {
  MLIRContext *ctx = getContext();
  return get(ctx, getOperandIndex(), getOpType(), getElementTypes(),
             /*user_indexing_maps=*/ArrayAttr(),
             /*bcast_map=*/AffineMapAttr(),
             /*round_dims_to=*/DenseI64ArrayAttr(),
             ArrayAttr::get(ctx, layouts));
}

/// Returns the bit-width of the scalar type. If the type is complex, it returns
/// the type of individual elements * 2 (1 for real and 1 for complex).
static unsigned getTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type)) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  return type.getIntOrFloatBitWidth();
}

/// Returns the number of bytes an element of the given type occupies in memory.
/// This is in the default dense conversion to machine words where sizes must be
/// powers of two aligned to bytes.
///
/// Examples:
///   getRoundedElementByteWidth(i1) = 1
///   getRoundedElementByteWidth(i23) = 4
///   getRoundedElementByteWidth(i32) = 4
///   getRoundedElementByteWidth(bf16) = 2
///   getRoundedElementByteWidth(i33) = 8
///   getRoundedElementByteWidth(complex<f32>) = 8
static int32_t getRoundedElementByteWidth(Type type) {
  unsigned bitsUnaligned = getTypeBitWidth(type);
  assert(bitsUnaligned > 0 && "0-width types unsupported");
  // Round up to 8-bit aligned bytes.
  unsigned byteAligned = (bitsUnaligned + 8 - 1) / 8;
  // Round up to the next power of two (unless already a power of two).
  return llvm::PowerOf2Ceil(byteAligned);
}

Value EncodingAttr::calculateStorageSizeInBytes(Location loc,
                                                OpBuilder &builder,
                                                RankedTensorType type,
                                                ValueRange dynamicDims) const {
  SmallVector<int64_t> paddedShape(type.getShape());
  SmallVector<Value> paddedDynamicDims(dynamicDims.begin(), dynamicDims.end());
  ArrayRef<int64_t> roundDimsTo = getRoundDimsToArray();
  FailureOr<linalg::ContractionDimensions> cDims =
      getEncodingContractionDims(*this);
  auto pad = [&](int dim, int value) {
    std::optional<unsigned> maybeMappedDim = mapDimToOperandIndex(dim);
    if (!maybeMappedDim) {
      return;
    }
    unsigned mappedDim = maybeMappedDim.value();
    if (type.isDynamicDim(mappedDim)) {
      mappedDim = type.getDynamicDimIndex(mappedDim);
      auto alignment = builder.create<arith::ConstantIndexOp>(loc, value);
      paddedDynamicDims[mappedDim] = builder.create<arith::CeilDivUIOp>(
          loc, paddedDynamicDims[mappedDim], alignment);
      paddedDynamicDims[mappedDim] = builder.create<arith::MulIOp>(
          loc, paddedDynamicDims[mappedDim], alignment);
    } else {
      paddedShape[mappedDim] = llvm::alignTo(paddedShape[mappedDim], value);
    }
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
  unsigned elementBits = getTypeBitWidth(type.getElementType());
  int64_t numBytesPerElem = 1;
  if (elementBits > kNumBitsInByte) {
    numBytesPerElem *= getRoundedElementByteWidth(type.getElementType());
  }

  int64_t staticCount = numBytesPerElem;
  for (unsigned i = 0, e = type.getRank(); i < e; ++i) {
    if (!type.isDynamicDim(i)) {
      staticCount *= paddedShape[i];
    }
  }

  Value result =
      builder.create<arith::ConstantIndexOp>(loc, staticCount).getResult();
  for (auto dim : paddedDynamicDims) {
    result = builder.create<arith::MulIOp>(loc, result, dim);
  }

  // Always pack the elements back-to-back for subtypes.
  if (elementBits < kNumBitsInByte) {
    if (kNumBitsInByte % elementBits) {
      assert(false && "unsupported subtype");
      return Value();
    }
    Value divisor = builder.create<arith::ConstantIndexOp>(
        loc, kNumBitsInByte / elementBits);
    result = builder.create<arith::CeilDivUIOp>(loc, result, divisor);
  }

  return result;
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
