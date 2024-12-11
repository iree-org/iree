// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <numeric>

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Encoding::EncodingAttr;
using IREE::Encoding::getEncodingAttr;
using IREE::Encoding::getEncodingContractionDims;

// If tensorType has the encoding of a matmul RESULT with narrow N, returns
// the transposed type. Otherwise, just returns tensorType.
static RankedTensorType transposeIfNarrowNResult(RankedTensorType tensorType) {
  auto encoding =
      llvm::dyn_cast_or_null<EncodingAttr>(tensorType.getEncoding());
  if (!encoding) {
    return tensorType;
  }
  if (!isNarrowNResult(encoding)) {
    return tensorType;
  }
  SmallVector<int64_t> newOriginalShape(tensorType.getShape());
  auto userIndexingMaps = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> maps;
  for (auto a : userIndexingMaps) {
    maps.push_back(cast<AffineMapAttr>(a).getAffineMap());
  }
  auto cDims = linalg::inferContractionDims(maps);
  SmallVector<int64_t> newShape(tensorType.getShape());
  SmallVector<int64_t> permIndices(maps[0].getNumDims());
  std::iota(std::begin(permIndices), std::end(permIndices), 0);
  // Matrix case: there are both M and N dimensions. Transposing means swapping
  // them.
  if (cDims->m.size() == 1 && cDims->n.size() == 1) {
    int m = cDims->m[0];
    int n = cDims->n[0];
    std::swap(permIndices[m], permIndices[n]);
    std::optional<unsigned> mDim = encoding.mapDimToOperandIndex(m);
    std::optional<unsigned> nDim = encoding.mapDimToOperandIndex(n);
    if (mDim.has_value() && nDim.has_value()) {
      std::swap(newShape[mDim.value()], newShape[nDim.value()]);
      std::swap(newOriginalShape[mDim.value()], newOriginalShape[nDim.value()]);
    }
  }
  // Vector case: there is no N dimension to swap the M dimension with. We
  // swap the maps themselves.
  if (cDims->n.empty()) {
    std::swap(maps[0], maps[1]);
  }

  SmallVector<int64_t> newRoundDimsTo(encoding.getRoundDimsToArray());
  assert(newRoundDimsTo.size() == 0 || newRoundDimsTo.size() >= 3);
  if (newRoundDimsTo.size() != 0) {
    std::swap(newRoundDimsTo[newRoundDimsTo.size() - 3],
              newRoundDimsTo[newRoundDimsTo.size() - 2]);
  }
  auto context = tensorType.getContext();
  AffineMap permutation = AffineMap::getPermutationMap(permIndices, context);
  for (auto &map : maps) {
    map = map.compose(permutation);
  }
  auto elemType = tensorType.getElementType();
  auto operandIndex = encoding.getOperandIndex().getInt();

  // TODO(#17718): Handle the broadcast map for transpose cases. It is on the
  // experimental path, so it is not clear what needs to be done here. For now
  // just use the original map for the new encoding.
  std::optional<AffineMap> newBcastMap;
  if (encoding.getBcastMap()) {
    newBcastMap = encoding.getBcastMap().getValue();
  }
  auto newEncoding = IREE::Encoding::EncodingAttr::get(
      context, operandIndex, encoding.getOpType().getValue(),
      encoding.getElementTypesArray(), maps, newBcastMap, newRoundDimsTo);
  return RankedTensorType::get(newShape, elemType, newEncoding);
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    bool transposeNarrowN, IREE::Codegen::LayoutAttrInterface layoutAttr)
    : transposeNarrowN(transposeNarrowN), layoutAttr(layoutAttr) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType type) -> RankedTensorType {
    // For a given tensor type with an encoding, return the materialized
    // type to use for it. If no encoding is set, then return the tensor type
    // itself.
    RankedTensorType tensorType =
        transposeNarrowN ? transposeIfNarrowNResult(type) : type;
    MaterializeEncodingInfo encodingInfo = getEncodingInfo(tensorType);
    if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
      return dropEncoding(type);
    }
    auto packedType = cast<RankedTensorType>(tensor::PackOp::inferPackedType(
        tensorType, encodingInfo.innerTileSizes, encodingInfo.innerDimsPos,
        encodingInfo.outerDimsPerm));

    // There is no swizzle, we are already done. Typically the case on CPU.
    if (!encodingInfo.swizzle) {
      return packedType;
    }

    // There is a swizzle, we need to handle it. Typically the case on GPU.
    auto swizzle = *encodingInfo.swizzle;
    SmallVector<int64_t> newShape(
        packedType.getShape().drop_back(encodingInfo.innerTileSizes.size()));
    SmallVector<int64_t> swizzledTileShape =
        IREE::Codegen::getExpandedTileShape(swizzle.expandShape);
    applyPermutationToVector(swizzledTileShape, swizzle.permutation);
    newShape.append(swizzledTileShape);
    return RankedTensorType::get(newShape, packedType.getElementType());
  });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasEncoding = [](Type t) -> bool {
      auto tensorType = dyn_cast<RankedTensorType>(t);
      return tensorType && tensorType.getEncoding();
    };
    auto valueHasEncoding = [=](Value v) -> bool {
      return typeHasEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

RankedTensorType dropEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

} // namespace mlir::iree_compiler
