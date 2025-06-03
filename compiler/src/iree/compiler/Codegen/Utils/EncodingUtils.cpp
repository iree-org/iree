// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/EncodingUtils.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

namespace mlir::iree_compiler {

static std::optional<IREE::Codegen::MaterializeEncodingInfo>
getEncodingInfoFromType(RankedTensorType type) {
  auto layoutAttr =
      dyn_cast_or_null<IREE::Encoding::LayoutAttr>(type.getEncoding());
  if (!layoutAttr) {
    return std::nullopt;
  }
  ArrayRef<Attribute> layouts = layoutAttr.getLayouts().getValue();
  assert(layouts.size() == 1 && "only single layout is supported");
  if (auto layout =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layouts[0])) {
    return layout.getEncodingInfo(type);
  }
  return std::nullopt;
}

IREE::Codegen::MaterializeEncodingInfo
getEncodingInfoFromLayout(RankedTensorType type,
                          IREE::Encoding::LayoutMaterializerAttr layoutAttr) {
  // If the layout is present in the encoding, use it directly. It means that
  // the layout is already resolved and some information could be dropped during
  // the lowering. Thus, we prioritize the resolved layout.
  if (auto maybeEncodingInfo = getEncodingInfoFromType(type)) {
    return maybeEncodingInfo.value();
  }
  if (auto packedLayoutAttr =
          dyn_cast<IREE::Codegen::PackedLayoutMaterializerAttr>(layoutAttr)) {
    return packedLayoutAttr.getEncodingInfo(type);
  }
  return IREE::Codegen::MaterializeEncodingInfo{};
}

FailureOr<SmallVector<OpFoldResult>> getInnerTileSizesOfrImpl(
    OpBuilder &rewriter, Location loc, RankedTensorType tensorType,
    IREE::Encoding::LayoutMaterializerAttr layoutAttr,
    const IREE::Codegen::MaterializeEncodingInfo &materializeEncodingInfo) {
  ArrayRef<int64_t> staticTileSizes = materializeEncodingInfo.innerTileSizes;
  if (!ShapedType::isDynamicShape(staticTileSizes)) {
    return getAsOpFoldResult(rewriter.getI64ArrayAttr(staticTileSizes));
  }

  // Only VMVX with ukernel config supports dynamic inner tile sizes.
  auto vmvxLayoutAttr = dyn_cast<IREE::CPU::VMVXEncodingLayoutAttr>(layoutAttr);
  if (!vmvxLayoutAttr || !hasUkernel(vmvxLayoutAttr.getConfiguration())) {
    return failure();
  }
  SmallVector<Type> resultTypes(tensorType.getRank(), rewriter.getIndexType());
  auto op = rewriter.create<IREE::Codegen::QueryTileSizesOp>(
      loc, resultTypes, TypeAttr::get(tensorType));
  SmallVector<Value> innerTileSizeValues = op.getResults();

  SmallVector<OpFoldResult> result(staticTileSizes.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (ShapedType::isDynamic(staticTileSizes[i])) {
      result[i] = innerTileSizeValues[i];
    } else if (tensorType.isDynamicDim(i)) {
      result[i] =
          rewriter.create<arith::ConstantIndexOp>(loc, staticTileSizes[i])
              .getResult();
    } else {
      result[i] = rewriter.getI64IntegerAttr(staticTileSizes[i]);
    }
  }
  return result;
}

FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensorImpl(
    OpBuilder &builder, Location loc,
    IREE::TensorExt::DispatchTensorType dispatchTensorType,
    ValueRange dynamicDims, IREE::Encoding::LayoutMaterializerAttr layoutAttr,
    IREE::Codegen::MaterializeEncodingInfo encodingInfo) {
  auto boundTensorType =
      llvm::dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
  if (!boundTensorType) {
    return failure();
  }

  if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
    return failure();
  }

  SmallVector<OpFoldResult> targetShape =
      getMixedValues(boundTensorType.getShape(), dynamicDims, builder);
  auto innerTileSizes = getInnerTileSizesOfrImpl(builder, loc, boundTensorType,
                                                 layoutAttr, encodingInfo);
  if (failed(innerTileSizes)) {
    return failure();
  }
  SmallVector<OpFoldResult> convertedTargetShape =
      linalg::PackOp::getResultShape(builder, loc, targetShape, *innerTileSizes,
                                     encodingInfo.innerDimsPos,
                                     encodingInfo.outerDimsPerm);
  return getSwizzledShape(convertedTargetShape, encodingInfo);
}

SmallVector<OpFoldResult>
getSwizzledShape(ArrayRef<OpFoldResult> packedShape,
                 IREE::Codegen::MaterializeEncodingInfo encodingInfo) {
  if (packedShape.empty() || !encodingInfo.swizzle) {
    return SmallVector<OpFoldResult>(packedShape);
  }

  int64_t srcRank = packedShape.size() - encodingInfo.innerTileSizes.size();
  SmallVector<int64_t> perm = llvm::to_vector(llvm::seq<int64_t>(0, srcRank));
  for (auto i : encodingInfo.swizzle->permutation) {
    perm.push_back(i + srcRank);
  }

  SmallVector<OpFoldResult> newShape(packedShape.take_front(srcRank));
  SmallVector<int64_t> expandedTileShape =
      IREE::Codegen::getExpandedTileShape(encodingInfo.swizzle->expandShape);
  MLIRContext *ctx = packedShape[0].getContext();
  Builder b(ctx);
  for (int64_t d : expandedTileShape) {
    newShape.push_back(b.getIndexAttr(d));
  }
  applyPermutationToVector(newShape, perm);

  return newShape;
}

} // namespace mlir::iree_compiler
