// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ElementPackingUtils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {

bool needToPackSubByteElements(Type type) {
  unsigned bitWidth = isa<TensorType>(type)
                          ? IREE::Util::getTypeBitWidth(
                                dyn_cast<TensorType>(type).getElementType())
                          : IREE::Util::getTypeBitWidth(type);

  // i1 with packed memory layout does not need to be extended.
  if (bitWidth == 1 && IREE::Encoding::hasPackedStorageAttr(type)) {
    return true;
  }

  // Require the original bit width to be some power of two for now to avoid
  // trickiness and weirdness of packing and cross-byte access.
  // Also disallow boolean values for now--they may require separate interface
  // choices.
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth) && bitWidth != 1;
}

Type legalizeStorageElementType(Type type) {
  auto tensorType = llvm::cast<TensorType>(type);
  auto elementType = tensorType.getElementType();

  // Only handle integers; floats in MLIR all have aligned widths (today).
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType)
    return elementType;

  // For sub-byte elements, default to pack them into bytes.
  if (needToPackSubByteElements(type))
    return elementType;

  unsigned bitWidth = intType.getWidth();
  // Otherwise, extend them to the next power-of-two bit width.
  unsigned alignedBitWidth =
      IREE::Util::getRoundedElementByteWidth(intType) * 8;
  if (alignedBitWidth == bitWidth)
    return elementType;
  return IntegerType::get(elementType.getContext(), alignedBitWidth,
                          intType.getSignedness());
}

Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder) {
  Attribute encoding = shapedType.getEncoding();
  if (auto encodingLayoutAttr =
          dyn_cast_or_null<IREE::Encoding::EncodingLayoutAttrInterface>(
              encoding)) {
    return encodingLayoutAttr.calculateStorageSizeInBytes(
        loc, builder, shapedType, dynamicDims);
  }

  Type alignedElementType = legalizeStorageElementType(shapedType);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  int64_t staticCount = 1;
  if (!needToPackSubByteElements(alignedElementType)) {
    staticCount *= IREE::Util::getRoundedElementByteWidth(alignedElementType);
  }

  for (unsigned i = 0; i < shapedType.getRank(); ++i) {
    if (!shapedType.isDynamicDim(i))
      staticCount *= shapedType.getDimSize(i);
  }
  // Scale by dynamic dims, if present.
  auto value =
      builder.create<arith::ConstantIndexOp>(loc, staticCount).getResult();
  for (auto dim : dynamicDims) {
    value = builder.createOrFold<arith::MulIOp>(loc, value, dim);
  }
  // Sub-byte packing requires putting multiple elements in the same byte.
  if (needToPackSubByteElements(alignedElementType)) {
    assert(8 % elementBits == 0);
    unsigned byteElements = 8 / elementBits;
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    auto divisor = builder.create<arith::ConstantIndexOp>(loc, byteElements);
    if (dynamicDims.empty() && (staticCount * elementBits) % 8 != 0) {
      return nullptr;
    }
    value = builder.createOrFold<arith::CeilDivUIOp>(loc, value, divisor);
  }

  return value;
}

Value calculateStorageElementOffsetInBytes(Location loc,
                                           RankedTensorType originalType,
                                           Value linearizedIndex,
                                           OpBuilder &builder) {
  Type alignedElementType = legalizeStorageElementType(originalType);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  // Sub-byte packing requires putting multiple elements in the same byte.
  if (needToPackSubByteElements(originalType)) {
    Value byteElements =
        builder.create<arith::ConstantIndexOp>(loc, 8 / elementBits);
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    return builder.createOrFold<arith::DivUIOp>(loc, linearizedIndex,
                                                byteElements);
  }

  Value elementBytes = builder.create<arith::ConstantIndexOp>(
      loc, IREE::Util::getRoundedElementByteWidth(alignedElementType));
  return builder.createOrFold<arith::MulIOp>(loc, linearizedIndex,
                                             elementBytes);
}

} // namespace mlir::iree_compiler
