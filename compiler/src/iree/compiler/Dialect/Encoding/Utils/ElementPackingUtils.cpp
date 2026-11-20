// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/Utils/ElementPackingUtils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler {

static bool needsDefaultSubBytePacking(unsigned bitWidth) {
  // Unencoded booleans use byte containers; i2 and i4 are tightly packed.
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth) && bitWidth != 1;
}

bool needToPackSubByteElements(RankedTensorType shapedType) {
  Type storageType = legalizeStorageElementType(shapedType);
  unsigned bitWidth = IREE::Util::getTypeBitWidth(storageType);
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth);
}

Type legalizeStorageElementType(RankedTensorType tensorType) {
  Type elementType = tensorType.getElementType();
  if (auto encoding = IREE::Encoding::getSerializableAttr(tensorType)) {
    if (Type storageType = encoding.getStorageElementType(elementType)) {
      return storageType;
    }
  }
  // Floats retain their logical type unless an encoding specifies a container.
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType) {
    return elementType;
  }

  // For sub-byte elements, default to pack them into bytes.
  unsigned bitWidth = intType.getWidth();
  if (needsDefaultSubBytePacking(bitWidth)) {
    return elementType;
  }

  // Otherwise, extend them to the next power-of-two bit width.
  unsigned alignedBitWidth =
      IREE::Util::getRoundedElementByteWidth(intType) * 8;
  if (alignedBitWidth == bitWidth) {
    return elementType;
  }
  return IntegerType::get(elementType.getContext(), alignedBitWidth,
                          intType.getSignedness());
}

Value convertStorageElementType(Location loc, Value source, Type targetType,
                                OpBuilder &builder) {
  Type sourceType = source.getType();
  if (sourceType == targetType) {
    return source;
  }
  Type sourceElementType = getElementTypeOrSelf(sourceType);
  Type targetElementType = getElementTypeOrSelf(targetType);
  if (!(sourceElementType.isSignlessInteger() &&
        (targetElementType.isSignlessInteger() ||
         isa<FloatType>(targetElementType))) &&
      !(isa<FloatType>(sourceElementType) &&
        targetElementType.isSignlessInteger())) {
    return {};
  }

  // Keep shape and encoding when expressing a float's bits as integers.
  auto getBitsType = [&builder](Type type) -> Type {
    Type bitsType = builder.getIntegerType(
        getElementTypeOrSelf(type).getIntOrFloatBitWidth());
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      return tensorType.clone(bitsType);
    }
    return bitsType;
  };
  if (isa<FloatType>(sourceElementType)) {
    source =
        arith::BitcastOp::create(builder, loc, getBitsType(sourceType), source);
  }
  Type targetBitsType = getBitsType(targetType);
  int64_t sourceWidthInBits = sourceElementType.getIntOrFloatBitWidth();
  int64_t targetWidthInBits = targetElementType.getIntOrFloatBitWidth();
  if (sourceWidthInBits > targetWidthInBits) {
    source = arith::TruncIOp::create(builder, loc, targetBitsType, source);
  } else if (sourceWidthInBits < targetWidthInBits) {
    source = arith::ExtUIOp::create(builder, loc, targetBitsType, source);
  }
  if (isa<FloatType>(targetElementType)) {
    source = arith::BitcastOp::create(builder, loc, targetType, source);
  }
  return source;
}

Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder) {
  if (auto serializableEncodingAttr =
          IREE::Encoding::getSerializableAttr(shapedType)) {
    return serializableEncodingAttr.calculateStorageSizeInBytes(
        loc, builder, shapedType, dynamicDims);
  }

  Type alignedElementType = legalizeStorageElementType(shapedType);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);
  // Packing is already handled by the SerializableAttr above.
  bool needsPacking = needsDefaultSubBytePacking(elementBits);

  // Calculate all static dims first, if any.
  int64_t staticCount = 1;
  if (!needsPacking) {
    staticCount *= IREE::Util::getRoundedElementByteWidth(alignedElementType);
  }

  for (unsigned i = 0; i < shapedType.getRank(); ++i) {
    if (!shapedType.isDynamicDim(i)) {
      staticCount *= shapedType.getDimSize(i);
    }
  }
  // Scale by dynamic dims, if present.
  auto value =
      arith::ConstantIndexOp::create(builder, loc, staticCount).getResult();
  for (auto dim : dynamicDims) {
    value = builder.createOrFold<arith::MulIOp>(loc, value, dim);
  }
  // Sub-byte packing requires putting multiple elements in the same byte.
  if (needsPacking) {
    assert(8 % elementBits == 0);
    unsigned byteElements = 8 / elementBits;
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    auto divisor = arith::ConstantIndexOp::create(builder, loc, byteElements);
    if (dynamicDims.empty() && (staticCount * elementBits) % 8 != 0) {
      return nullptr;
    }
    value = builder.createOrFold<arith::CeilDivUIOp>(loc, value, divisor);
  }

  return value;
}

AffineMap getStorageElementOffsetMap(RankedTensorType type) {
  if (auto encoding = IREE::Encoding::getSerializableAttr(type)) {
    return encoding.getStorageElementOffsetMap(type);
  }
  if (type.getEncoding()) {
    return {};
  }
  return IREE::Encoding::getRowMajorStorageElementOffsetMap(
      type, IREE::Util::getTypeBitWidth(legalizeStorageElementType(type)));
}

Value calculateStorageElementOffsetInBytes(Location loc, RankedTensorType type,
                                           ValueRange dynamicDims,
                                           ValueRange indices,
                                           OpBuilder &builder) {
  if (auto encoding = IREE::Encoding::getSerializableAttr(type)) {
    return encoding.calculateStorageElementOffsetInBytes(loc, builder, type,
                                                         dynamicDims, indices);
  }
  AffineMap map = getStorageElementOffsetMap(type);
  if (!map) {
    return {};
  }
  return IREE::Encoding::buildStorageElementByteOffset(
      loc, builder, map.getResult(0), indices, dynamicDims);
}

} // namespace mlir::iree_compiler
