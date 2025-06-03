// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ElementPackingUtils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

// TODO(lialan): remove cl options once frontend can emit packed i1 tensors.
llvm::cl::opt<bool> clEnableI1Support(
    "iree-experimental-packed-i1-storage",
    llvm::cl::desc(
        "Experimental feature: force to use packed storage for i1 tensors."
        "Turning on this option will see i1 tensors as if it has "
        "#iree_encoding.packed_storage attribute."
        "This is to allow an alternative way to test the packed storage "
        "feature before frontend can emit packed i1 tensors."
        "This option can be dropped once the frontend can emit packed i1 "
        "tensors."),
    llvm::cl::init(false));

namespace mlir::iree_compiler {

static bool needToPackSubByteElementBitWidthImpl(unsigned bitWidth,
                                                 bool isPackedStorage) {
  // Enable i1 support if requested.
  if (isPackedStorage && bitWidth == 1) {
    return true;
  }
  // Require the original bit width to be some power of two for now to avoid
  // trickiness and weirdness of packing and cross-byte access.
  // Also disallow boolean values for now--they may require separate interface
  // choices.
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth) && bitWidth != 1;
}

bool needToPackSubByteElementBitWidth(unsigned bitWidth) {
  return needToPackSubByteElementBitWidthImpl(
      bitWidth, /*isPackedStorage=*/clEnableI1Support);
}

bool needToPackSubByteElements(RankedTensorType shapedType) {
  unsigned bitWidth = IREE::Util::getTypeBitWidth(shapedType.getElementType());
  // Two paths to enable packed storage for i1 tensors: the attribute or cl
  // option. The cl option will be dropped once frontend supports emitting
  // tensors with attributes.
  bool isPackedStorage =
      IREE::Encoding::hasPackedStorageAttr(shapedType) || clEnableI1Support;
  return needToPackSubByteElementBitWidthImpl(bitWidth, isPackedStorage);
}

static Type legalizeStorageElementTypeImpl(Type elementType,
                                           bool isPackedStorage) {
  // Only handle integers; floats in MLIR all have aligned widths (today).
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType)
    return elementType;

  // For sub-byte elements, default to pack them into bytes.
  unsigned bitWidth = intType.getWidth();
  if (needToPackSubByteElementBitWidthImpl(bitWidth, isPackedStorage))
    return elementType;

  // Otherwise, extend them to the next power-of-two bit width.
  unsigned alignedBitWidth =
      IREE::Util::getRoundedElementByteWidth(intType) * 8;
  if (alignedBitWidth == bitWidth)
    return elementType;
  return IntegerType::get(elementType.getContext(), alignedBitWidth,
                          intType.getSignedness());
}

Type legalizeStorageElementType(Type elementType) {
  // Consider packed storage for i1 tensors if cl opt is set.
  return legalizeStorageElementTypeImpl(elementType,
                                        /*isPackedStorage=*/clEnableI1Support);
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

  bool isPackedStorage =
      IREE::Encoding::hasPackedStorageAttr(shapedType) || clEnableI1Support;
  Type alignedElementType = legalizeStorageElementTypeImpl(
      shapedType.getElementType(), isPackedStorage);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  // Calculate all static dims first, if any.
  int64_t staticCount = 1;
  if (!needToPackSubByteElementBitWidthImpl(elementBits, isPackedStorage)) {
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
  if (needToPackSubByteElementBitWidthImpl(elementBits, isPackedStorage)) {
    assert(8 % elementBits == 0);
    unsigned byteElements = 8 / elementBits;
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    auto divisor = builder.create<arith::ConstantIndexOp>(loc, byteElements);
    if (!isPackedStorage && dynamicDims.empty() &&
        (staticCount * elementBits) % 8 != 0) {
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
  bool isPackedStorage =
      IREE::Encoding::hasPackedStorageAttr(originalType) || clEnableI1Support;
  Type alignedElementType = legalizeStorageElementTypeImpl(
      originalType.getElementType(), isPackedStorage);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  // Sub-byte packing requires putting multiple elements in the same byte.
  if (needToPackSubByteElementBitWidthImpl(elementBits, isPackedStorage)) {
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
