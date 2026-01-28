// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Interfaces/HoistableTypeInterface.h"

#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir::iree_compiler {

static Value bitcastToStaticTypeImpl(OpBuilder &b, Location loc,
                                     RankedTensorType targetType,
                                     Value global) {
  if (global.getType() == targetType) {
    return global;
  }
  if (!isa<RankedTensorType>(global.getType())) {
    return global;
  }
  // No dynamic dims because we are always bitcasting constants.
  return IREE::TensorExt::BitCastOp::create(b, loc, targetType, global,
                                            ValueRange(), ValueRange());
}

static inline unsigned int
getDataTypeStorageBitWidth(Type type, const DataLayout &dataLayout) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isIndex()) {
    return *dataLayout.getTypeIndexBitwidth(elementType);
  }
  return IREE::Util::getTypeBitWidth(elementType);
}

struct HoistableTensorTypeInterface
    : public IREE::Util::HoistableTypeInterface::ExternalModel<
          HoistableTensorTypeInterface, RankedTensorType> {
  bool isHoistableType(Type type, const DataLayout &dataLayout) const {
    unsigned bitWidth = getDataTypeStorageBitWidth(type, dataLayout);
    return llvm::isPowerOf2_32(bitWidth) && bitWidth <= 64;
  }
  bool isHoistableLeafType(Type type, const DataLayout &dataLayout) const {
    unsigned bitWidth = getDataTypeStorageBitWidth(type, dataLayout);
    // Never hoist boolean values; IREE still does implicit extension of
    // booleans to a byte width so we avoid packing them.
    return bitWidth != 1;
  }
  Type getPreferredStorageType(Type type, const DataLayout &dataLayout) const {
    auto tensorType = cast<RankedTensorType>(type);
    // Constant data should be statically shaped.
    if (!tensorType.hasStaticShape()) {
      return type;
    }

    unsigned elementBitWidth =
        getDataTypeStorageBitWidth(tensorType, dataLayout);
    // Bools get special treatment - don't pack them.
    if (elementBitWidth == 1) {
      return type;
    }
    // Byte-aligned types (8, 16, 32, 64 bits) don't need conversion.
    if (llvm::isPowerOf2_32(elementBitWidth) && elementBitWidth >= 8) {
      return type;
    }

    // Sub-byte types need to be packed into bytes for storage - use i8 storage.
    Type i8Type = Builder(type.getContext()).getIntegerType(8);

    // If the tensor has an encoding, let it handle the conversion.
    if (auto serializableAttr =
            dyn_cast_if_present<IREE::Encoding::SerializableAttr>(
                tensorType.getEncoding())) {
      FailureOr<IREE::Encoding::BitcastEncodingInfo> result =
          serializableAttr.convertForBitcast(
              tensorType.getShape(), tensorType.getElementType(), i8Type);
      if (succeeded(result)) {
        return RankedTensorType::get(result->newShape, i8Type,
                                     result->encoding);
      }
      // Encoding couldn't handle conversion, so return the original type.
      return type;
    }

    // In case of no encoding, convert to flat i8 storage.
    int64_t numElements = ShapedType::getNumElements(tensorType.getShape());
    int64_t totalBits = numElements * elementBitWidth;
    // Bail out if the data can't be aligned on bytes.
    if (totalBits % 8 != 0) {
      return type;
    }
    return RankedTensorType::get({totalBits / 8}, i8Type);
  }

  static Value encodeStorageType(OpBuilder &builder, Location loc,
                                 Type storageType, Value init) {
    auto storageTensorType = dyn_cast<RankedTensorType>(storageType);
    if (!storageTensorType) {
      return init;
    }
    return bitcastToStaticTypeImpl(builder, loc, storageTensorType, init);
  }
  static Value decodeStorageType(OpBuilder &builder, Location loc,
                                 Type originalType, Value loadedGlobal) {
    auto originalTensorType = dyn_cast<RankedTensorType>(originalType);
    if (!originalTensorType) {
      return loadedGlobal;
    }
    return bitcastToStaticTypeImpl(builder, loc, originalTensorType,
                                   loadedGlobal);
  }
};

struct HoistableIndexTypeInterface
    : public IREE::Util::HoistableTypeInterface::ExternalModel<
          HoistableIndexTypeInterface, IndexType> {
  bool isHoistableType(Type type, const DataLayout &dataLayout) const {
    return true;
  }
  bool isHoistableLeafType(Type type, const DataLayout &dataLayout) const {
    return true;
  }

  Type getPreferredStorageType(Type type, const DataLayout &dataLayout) const {
    return IntegerType::get(type.getContext(),
                            getDataTypeStorageBitWidth(type, dataLayout));
  }
  static Value encodeStorageType(OpBuilder &builder, Location loc,
                                 Type storageType, Value init) {
    auto storageIndexType = dyn_cast<IntegerType>(storageType);
    if (!storageIndexType || init.getType() == storageIndexType ||
        !isa<IndexType>(init.getType())) {
      return init;
    }
    return arith::IndexCastOp::create(builder, loc, storageType, init);
  }
  static Value decodeStorageType(OpBuilder &builder, Location loc,
                                 Type originalType, Value loadedGlobal) {
    auto originalIndexType = dyn_cast<IndexType>(originalType);
    if (!originalIndexType || loadedGlobal.getType() == originalIndexType ||
        !isa<IntegerType>(loadedGlobal.getType())) {
      return loadedGlobal;
    }
    return arith::IndexCastOp::create(builder, loc, originalType, loadedGlobal);
  }
};

//===----------------------------------------------------------------------===//
// IREE specific post analysis transformations.
//===----------------------------------------------------------------------===//

void registerHoistableTypeInterfaces(DialectRegistry &registry) {
  // Register hoistable type interfaces for builtin types.
  registry.addExtension(+[](MLIRContext *ctx) {
    RankedTensorType::attachInterface<HoistableTensorTypeInterface>(*ctx);
    IndexType::attachInterface<HoistableIndexTypeInterface>(*ctx);
  });
}

} // namespace mlir::iree_compiler
