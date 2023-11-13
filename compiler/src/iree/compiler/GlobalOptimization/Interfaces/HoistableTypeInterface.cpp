// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Interfaces/HoistableTypeInterface.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

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
  return b.create<IREE::Flow::TensorBitCastOp>(loc, targetType, global,
                                               ValueRange(), ValueRange());
}

struct HoistableTensorTypeInterface
    : public IREE::Util::HoistableTypeInterface::ExternalModel<
          HoistableTensorTypeInterface, RankedTensorType> {
  bool isHoistableType(Type type) const {
    auto tensorType = llvm::cast<RankedTensorType>(type);
    unsigned bitWidth =
        IREE::Util::getTypeBitWidth(tensorType.getElementType());
    return llvm::isPowerOf2_32(bitWidth) && bitWidth <= 64;
  }
  bool isHoistableLeafType(Type type) const {
    auto tensorType = llvm::cast<RankedTensorType>(type);
    unsigned bitWidth =
        IREE::Util::getTypeBitWidth(tensorType.getElementType());
    // Never hoist boolean values; IREE still does implicit extension of
    // booleans to a byte width so we avoid packing them.
    return bitWidth != 1;
  }
  Type getPreferredStorageType(Type type) const {
    auto tensorType = llvm::cast<RankedTensorType>(type);
    // Constant data should be statically shaped.
    if (!tensorType.hasStaticShape()) {
      return type;
    }
    unsigned elementBitWidth =
        IREE::Util::getTypeBitWidth(tensorType.getElementType());
    // Bit cast sub-byte and non-byte aligned tensor types as MLIR cannot store
    // them packed at the moment. Also bools continue getting special treatment.
    if (elementBitWidth == 1 ||
        (llvm::isPowerOf2_32(elementBitWidth) && elementBitWidth >= 8)) {
      return type;
    }
    int64_t numElements = ShapedType::getNumElements(tensorType.getShape());
    // Bail out if the data can't be aligned on bytes.
    if (numElements * elementBitWidth % 8 != 0) {
      return type;
    }
    return RankedTensorType::get({numElements * elementBitWidth / 8},
                                 Builder(type.getContext()).getIntegerType(8));
  }
  static Value setPreferredStorageType(OpBuilder &builder, Location loc,
                                       Type storageType, Value init) {
    auto storageTensorType = dyn_cast<RankedTensorType>(storageType);
    if (!storageTensorType) {
      return init;
    }
    return bitcastToStaticTypeImpl(builder, loc, storageTensorType, init);
  }
  static Value unsetPreferredStorageType(OpBuilder &builder, Location loc,
                                         Type originalType,
                                         Value loadedGlobal) {
    auto originalTensorType = dyn_cast<RankedTensorType>(originalType);
    if (!originalTensorType) {
      return loadedGlobal;
    }
    return bitcastToStaticTypeImpl(builder, loc, originalTensorType,
                                   loadedGlobal);
  }
};

//===----------------------------------------------------------------------===//
// IREE specific post analysis transformations.
//===----------------------------------------------------------------------===//

void registerHoistableTypeInterfaces(DialectRegistry &registry) {
  // Register hoistable type interfaces for builtin types.
  registry.addExtension(+[](MLIRContext *ctx) {
    RankedTensorType::attachInterface<HoistableTensorTypeInterface>(*ctx);
  });
}

} // namespace iree_compiler
} // namespace mlir
