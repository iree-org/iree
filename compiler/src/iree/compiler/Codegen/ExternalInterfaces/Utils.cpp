// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE {
using IREE::Codegen::MaterializeEncodingInfo;

Value calculatePackedStorageSizeInBytesImpl(Attribute attr, Location loc,
                                            OpBuilder &builder,
                                            RankedTensorType type,
                                            ValueRange dynamicDims) {
  auto deviceLayoutAttr =
      cast<IREE::Codegen::PackedLayoutMaterializerAttr>(attr);
  MaterializeEncodingInfo encodingInfo = deviceLayoutAttr.getEncodingInfo(type);
  SmallVector<int64_t> paddedShape(type.getShape());
  SmallVector<Value> paddedDynamicDims(dynamicDims.begin(), dynamicDims.end());
  for (auto [dim, size] : llvm::zip_equal(encodingInfo.innerDimsPos,
                                          encodingInfo.innerTileSizes)) {
    // Only VMVX backend has dynamic inner tile sizes when ukernel is enabled.
    // It assumes that the padding size is 16. Ideally, the logic should be
    // moved to VMVX implementation details. However, we cook the logic here to
    // reduce code duplication.
    if (ShapedType::isDynamic(size)) {
      assert(isa<IREE::CPU::VMVXEncodingLayoutAttr>(attr) &&
             "only VMVX backend attribute can handle dynamic tile sizes");
      size = 16;
    }

    // Do not create additional operations in the first place if the padding is
    // not needed.
    if (size == 1) {
      continue;
    }

    if (type.isDynamicDim(dim)) {
      dim = type.getDynamicDimIndex(dim);
      auto alignment = builder.create<arith::ConstantIndexOp>(loc, size);
      paddedDynamicDims[dim] = builder.create<arith::CeilDivSIOp>(
          loc, paddedDynamicDims[dim], alignment);
      paddedDynamicDims[dim] =
          builder.create<arith::MulIOp>(loc, paddedDynamicDims[dim], alignment);
    } else {
      paddedShape[dim] = llvm::alignTo(paddedShape[dim], size);
    }
  }

  constexpr int64_t kNumBitsInByte = 8;
  int64_t numBytesPerElem = 1;
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits > kNumBitsInByte) {
    numBytesPerElem *= elementBits / kNumBitsInByte;
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

DictionaryAttr getPackedLayoutImpl(Attribute attr, RankedTensorType type,
                                   bool addEncodingAttr) {
  MLIRContext *ctx = attr.getContext();
  auto deviceLayoutAttr =
      cast<IREE::Codegen::PackedLayoutMaterializerAttr>(attr);
  const MaterializeEncodingInfo info = deviceLayoutAttr.getEncodingInfo(type);
  Attribute encodingInfoAttr =
      IREE::Codegen::serializeEncodingInfo(attr.getContext(), info);
  SmallVector<NamedAttribute> items;
  items.push_back(NamedAttribute(kEncodingInfoAttrName, encodingInfoAttr));
  auto encodingAttr = IREE::Encoding::getEncodingAttr(type);
  if (addEncodingAttr && encodingAttr) {
    items.push_back(NamedAttribute("encoding_attr", encodingAttr));
  }
  return DictionaryAttr::get(ctx, items);
}

void storeNamedAttrIfPresent(SmallVectorImpl<NamedAttribute> &config,
                             DictionaryAttr dictAttr, StringRef name) {
  auto attr = dictAttr.getNamed(name);
  if (!attr) {
    return;
  }
  config.push_back(attr.value());
}

} // namespace mlir::iree_compiler::IREE
