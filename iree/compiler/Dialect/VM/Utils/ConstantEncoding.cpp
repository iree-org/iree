// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Utils/ConstantEncoding.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// TODO(benvanik): switch to LLVM's BinaryStreamWriter to handle endianness.

static void serializeConstantI8Array(DenseIntElementsAttr attr,
                                     size_t alignment, uint8_t *bytePtr) {
  // vm.rodata and other very large constants end up as this; since i8 is i8
  // everywhere (endianness doesn't matter when you have one byte :) we can
  // directly access the data and memcpy.
  if (attr.isSplat()) {
    // NOTE: this is a slow path and we should have eliminated it earlier on
    // during constant op conversion.
    for (const APInt &value : attr.getIntValues()) {
      *(bytePtr++) = value.extractBitsAsZExtValue(8, 0) & UINT8_MAX;
    }
  } else {
    auto rawData = attr.getRawData();
    std::memcpy(bytePtr, rawData.data(), rawData.size());
  }
}

static void serializeConstantI16Array(DenseIntElementsAttr attr,
                                      size_t alignment, uint8_t *bytePtr) {
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
}

static void serializeConstantI32Array(DenseIntElementsAttr attr,
                                      size_t alignment, uint8_t *bytePtr) {
  uint32_t *nativePtr = reinterpret_cast<uint32_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(32, 0) & UINT32_MAX;
  }
}

static void serializeConstantI64Array(DenseIntElementsAttr attr,
                                      size_t alignment, uint8_t *bytePtr) {
  uint64_t *nativePtr = reinterpret_cast<uint64_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(64, 0) & UINT64_MAX;
  }
}

static void serializeConstantF16Array(DenseFPElementsAttr attr,
                                      size_t alignment, uint8_t *bytePtr) {
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) =
        value.bitcastToAPInt().extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
}

static void serializeConstantF32Array(DenseFPElementsAttr attr,
                                      size_t alignment, uint8_t *bytePtr) {
  float *nativePtr = reinterpret_cast<float *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToFloat();
  }
}

static void serializeConstantF64Array(DenseFPElementsAttr attr,
                                      size_t alignment, uint8_t *bytePtr) {
  double *nativePtr = reinterpret_cast<double *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToDouble();
  }
}

LogicalResult serializeConstantArray(Location loc, ElementsAttr elementsAttr,
                                     size_t alignment, uint8_t *dst) {
  auto bitwidth = elementsAttr.getType().getElementTypeBitWidth();

  if (auto attr = elementsAttr.dyn_cast<DenseIntElementsAttr>()) {
    switch (bitwidth) {
      case 8:
        serializeConstantI8Array(attr, alignment, dst);
        break;
      case 16:
        serializeConstantI16Array(attr, alignment, dst);
        break;
      case 32:
        serializeConstantI32Array(attr, alignment, dst);
        break;
      case 64:
        serializeConstantI64Array(attr, alignment, dst);
        break;
      default:
        return emitError(loc) << "unhandled element bitwidth " << bitwidth;
    }
  } else if (auto attr = elementsAttr.dyn_cast<DenseFPElementsAttr>()) {
    switch (bitwidth) {
      case 16:
        serializeConstantF16Array(attr, alignment, dst);
        break;
      case 32:
        serializeConstantF32Array(attr, alignment, dst);
        break;
      case 64:
        serializeConstantF64Array(attr, alignment, dst);
        break;
      default:
        return emitError(loc) << "unhandled element bitwidth " << bitwidth;
    }
  } else {
    return emitError(loc) << "unimplemented attribute encoding: "
                          << elementsAttr.getType();
  }

  return success();
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
