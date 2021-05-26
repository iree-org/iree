// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/Bytecode/ConstantEncoder.h"

#include "llvm/Support/CRC.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// TODO(benvanik): switch to LLVM's BinaryStreamWriter to handle endianness.

static void serializeConstantI8Array(DenseIntElementsAttr attr,
                                     size_t alignment, FlatbufferBuilder &fbb) {
  // vm.rodata and other very large constants end up as this; since i8 is i8
  // everywhere (endianness doesn't matter when you have one byte :) we can
  // directly access the data and memcpy.
  int64_t totalSize = attr.getNumElements() * sizeof(int8_t);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
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
                                      size_t alignment,
                                      FlatbufferBuilder &fbb) {
  int64_t totalSize = attr.getNumElements() * sizeof(int16_t);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
}

static void serializeConstantI32Array(DenseIntElementsAttr attr,
                                      size_t alignment,
                                      FlatbufferBuilder &fbb) {
  int64_t totalSize = attr.getNumElements() * sizeof(int32_t);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
  uint32_t *nativePtr = reinterpret_cast<uint32_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(32, 0) & UINT32_MAX;
  }
}

static void serializeConstantI64Array(DenseIntElementsAttr attr,
                                      size_t alignment,
                                      FlatbufferBuilder &fbb) {
  int64_t totalSize = attr.getNumElements() * sizeof(int64_t);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
  uint64_t *nativePtr = reinterpret_cast<uint64_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(64, 0) & UINT64_MAX;
  }
}

static void serializeConstantF16Array(DenseFPElementsAttr attr,
                                      size_t alignment,
                                      FlatbufferBuilder &fbb) {
  int64_t totalSize = attr.getNumElements() * sizeof(uint16_t);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) =
        value.bitcastToAPInt().extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
}

static void serializeConstantF32Array(DenseFPElementsAttr attr,
                                      size_t alignment,
                                      FlatbufferBuilder &fbb) {
  int64_t totalSize = attr.getNumElements() * sizeof(float);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
  float *nativePtr = reinterpret_cast<float *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToFloat();
  }
}

static void serializeConstantF64Array(DenseFPElementsAttr attr,
                                      size_t alignment,
                                      FlatbufferBuilder &fbb) {
  int64_t totalSize = attr.getNumElements() * sizeof(double);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, totalSize);
  double *nativePtr = reinterpret_cast<double *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToDouble();
  }
}

SerializedConstantRef serializeConstant(Location loc, ElementsAttr elementsAttr,
                                        size_t alignment, bool calculateCRC32,
                                        FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  if (auto attr = elementsAttr.dyn_cast<DenseIntElementsAttr>()) {
    switch (attr.getType().getElementTypeBitWidth()) {
      case 8:
        serializeConstantI8Array(attr, alignment, fbb);
        break;
      case 16:
        serializeConstantI16Array(attr, alignment, fbb);
        break;
      case 32:
        serializeConstantI32Array(attr, alignment, fbb);
        break;
      case 64:
        serializeConstantI64Array(attr, alignment, fbb);
        break;
      default:
        emitError(loc) << "unhandled element bitwidth "
                       << attr.getType().getElementTypeBitWidth();
        return {};
    }
  } else if (auto attr = elementsAttr.dyn_cast<DenseFPElementsAttr>()) {
    switch (attr.getType().getElementTypeBitWidth()) {
      case 16:
        serializeConstantF16Array(attr, alignment, fbb);
        break;
      case 32:
        serializeConstantF32Array(attr, alignment, fbb);
        break;
      case 64:
        serializeConstantF64Array(attr, alignment, fbb);
        break;
      default:
        emitError(loc) << "unhandled element bitwidth "
                       << attr.getType().getElementTypeBitWidth();
        return {};
    }
  } else {
    emitError(loc) << "unimplemented attribute encoding: "
                   << elementsAttr.getType();
    return {};
  }

  uint8_t *dataPtr =
      reinterpret_cast<uint8_t *>(flatcc_builder_vector_edit(fbb));
  size_t totalSize = flatcc_builder_vector_count(fbb);
  uint32_t crc32Value = 0;
  if (calculateCRC32) {
    crc32Value = llvm::crc32(0u, ArrayRef<uint8_t>(dataPtr, totalSize));
  }
  return SerializedConstantRef{
      flatbuffers_uint8_vec_end(fbb),
      static_cast<int64_t>(totalSize),
      crc32Value,
  };
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
