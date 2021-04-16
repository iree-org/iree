// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/VM/Target/Bytecode/ConstantEncoder.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// TODO(benvanik): switch to LLVM's BinaryStreamWriter to handle endianness.

static flatbuffers_uint8_vec_ref_t serializeConstantI8Array(
    DenseIntElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  // vm.rodata and other very large constants end up as this; since i8 is i8
  // everywhere (endianness doesn't matter when you have one byte :) we can
  // directly access the data and memcpy.
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr =
      flatbuffers_uint8_vec_extend(fbb, attr.getNumElements() * sizeof(int8_t));
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
  return flatbuffers_uint8_vec_end(fbb);
}

static flatbuffers_uint8_vec_ref_t serializeConstantI16Array(
    DenseIntElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(
      fbb, attr.getNumElements() * sizeof(int16_t));
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
  return flatbuffers_uint8_vec_end(fbb);
}

static flatbuffers_uint8_vec_ref_t serializeConstantI32Array(
    DenseIntElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(
      fbb, attr.getNumElements() * sizeof(int32_t));
  uint32_t *nativePtr = reinterpret_cast<uint32_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(32, 0) & UINT32_MAX;
  }
  return flatbuffers_uint8_vec_end(fbb);
}

static flatbuffers_uint8_vec_ref_t serializeConstantI64Array(
    DenseIntElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(
      fbb, attr.getNumElements() * sizeof(int64_t));
  uint64_t *nativePtr = reinterpret_cast<uint64_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(64, 0) & UINT64_MAX;
  }
  return flatbuffers_uint8_vec_end(fbb);
}

static flatbuffers_uint8_vec_ref_t serializeConstantF32Array(
    DenseFPElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr =
      flatbuffers_uint8_vec_extend(fbb, attr.getNumElements() * sizeof(float));
  float *nativePtr = reinterpret_cast<float *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToFloat();
  }
  return flatbuffers_uint8_vec_end(fbb);
}

static flatbuffers_uint8_vec_ref_t serializeConstantF64Array(
    DenseFPElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr =
      flatbuffers_uint8_vec_extend(fbb, attr.getNumElements() * sizeof(double));
  double *nativePtr = reinterpret_cast<double *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToDouble();
  }
  return flatbuffers_uint8_vec_end(fbb);
}

static flatbuffers_uint8_vec_ref_t serializeConstantF16Array(
    DenseFPElementsAttr attr, size_t alignment, FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(
      fbb, attr.getNumElements() * sizeof(uint16_t));
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) =
        value.bitcastToAPInt().extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
  return flatbuffers_uint8_vec_end(fbb);
}

flatbuffers_uint8_vec_ref_t serializeConstant(Location loc,
                                              ElementsAttr elementsAttr,
                                              size_t alignment,
                                              FlatbufferBuilder &fbb) {
  if (auto attr = elementsAttr.dyn_cast<DenseIntElementsAttr>()) {
    switch (attr.getType().getElementTypeBitWidth()) {
      case 8:
        return serializeConstantI8Array(attr, alignment, fbb);
      case 16:
        return serializeConstantI16Array(attr, alignment, fbb);
      case 32:
        return serializeConstantI32Array(attr, alignment, fbb);
      case 64:
        return serializeConstantI64Array(attr, alignment, fbb);
      default:
        emitError(loc) << "unhandled element bitwidth "
                       << attr.getType().getElementTypeBitWidth();
        return {};
    }
  } else if (auto attr = elementsAttr.dyn_cast<DenseFPElementsAttr>()) {
    switch (attr.getType().getElementTypeBitWidth()) {
      case 16:
        return serializeConstantF16Array(attr, alignment, fbb);
      case 32:
        return serializeConstantF32Array(attr, alignment, fbb);
      case 64:
        return serializeConstantF64Array(attr, alignment, fbb);
      default:
        emitError(loc) << "unhandled element bitwidth "
                       << attr.getType().getElementTypeBitWidth();
        return {};
    }
  }
  emitError(loc) << "unimplemented attribute encoding: "
                 << elementsAttr.getType();
  return {};
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
