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

#include "flatbuffers/flatbuffers.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

namespace {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

}  // namespace

// TODO(benvanik): switch to LLVM's BinaryStreamWriter to handle endianness.

static Offset<Vector<uint8_t>> serializeConstantI8Array(
    DenseIntElementsAttr attr, FlatBufferBuilder &fbb) {
  uint8_t *bytePtr = nullptr;
  auto byteVector =
      fbb.CreateUninitializedVector(attr.getNumElements() * 1, &bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(bytePtr++) = value.extractBitsAsZExtValue(8, 0) & UINT8_MAX;
  }
  return byteVector;
}

static Offset<Vector<uint8_t>> serializeConstantI16Array(
    DenseIntElementsAttr attr, FlatBufferBuilder &fbb) {
  uint8_t *bytePtr = nullptr;
  auto byteVector =
      fbb.CreateUninitializedVector(attr.getNumElements() * 2, &bytePtr);
  uint16_t *nativePtr = reinterpret_cast<uint16_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(16, 0) & UINT16_MAX;
  }
  return byteVector;
}

static Offset<Vector<uint8_t>> serializeConstantI32Array(
    DenseIntElementsAttr attr, FlatBufferBuilder &fbb) {
  uint8_t *bytePtr = nullptr;
  auto byteVector =
      fbb.CreateUninitializedVector(attr.getNumElements() * 4, &bytePtr);
  uint32_t *nativePtr = reinterpret_cast<uint32_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(32, 0) & UINT32_MAX;
  }
  return byteVector;
}

static Offset<Vector<uint8_t>> serializeConstantI64Array(
    DenseIntElementsAttr attr, FlatBufferBuilder &fbb) {
  uint8_t *bytePtr = nullptr;
  auto byteVector =
      fbb.CreateUninitializedVector(attr.getNumElements() * 8, &bytePtr);
  uint64_t *nativePtr = reinterpret_cast<uint64_t *>(bytePtr);
  for (const APInt &value : attr.getIntValues()) {
    *(nativePtr++) = value.extractBitsAsZExtValue(64, 0) & UINT64_MAX;
  }
  return byteVector;
}

static Offset<Vector<uint8_t>> serializeConstantF32Array(
    DenseFPElementsAttr attr, FlatBufferBuilder &fbb) {
  uint8_t *bytePtr = nullptr;
  auto byteVector =
      fbb.CreateUninitializedVector(attr.getNumElements() * 4, &bytePtr);
  float *nativePtr = reinterpret_cast<float *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToFloat();
  }
  return byteVector;
}

static Offset<Vector<uint8_t>> serializeConstantF64Array(
    DenseFPElementsAttr attr, FlatBufferBuilder &fbb) {
  uint8_t *bytePtr = nullptr;
  auto byteVector =
      fbb.CreateUninitializedVector(attr.getNumElements() * 8, &bytePtr);
  double *nativePtr = reinterpret_cast<double *>(bytePtr);
  for (const APFloat &value : attr.getFloatValues()) {
    *(nativePtr++) = value.convertToDouble();
  }
  return byteVector;
}

Offset<Vector<uint8_t>> serializeConstant(Location loc,
                                          ElementsAttr elementsAttr,
                                          FlatBufferBuilder &fbb) {
  if (auto attr = elementsAttr.dyn_cast<DenseIntElementsAttr>()) {
    switch (attr.getType().getElementTypeBitWidth()) {
      case 8:
        return serializeConstantI8Array(attr, fbb);
      case 16:
        return serializeConstantI16Array(attr, fbb);
      case 32:
        return serializeConstantI32Array(attr, fbb);
      case 64:
        return serializeConstantI64Array(attr, fbb);
      default:
        emitError(loc) << "unhandled element bitwidth "
                       << attr.getType().getElementTypeBitWidth();
        return {};
    }
  } else if (auto attr = elementsAttr.dyn_cast<DenseFPElementsAttr>()) {
    switch (attr.getType().getElementTypeBitWidth()) {
      case 32:
        return serializeConstantF32Array(attr, fbb);
      case 64:
        return serializeConstantF64Array(attr, fbb);
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
