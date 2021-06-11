// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/Bytecode/ConstantEncoder.h"

#include "iree/compiler/Dialect/VM/Target/ConstantEncodingUtils.h"
#include "llvm/Support/CRC.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

SerializedConstantRef serializeConstant(Location loc, ElementsAttr elementsAttr,
                                        size_t alignment, bool calculateCRC32,
                                        FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));

  int32_t bitwidth = elementsAttr.getType().getElementTypeBitWidth();
  int64_t size = elementsAttr.getNumElements() * (bitwidth / 8);
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, size);

  if (failed(serializeConstantArray(loc, elementsAttr, alignment, bytePtr))) {
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
