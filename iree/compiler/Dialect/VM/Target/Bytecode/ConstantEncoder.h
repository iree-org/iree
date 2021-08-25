// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_CONSTANTENCODER_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_CONSTANTENCODER_H_

#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

struct SerializedConstantRef {
  flatbuffers_uint8_vec_ref_t ref = 0;
  int64_t totalSize = 0;
  uint32_t crc32 = 0;
};

// Serializes a constant attribute to the FlatBuffer as a binary blob.
// Returns the size in bytes of the serialized value and the flatbuffers offset
// to the uint8 vec containing the data. If |calculateCRC32| is provided then a
// CRC32 of the data will be computed and returned as well.
SerializedConstantRef serializeConstant(Location loc, ElementsAttr elementsAttr,
                                        size_t alignment, bool calculateCRC32,
                                        FlatbufferBuilder &fbb);

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_CONSTANTENCODER_H_
