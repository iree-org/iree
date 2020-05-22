// Copyright 2020 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_VMLA_CONVERSION_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VMLA_CONVERSION_TYPECONVERTER_H_

#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class VMLATypeConverter : public TypeConverter {
 public:
  VMLATypeConverter();

  // Returns the number of bytes an element of the given type occupies
  // post-conversion. For example, the size of i1 would be '1 byte'.
  static int32_t getRoundedElementByteWidth(Type type) {
    auto bitWidth = 0;
    if (type.isIntOrFloat()) {
      bitWidth = type.getIntOrFloatBitWidth();
    } else if (type.isIndex()) {
      bitWidth = IndexType::kInternalStorageBitWidth;
    } else {
      llvm_unreachable(
          "getRoundedElementByteWidth only support int, float"
          "and index type now.");
    }

    return (bitWidth + 8 - 1) / 8;
  }

  // TODO(benvanik): signature conversion for output buffers.
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_CONVERSION_TYPECONVERTER_H_
