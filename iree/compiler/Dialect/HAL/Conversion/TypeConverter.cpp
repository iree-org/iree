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

#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

LogicalResult HALTypeConverter::convertType(Type type,
                                            SmallVectorImpl<Type> &results) {
  if (auto ptrType = type.dyn_cast<IREE::PtrType>()) {
    // Recursively handle pointer target types (we want to convert ptr<index> to
    // ptr<i32>, for example).
    auto targetType = convertType(ptrType.getTargetType());
    if (!targetType) {
      return failure();
    }
    results.push_back(IREE::PtrType::get(targetType));
    return success();
  } else if (type.isa<TensorType>()) {
    // TODO(benvanik): composite-type conversion (buffer + dynamic dims).
    results.push_back(
        IREE::RefPtrType::get(IREE::HAL::BufferType::get(type.getContext())));
    return success();
  }
  for (auto *conversionInterface : conversionInterfaces) {
    if (succeeded(conversionInterface->convertType(type, results))) {
      return success();
    }
  }
  results.push_back(type);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
