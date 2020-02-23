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

HALTypeConverter::HALTypeConverter(
    ArrayRef<const HALConversionDialectInterface *> conversionInterfaces)
    : conversionInterfaces(conversionInterfaces.vec()) {
  addConversion([this](Type type, SmallVectorImpl<Type> &results) {
    for (auto *conversionInterface : this->conversionInterfaces) {
      if (succeeded(conversionInterface->convertType(type, results))) {
        return success();
      }
    }
    results.push_back(type);
    return success();
  });
  addConversion([](TensorType type) {
    // TODO(benvanik): composite-type conversion (buffer + dynamic dims).
    return IREE::HAL::BufferType::get(type.getContext());
  });
  addConversion([this](IREE::PtrType type) -> Type {
    // Recursively handle pointer target types (we want to convert ptr<index> to
    // ptr<i32>, for example).
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return Type();
    }
    return IREE::PtrType::get(targetType);
  });
}

}  // namespace iree_compiler
}  // namespace mlir
