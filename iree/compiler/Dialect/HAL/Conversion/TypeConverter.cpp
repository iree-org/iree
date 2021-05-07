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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"

namespace mlir {
namespace iree_compiler {

HALTypeConverter::HALTypeConverter(
    ArrayRef<const HALConversionDialectInterface *> conversionInterfaces)
    : conversionInterfaces(conversionInterfaces.vec()) {
  // Custom conversion interfaces for external dialects.
  addConversion([this](Type type, SmallVectorImpl<Type> &results) {
    for (auto *conversionInterface : this->conversionInterfaces) {
      if (succeeded(conversionInterface->convertType(type, results))) {
        return success();
      }
    }
    results.push_back(type);
    return success();
  });

  // Tensors become buffers views by default.
  // They may be stripped to buffers by canonicalization if they are not
  // required to remain as buffer views.
  addConversion([](TensorType type) -> Optional<Type> {
    // HAL only should be concerned with numeric values.
    if (HALTypeConverter::shouldConvertToBuffer(type)) {
      return IREE::HAL::BufferViewType::get(type.getContext());
    }
    return llvm::None;
  });

  addTargetMaterialization([](OpBuilder &builder, IREE::HAL::BufferType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    if (inputs[0].getType().isa<TensorType>()) {
      return builder.create<IREE::HAL::TensorCastOp>(loc, type, inputs[0]);
    } else if (inputs[0].getType().isa<IREE::HAL::BufferViewType>()) {
      return builder.create<IREE::HAL::BufferViewBufferOp>(loc, type,
                                                           inputs[0]);
    } else {
      emitError(loc) << "unsupported HAL target materialization: "
                     << inputs[0].getType();
      return nullptr;
    }
  });
  addTargetMaterialization([](OpBuilder &builder,
                              IREE::HAL::BufferViewType type, ValueRange inputs,
                              Location loc) -> Value {
    assert(inputs.size() == 1);
    auto inputValue = inputs[0];
    auto inputType = inputValue.getType();
    if (inputType.isa<TensorType>()) {
      return builder.create<IREE::HAL::TensorCastOp>(loc, type, inputValue);
    } else if (inputType.isa<IREE::HAL::BufferType>()) {
      // Look for the buffer view this buffer came from, if any.
      // If we don't have the origin buffer view then we can't know the shape
      // and can't materialize one here - it's too late.
      if (auto bvbOp = dyn_cast_or_null<IREE::HAL::BufferViewBufferOp>(
              inputValue.getDefiningOp())) {
        return bvbOp.buffer_view();
      }
      return nullptr;
    } else {
      return nullptr;
    }
  });

  // Recursively handle pointer target types (we want to convert
  // ptr<tensor<...>> to ptr<!hal.buffer<...>>, for example).
  addConversion([this](IREE::PtrType type) -> Type {
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return Type();
    }
    return IREE::PtrType::get(targetType);
  });
}

}  // namespace iree_compiler
}  // namespace mlir
