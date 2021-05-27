// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

  // Tensors become buffers by default.
  // TODO(benvanik): make them buffer views instead? then they carry shape but
  // are memory type erased which is not good.
  addConversion([](TensorType type) -> Optional<Type> {
    // HAL only should be concerned with numeric values.
    if (HALTypeConverter::shouldConvertToBuffer(type)) {
      // TODO(benvanik): composite-type conversion (buffer + dynamic dims).
      return IREE::HAL::BufferType::get(type.getContext());
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
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<IREE::HAL::TensorCastOp>(loc, type, inputs[0]);
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
