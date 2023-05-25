// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

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
  // Shapes and types are carried independently or folded away entirely - all
  // we need at the HAL level is a blob of bytes.
  addConversion([=](TensorType type) -> std::optional<Type> {
    // HAL only should be concerned with numeric values.
    if (HALTypeConverter::shouldConvertToBufferView(type)) {
      return IREE::HAL::BufferViewType::get(type.getContext());
    }
    return std::nullopt;
  });

  addTargetMaterialization([](OpBuilder &builder, IREE::HAL::BufferType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    if (llvm::isa<TensorType>(inputs[0].getType())) {
      return builder.create<IREE::HAL::TensorExportOp>(loc, type, inputs[0]);
    } else if (llvm::isa<IREE::HAL::BufferViewType>(inputs[0].getType())) {
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
    if (llvm::isa<TensorType>(inputType)) {
      return builder.create<IREE::HAL::TensorExportOp>(loc, type, inputValue);
    } else if (llvm::isa<IREE::HAL::BufferType>(inputType)) {
      // Look for the buffer view this buffer came from, if any.
      // If we don't have the origin buffer view then we can't know the shape
      // and can't materialize one here - it's too late.
      if (auto bvbOp = dyn_cast_or_null<IREE::HAL::BufferViewBufferOp>(
              inputValue.getDefiningOp())) {
        return bvbOp.getBufferView();
      }
      return nullptr;
    } else {
      return nullptr;
    }
  });

  // Recursively handle pointer target types (we want to convert
  // ptr<tensor<...>> to ptr<!hal.buffer<...>>, for example).
  addConversion([this](IREE::Util::PtrType type) -> Type {
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return Type();
    }
    return IREE::Util::PtrType::get(targetType);
  });
}

}  // namespace iree_compiler
}  // namespace mlir
