// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_TYPECONVERTER_H_

#include <vector>

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class HALTypeConverter : public TypeConverter {
 public:
  HALTypeConverter(
      ArrayRef<const HALConversionDialectInterface *> conversionInterfaces,
      bool supportTensors);

  // TODO(benvanik): signature conversion for output buffers.

  static bool shouldConvertToBuffer(Type type) {
    if (TensorType tensor_type = type.template dyn_cast<TensorType>()) {
      return tensor_type.getElementType().isIntOrFloat();
    }
    return false;
  }

 private:
  // The set of dialect conversion interfaces we should query to convert types.
  std::vector<const HALConversionDialectInterface *> conversionInterfaces;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_TYPECONVERTER_H_
