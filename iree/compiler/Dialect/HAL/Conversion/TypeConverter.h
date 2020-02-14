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

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_TYPECONVERTER_H_

#include <vector>

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class HALTypeConverter : public TypeConverter {
 public:
  HALTypeConverter(
      ArrayRef<const HALConversionDialectInterface *> conversionInterfaces)
      : conversionInterfaces(conversionInterfaces.vec()) {}

  LogicalResult convertType(Type type, SmallVectorImpl<Type> &results) override;
  // TODO(benvanik): signature conversion for output buffers.

  // Since we override the more complex convertType above, the default
  // behavior of the simpler convertType becomes a footgun. Override it here so
  // that it respects the complex convertType, and also assert in case it
  // is is used when the more complex one is needed.
  // TODO(b/148971171): It's really confusing that this doesn't automatically
  // inherit behavior from the more complex convertType above.
  Type convertType(Type type) override {
    SmallVector<Type, 4> results;
    if (failed(convertType(type, results))) {
      return nullptr;
    }
    assert(results.size() == 1 &&
           "using simple convertType when a complex case needs to be handled!");
    return results[0];
  }

 private:
  // The set of dialect conversion interfaces we should query to convert types.
  std::vector<const HALConversionDialectInterface *> conversionInterfaces;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_TYPECONVERTER_H_
