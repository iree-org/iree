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

#ifndef IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_CONVERT_FLOW_TO_HAL_H_
#define IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_CONVERT_FLOW_TO_HAL_H_

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/types.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Types.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace tf_strings {

// Populates conversion patterns from the tensor-based custom dialect ops to the
// HAL buffer-based ones.
void populateTFStringsToHALPatterns(MLIRContext *ctx,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter);

// Exposes conversion patterns fthat transition tensors to buffers during teh
// Flow->HAL dialect lowering. This is only required if the dialect has ops that
// use tensor types.
class TfStringsToHALConversionInterface : public HALConversionDialectInterface {
 public:
  using HALConversionDialectInterface::HALConversionDialectInterface;
  void setupConversionTarget(ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) const override {
    target.addLegalDialect<IREE::Strings::StringsDialect>();
    populateTFStringsToHALPatterns(getDialect()->getContext(), patterns,
                                   typeConverter);
  };

  LogicalResult convertType(Type type,
                            SmallVectorImpl<Type> &results) const override {
    if (type.isa<tf_strings::StringType>()) {
      results.push_back(IREE::Strings::StringType::get(type.getContext()));
      return success();
    }

    if (auto tensor = type.dyn_cast<TensorType>()) {
      if (tensor.getElementType().isa<tf_strings::StringType>()) {
        results.push_back(
            IREE::Strings::StringTensorType::get(type.getContext()));
        return success();
      }
    }

    return failure();
  }
};

}  // namespace tf_strings
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TFSTRINGS_TRANSFORMS_TFSTRINGSTOSTRINGS_H_
