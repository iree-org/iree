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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_TFTENSORLIST_CONVERSION_CONVERTFLOWTOHAL_H_
#define IREE_INTEGRATIONS_TENSORFLOW_TFTENSORLIST_CONVERSION_CONVERTFLOWTOHAL_H_

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListDialect.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns from the tensor-based custom dialect ops to the
// HAL buffer-based ones.
void populateTensorListToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter);

// Exposes conversion patterns that transition tensors to buffers during the
// Flow->HAL dialect lowering. This is only required if the dialect has ops that
// use tensor types.
class TfTensorListToHALConversionInterface
    : public HALConversionDialectInterface {
 public:
  using HALConversionDialectInterface::HALConversionDialectInterface;

  void setupConversionTarget(ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) const override {
    target.addLegalDialect<IREE::TensorList::TensorListDialect>();
    populateTensorListToHALPatterns(getDialect()->getContext(), patterns,
                                    typeConverter);
  }

  LogicalResult convertType(Type type,
                            SmallVectorImpl<Type> &results) const override {
    if (type.isa<tf_tensorlist::TensorListType>()) {
      results.push_back(
          IREE::TensorList::TensorListType::get(type.getContext()));
      return success();
    }
    return failure();
  }
};

}  // namespace iree_compiler
}  // namespace mlir

#endif
