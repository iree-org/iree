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

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion/convert_flow_to_hal.h"

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"

namespace mlir {
namespace iree_compiler {

void populateTensorListToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter) {
  // We can use the HAL conversion handler for this tensor->buffer conversion
  // as we just want the simple form. If we wanted to perform additional
  // verification or have a specific use case (such as a place where only the
  // buffer is required and the shape is not) we could add our own.
  patterns.insert<
      HALOpConversion<tf_tensorlist::Reserve, IREE::TensorList::Reserve>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<tf_tensorlist::GetItem, IREE::TensorList::GetItem>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<tf_tensorlist::SetItem, IREE::TensorList::SetItem>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<tf_tensorlist::FromTensor, IREE::TensorList::FromTensor>>(
      context, typeConverter);
  patterns
      .insert<HALOpConversion<tf_tensorlist::Stack, IREE::TensorList::Stack>>(
          context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
