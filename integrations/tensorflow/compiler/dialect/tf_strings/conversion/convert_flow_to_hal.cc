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

#include "integrations/tensorflow/compiler/dialect/tf_strings/conversion/convert_flow_to_hal.h"

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/ops.h"
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/types.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"

namespace mlir {
namespace iree_compiler {
namespace TFStrings {

void populateTFStringsToHALPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter) {
  // We can use the HAL conversion handler for this tensor->buffer conversion
  // as we just want the simple form. If we wanted to perform additional
  // verification or have a specific use case (such as a place where only the
  // buffer is required and the shape is not) we could add our own.
  patterns.insert<HALOpConversion<TFStrings::PrintOp, IREE::Strings::PrintOp>>(
      context, typeConverter);
  patterns.insert<HALOpConversion<TFStrings::ToStringTensorOp,
                                  IREE::Strings::ToStringTensorOp>>(
      context, typeConverter);
  patterns.insert<HALOpConversion<TFStrings::StringTensorToStringOp,
                                  IREE::Strings::StringTensorToStringOp>>(
      context, typeConverter);
  // TODO(suderman): This should be able to handle generic IREE values.
  patterns.insert<
      HALOpConversion<TFStrings::ToStringOp, IREE::Strings::I32ToStringOp>>(
      context, typeConverter);
}

}  // namespace TFStrings
}  // namespace iree_compiler
}  // namespace mlir
