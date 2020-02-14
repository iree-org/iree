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

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONDIALECTINTERFACE_H_

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// An interface for dialects to expose HAL conversion functionality.
// The HAL conversion pass will query used dialects via this interface to find
// conversion patterns that map from a higher-level dialect containing ops that
// work on tensors to lower-level ops that work with HAL buffers and raw shapes.
//
// Implementations may choose to have different dialects for these levels of IR
// or for simplicity (and reduction of boilerplate) define the ops within the
// same dialect.
class HALConversionDialectInterface
    : public DialectInterface::Base<HALConversionDialectInterface> {
 public:
  HALConversionDialectInterface(Dialect *dialect) : Base(dialect) {}

  // Populates |patterns| with rewrites that convert from a higher-level
  // tensor-based dialect to ops that interoperate with HAL types.
  // |target| must have newly legal and illegal ops/dialects specified to ensure
  // the conversion takes place.
  virtual void setupConversionTarget(ConversionTarget &target,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter) const = 0;

  // Converts a type.
  // Will be called from the corresponding TypeConverter hook.
  virtual LogicalResult convertType(Type t,
                                    SmallVectorImpl<Type> &results) const {
    return failure();
  }
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
