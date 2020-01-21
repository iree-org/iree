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

#ifndef IREE_COMPILER_DIALECT_IREE_CONVERSION_PRESERVECOMPILERHINTS_H_
#define IREE_COMPILER_DIALECT_IREE_CONVERSION_PRESERVECOMPILERHINTS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Adds op legality rules to |conversionTarget| to preserve compiler hints
// that satisfy the type constraints of |typeConverter|.
void setupCompilerHintsLegality(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter);

// Appends all patterns for preserving compiler hints while they are transformed
// by the dialect conversion framework.
void populatePreserveCompilerHintsPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_CONVERSION_PRESERVECOMPILERHINTS_H_
