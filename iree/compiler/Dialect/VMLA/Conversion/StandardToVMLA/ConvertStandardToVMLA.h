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

#ifndef IREE_COMPILER_DIALECT_VMLA_CONVERSION_STANDARDTOVMLA_CONVERTSTANDARDTOVMLA_H_  // NOLINT
#define IREE_COMPILER_DIALECT_VMLA_CONVERSION_STANDARDTOVMLA_CONVERTSTANDARDTOVMLA_H_  // NOLINT

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns from the std dialect to the VMLA dialect.
void populateStandardToVMLAPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_CONVERSION_STANDARDTOVMLA_CONVERTSTANDARDTOVMLA_H_
        // // NOLINT
