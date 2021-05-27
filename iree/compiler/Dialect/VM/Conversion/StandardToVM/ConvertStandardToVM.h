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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_STANDARDTOVM_CONVERTSTANDARDTOVM_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_STANDARDTOVM_CONVERTSTANDARDTOVM_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Appends standard dialect to vm dialect patterns to the given pattern list.
void populateStandardToVMPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_STANDARDTOVM_CONVERTSTANDARDTOVM_H_
