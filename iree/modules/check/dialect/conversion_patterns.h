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

#ifndef IREE_MODULES_CHECK_DIALECT_CONVERSION_PATTERNS_H_
#define IREE_MODULES_CHECK_DIALECT_CONVERSION_PATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

// Populates conversion patterns from the Check dialect to the VM dialect.
void populateCheckToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter);

// Populates conversion patterns from the Check dialect to the HAL dialect.
// Mostly lowers tensors to buffer views.
void populateCheckToHALPatterns(MLIRContext *context,
                                OwningRewritePatternList &patterns,
                                TypeConverter &typeConverter);

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_MODULES_CHECK_DIALECT_CONVERSION_PATTERNS_H_
