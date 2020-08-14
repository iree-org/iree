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

#ifndef IREE_COMPILER_DIALECT_VMLA_CONVERSION_VMLATOVM_CONVERTVMLATOVM_H_
#define IREE_COMPILER_DIALECT_VMLA_CONVERSION_VMLATOVM_CONVERTVMLATOVM_H_

#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns from the VMLA dialect to the VM dialect.
void populateVMLAToVMPatterns(MLIRContext *context,
                              TypeConverter &typeConverter,
                              SymbolTable &importSymbols,
                              OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_CONVERSION_VMLATOVM_CONVERTVMLATOVM_H_
