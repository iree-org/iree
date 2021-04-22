// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_CONVERSION_VECTORTOLLVM_PASSES_H_
#define IREE_COMPILER_CONVERSION_VECTORTOLLVM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class LLVMTypeConverter;

namespace iree_compiler {

/// A pass that converts vector dialect operations to inline assembly
std::unique_ptr<FunctionPass> createVectorToAArch64InlineAssemblyPass();

/// Populates `patterns` to convert vector.contract op to a sequence
/// of AArch64 inline assembly operations.
void populateVectorContractToAArch64InlineAsm(
    OwningRewritePatternList &patterns, MLIRContext *context);

}  // namespace iree_compiler
}  // namespace mlir

#endif
