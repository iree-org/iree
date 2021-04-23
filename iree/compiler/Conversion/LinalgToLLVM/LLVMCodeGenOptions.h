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

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVM_LLVMCODEGENOPTIONS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLLVM_LLVMCODEGENOPTIONS_H_

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace iree_compiler {

// Options used to configure LLVM passes.
struct LLVMCodegenOptions {
  bool useConvImg2Col = false;
  // Target specific options.
  bool unfuseFMAOps = false;
  bool useVectorToAarch64 = false;
  bool useLinalgOnTensorsToVectors = false;
};

// Returns LLVM CodeGen options from command-line options.
LLVMCodegenOptions getLLVMCodegenOptionsFromClOptions();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVM_LLVMCODEGENOPTIONS_H_
