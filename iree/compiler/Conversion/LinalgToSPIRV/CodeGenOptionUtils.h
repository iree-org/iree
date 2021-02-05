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
//
//===- ClOptionUtils.h - Utilities for controlling SPIR-V CodeGen ---------===//
//
// This file contains helper functions to read command-line options controlling
// the SPIR-V code generation pipeline. This allows us to put all command-line
// options in one place. Otherwise, we may need to duplicate the same option
// three times: in the pass itself, in the pass pipeline, and in the callers
// constructing the pass pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_CODEGENOPTIONUTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_CODEGENOPTIONUTILS_H_

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace iree_compiler {

// Options that can be used to configure SPIR-V code generation.
struct SPIRVCodegenOptions {
  llvm::SmallVector<unsigned, 3> workgroupSize = {};
  llvm::SmallVector<unsigned, 3> tileSizes = {};
  bool enableVectorization = false;
  bool useWorkgroupMemory = false;
  bool vectorizeMemref = false;
  bool useLinalgOnTensors = false;
};

// Returns SPIR-V CodeGen options from command-line options.
SPIRVCodegenOptions getSPIRVCodegenOptionsFromClOptions();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_CODEGENOPTIONUTILS_H_
