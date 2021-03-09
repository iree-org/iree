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

#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

static llvm::cl::opt<COutputFormat> outputFormatFlag{
    "iree-vm-c-module-output-format",
    llvm::cl::desc("Output format used to write the C module"),
    llvm::cl::init(COutputFormat::kCode),
    llvm::cl::values(
        clEnumValN(COutputFormat::kCode, "code", "C Code file"),
        clEnumValN(COutputFormat::kMlirText, "mlir-text",
                   "MLIR module file in the VM and EmitC dialects")),
};

static llvm::cl::opt<bool> optimizeFlag{
    "iree-vm-c-module-optimize",
    llvm::cl::desc(
        "Optimizes the VM module with CSE/inlining/etc prior to serialization"),
    llvm::cl::init(true),
};

static llvm::cl::opt<bool> stripDebugOpsFlag{
    "iree-vm-c-module-strip-debug-ops",
    llvm::cl::desc("Strips debug-only ops from the module"),
    llvm::cl::init(false),
};

CTargetOptions getCTargetOptionsFromFlags() {
  CTargetOptions targetOptions;
  targetOptions.outputFormat = outputFormatFlag;
  targetOptions.optimize = optimizeFlag;
  targetOptions.stripDebugOps = stripDebugOpsFlag;
  return targetOptions;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
