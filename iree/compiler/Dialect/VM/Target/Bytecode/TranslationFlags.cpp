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

#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// TODO(b/145140345): fix LLVM category registration with ASAN.
// static llvm::cl::OptionCategory vmBytecodeOptionsCategory(
//     "IREE VM bytecode options");

static llvm::cl::opt<BytecodeOutputFormat> outputFormatFlag{
    "iree-vm-bytecode-module-output-format",
    llvm::cl::desc("Output format the bytecode module is written in"),
    llvm::cl::init(BytecodeOutputFormat::kFlatBufferBinary),
    llvm::cl::values(
        clEnumValN(BytecodeOutputFormat::kFlatBufferBinary, "flatbuffer-binary",
                   "Binary FlatBuffer file"),
        clEnumValN(BytecodeOutputFormat::kFlatBufferText, "flatbuffer-text",
                   "Text FlatBuffer file, debug-only"),
        clEnumValN(BytecodeOutputFormat::kMlirText, "mlir-text",
                   "MLIR module file in the VM dialect"),
        clEnumValN(BytecodeOutputFormat::kAnnotatedMlirText,
                   "annotated-mlir-text",
                   "MLIR module file in the VM dialect with annotations")),
};

static llvm::cl::opt<bool> optimizeFlag{
    "iree-vm-bytecode-module-optimize",
    llvm::cl::desc(
        "Optimizes the VM module with CSE/inlining/etc prior to serialization"),
    llvm::cl::init(true),
};

static llvm::cl::opt<bool> stripSymbolsFlag{
    "iree-vm-bytecode-module-strip-symbols",
    llvm::cl::desc("Strips all internal symbol names from the module"),
    llvm::cl::init(false),
};

static llvm::cl::opt<bool> stripSourceMapFlag{
    "iree-vm-bytecode-module-strip-source-map",
    llvm::cl::desc("Strips the source map from the module"),
    llvm::cl::init(false),
};

static llvm::cl::opt<bool> stripDebugOpsFlag{
    "iree-vm-bytecode-module-strip-debug-ops",
    llvm::cl::desc("Strips debug-only ops from the module"),
    llvm::cl::init(false),
};

BytecodeTargetOptions getBytecodeTargetOptionsFromFlags() {
  BytecodeTargetOptions targetOptions;
  targetOptions.outputFormat = outputFormatFlag;
  targetOptions.optimize = optimizeFlag;
  targetOptions.stripSymbols = stripSymbolsFlag;
  targetOptions.stripSourceMap = stripSourceMapFlag;
  targetOptions.stripDebugOps = stripDebugOpsFlag;
  return targetOptions;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
