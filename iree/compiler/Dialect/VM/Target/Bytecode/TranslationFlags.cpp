// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

static llvm::cl::opt<std::string> sourceListingFlag{
    "iree-vm-bytecode-source-listing",
    llvm::cl::desc("Dump a VM MLIR file and annotate source locations with it"),
    llvm::cl::init(""),
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

static llvm::cl::opt<bool> emitPolyglotZipFlag{
    "iree-vm-emit-polyglot-zip",
    llvm::cl::desc(
        "Enables output files to be viewed as zip files for debugging"),
    llvm::cl::init(true),
};

BytecodeTargetOptions getBytecodeTargetOptionsFromFlags() {
  BytecodeTargetOptions targetOptions;
  targetOptions.outputFormat = outputFormatFlag;
  targetOptions.optimize = optimizeFlag;
  targetOptions.sourceListing = sourceListingFlag;
  targetOptions.stripSourceMap = stripSourceMapFlag;
  targetOptions.stripDebugOps = stripDebugOpsFlag;
  targetOptions.emitPolyglotZip = emitPolyglotZipFlag;
  if (outputFormatFlag != BytecodeOutputFormat::kFlatBufferBinary) {
    // Only allow binary output formats to also be .zip files.
    targetOptions.emitPolyglotZip = false;
  }
  return targetOptions;
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
