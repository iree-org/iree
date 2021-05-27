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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMTARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMTARGETOPTIONS_H_

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetOptions.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Defines kinds of Sanitizer
// The order in enum class should be same as one in flat buffer schema
enum class SanitizerKind {
  kNone = 0,
  kAddress,
};

struct LLVMTargetOptions {
  // Target machine configuration.
  std::string targetTriple;
  std::string targetCPU;
  std::string targetCPUFeatures;

  llvm::PipelineTuningOptions pipelineTuningOptions;
  llvm::PassBuilder::OptimizationLevel optLevel;
  llvm::TargetOptions options;

  // Include debug information in output files (PDB, DWARF, etc).
  // Though this can be set independently from the optLevel (so -O3 with debug
  // information is valid) it may significantly change the output program
  // and benchmarking
  bool debugSymbols = true;

  // Sanitizer Kind for CPU Kernels
  SanitizerKind sanitizerKind = SanitizerKind::kNone;

  // Build for the IREE embedded platform-agnostic ELF loader.
  bool linkEmbedded = false;

  // Link any required runtime libraries into the produced binaries statically.
  // This increases resulting binary size but enables the binaries to be used on
  // any machine without requiring matching system libraries to be installed.
  bool linkStatic = false;

  // True to keep linker artifacts for debugging.
  bool keepLinkerArtifacts = false;
};

// Returns LLVMTargetOptions struct intialized with the iree-llvm-* flags.
LLVMTargetOptions getLLVMTargetOptionsFromFlags();

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMTARGETOPTIONS_H_
