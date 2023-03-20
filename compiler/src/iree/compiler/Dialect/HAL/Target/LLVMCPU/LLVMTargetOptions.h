// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_

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
  kThread,
};

struct LLVMTarget {
  std::string triple;
  std::string cpu;
  std::string cpuFeatures;
};

struct LLVMTargetOptions {
  // Default target machine configuration.
  LLVMTarget target;

  llvm::PipelineTuningOptions pipelineTuningOptions;
  // Optimization level to be used by the LLVM optimizer (middle-end).
  llvm::OptimizationLevel optimizerOptLevel;
  // Optimization level to be used by the LLVM code generator (back-end).
  llvm::CodeGenOpt::Level codeGenOptLevel;
  llvm::TargetOptions options;

  // Include debug information in output files (PDB, DWARF, etc).
  // Though this can be set independently from the optLevel (so -O3 with debug
  // information is valid) it may significantly change the output program
  // contents and benchmarking of binary sizes and to some extent execution
  // time should be avoided with symbols present.
  bool debugSymbols = true;

  // Sanitizer Kind for CPU Kernels
  SanitizerKind sanitizerKind = SanitizerKind::kNone;

  // Tool to use for native platform linking (like ld on Unix or link.exe on
  // Windows). Acts as a prefix to the command line and can contain additional
  // arguments.
  std::string systemLinkerPath;

  // Tool to use for linking embedded ELFs. Must be lld.
  std::string embeddedLinkerPath;

  // Tool to use for linking WebAssembly modules. Must be wasm-ld or lld.
  std::string wasmLinkerPath;

  // Build for the IREE embedded platform-agnostic ELF loader.
  // Note: this is ignored for target machines that do not support the ELF
  // loader, such as WebAssembly.
  bool linkEmbedded = true;

  // Link any required runtime libraries into the produced binaries statically.
  // This increases resulting binary size but enables the binaries to be used on
  // any machine without requiring matching system libraries to be installed.
  bool linkStatic = false;

  // True to keep linker artifacts for debugging.
  bool keepLinkerArtifacts = false;

  // Build for IREE static library loading using this output path for
  // a "{staticLibraryOutput}.o" object file and "{staticLibraryOutput}.h"
  // header file.
  //
  // This option is incompatible with the linkEmbedded option.
  std::string staticLibraryOutput;
};

// Returns LLVMTargetOptions struct intialized with the iree-llvmcpu-* flags.
LLVMTargetOptions getLLVMTargetOptionsFromFlags();

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_
