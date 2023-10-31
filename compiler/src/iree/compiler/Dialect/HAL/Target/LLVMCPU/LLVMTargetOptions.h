// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_

#include <string_view>

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"

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

// The LLVMTarget contains all of the information to perform code generation
// and linking for an ExecutableVariant. It should not contain any
// environmental configuration like linker paths, diagnostic aids, etc.
struct LLVMTarget {
  static constexpr bool DEFAULT_LINK_EMBEDDED = true;
  static constexpr bool DEFAULT_DEBUG_SYMBOLS = true;
  static constexpr SanitizerKind DEFAULT_SANITIZER_KIND = SanitizerKind::kNone;
  static constexpr bool DEFAULT_LINK_STATIC = false;
  static constexpr bool DEFAULT_LOOP_INTERLEAVING = false;
  static constexpr bool DEFAULT_LOOP_VECTORIZATION = false;
  static constexpr bool DEFAULT_LOOP_UNROLLING = true;
  static constexpr bool DEFAULT_SLP_VECTORIZATION = false;
  static constexpr llvm::FloatABI::ABIType DEFAULT_FLOAT_ABI =
      llvm::FloatABI::ABIType::Hard;

  // Default initialize all fields.
  LLVMTarget();
  // Initialize for specific triple, CPU and link features.
  LLVMTarget(std::string_view triple, std::string_view cpu,
             std::string_view cpuFeatures, bool requestLinkEmbedded);
  static const LLVMTarget &getForHost();
  void print(llvm::raw_ostream &os) const;

  // Stores the target to the given DictionaryAttr in a way that can be
  // later loaded from loadFromConfigAttr().
  void storeToConfigAttrs(MLIRContext *context,
                          SmallVector<NamedAttribute> &config) const;

  // Loads from a DictionaryAttr. On failure returns none and emits.
  static std::optional<LLVMTarget>
  loadFromConfigAttr(Location loc, DictionaryAttr config,
                     const LLVMTarget &defaultTarget);

  // Key fields about the machine can only be set via constructor.
  const std::string &getTriple() const { return triple; }
  const std::string &getCpu() const { return cpu; }
  const std::string &getCpuFeatures() const { return cpuFeatures; }
  bool getLinkEmbedded() const { return linkEmbedded; }

  llvm::PipelineTuningOptions pipelineTuningOptions;
  // Optimization level to be used by the LLVM optimizer (middle-end).
  llvm::OptimizationLevel optimizerOptLevel;
  // Optimization level to be used by the LLVM code generator (back-end).
  llvm::CodeGenOptLevel codeGenOptLevel;
  llvm::TargetOptions llvmTargetOptions;

  // Include debug information in output files (PDB, DWARF, etc).
  // Though this can be set independently from the optLevel (so -O3 with debug
  // information is valid) it may significantly change the output program
  // contents and benchmarking of binary sizes and to some extent execution
  // time should be avoided with symbols present.
  bool debugSymbols = DEFAULT_DEBUG_SYMBOLS;

  // Sanitizer Kind for CPU Kernels
  SanitizerKind sanitizerKind = DEFAULT_SANITIZER_KIND;

  // Build for IREE static library loading using this output path for
  // a "{staticLibraryOutput}.o" object file and "{staticLibraryOutput}.h"
  // header file.
  //
  // This option is incompatible with the linkEmbedded option.
  std::string staticLibraryOutput;

  // Link any required runtime libraries into the produced binaries statically.
  // This increases resulting binary size but enables the binaries to be used on
  // any machine without requiring matching system libraries to be installed.
  bool linkStatic = DEFAULT_LINK_STATIC;

private:
  void addTargetCPUFeaturesForCPU();

  std::string triple;
  std::string cpu;
  std::string cpuFeatures;

  // Build for the IREE embedded platform-agnostic ELF loader.
  // Note: this is ignored for target machines that do not support the ELF
  // loader, such as WebAssembly.
  bool linkEmbedded = DEFAULT_LINK_EMBEDDED;
};

struct LLVMTargetOptions {
  // Default target machine configuration.
  LLVMTarget target;

  // Tool to use for native platform linking (like ld on Unix or link.exe on
  // Windows). Acts as a prefix to the command line and can contain additional
  // arguments.
  std::string systemLinkerPath;

  // Tool to use for linking embedded ELFs. Must be lld.
  std::string embeddedLinkerPath;

  // Tool to use for linking WebAssembly modules. Must be wasm-ld or lld.
  std::string wasmLinkerPath;

  // True to keep linker artifacts for debugging.
  bool keepLinkerArtifacts = false;

  // Returns LLVMTargetOptions that are suitable for running on the host.
  // This does not configure the options from global flags unless if they
  // are target invariant.
  static LLVMTargetOptions getHostOptions();

  // Returns LLVMTargetOptions struct intialized with the iree-llvmcpu-* flags.
  static LLVMTargetOptions getFromFlags();

private:
  void initializeTargetInvariantFlags();
};

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_
