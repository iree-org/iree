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

namespace mlir::iree_compiler::IREE::HAL {

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
  static constexpr const char *DEFAULT_DATA_LAYOUT = "";
  static constexpr int64_t DEFAULT_VECTOR_WIDTH_IN_BYTES = 0;
  static constexpr bool DEFAULT_LINK_EMBEDDED = true;
  static constexpr bool DEFAULT_DEBUG_SYMBOLS = true;
  static constexpr SanitizerKind DEFAULT_SANITIZER_KIND = SanitizerKind::kNone;
  static constexpr bool DEFAULT_LINK_STATIC = false;
  static constexpr bool DEFAULT_LOOP_INTERLEAVING = false;
  static constexpr bool DEFAULT_LOOP_VECTORIZATION = false;
  static constexpr bool DEFAULT_LOOP_UNROLLING = false;
  static constexpr bool DEFAULT_SLP_VECTORIZATION = false;
  static constexpr llvm::FloatABI::ABIType DEFAULT_FLOAT_ABI =
      llvm::FloatABI::ABIType::Hard;
  static constexpr const char *DEFAULT_ENABLE_UKERNELS = "default";
  static constexpr bool DEFAULT_LINK_UKERNEL_BITCODE = true;

  // Default initialize all fields.
  LLVMTarget();

  void copy(const LLVMTarget &other) {
    triple = other.triple;
    cpu = other.cpu;
    cpuFeatures = other.cpuFeatures;
    dataLayout = other.dataLayout;
    vectorWidthInBytes = other.vectorWidthInBytes;
    linkEmbedded = other.linkEmbedded;
    ukernels = other.ukernels;
    linkUkernelBitcode = other.linkUkernelBitcode;
  }

  void print(llvm::raw_ostream &os) const;

  // Stores the target to the given DictionaryAttr in a way that can be
  // later loaded from loadFromConfigAttr().
  void storeToConfigAttrs(MLIRContext *context,
                          SmallVector<NamedAttribute> &config) const;

  static std::optional<LLVMTarget> create(std::string_view triple,
                                          std::string_view cpu,
                                          std::string_view cpuFeatures,
                                          bool requestLinkEmbedded);

  static std::optional<LLVMTarget> createForHost();

  // Loads from a DictionaryAttr. On failure returns none and emits.
  static std::optional<LLVMTarget>
  loadFromConfigAttr(Location loc, DictionaryAttr config,
                     const LLVMTarget &defaultTarget);

  // Key fields about the machine can only be set via constructor.
  const std::string &getTriple() const { return triple; }
  const std::string &getCpu() const { return cpu; }
  const std::string &getCpuFeatures() const { return cpuFeatures; }

  // Overrides the data layout of the target.
  std::string dataLayout = DEFAULT_DATA_LAYOUT;
  // Overrides the vector width (in bytes) of the target.
  int64_t vectorWidthInBytes = DEFAULT_VECTOR_WIDTH_IN_BYTES;

  llvm::PipelineTuningOptions pipelineTuningOptions;
  // Optimization level to be used by the LLVM optimizer (middle-end).
  llvm::OptimizationLevel optimizerOptLevel;
  // Optimization level to be used by the LLVM code generator (back-end).
  llvm::CodeGenOptLevel codeGenOptLevel;
  llvm::TargetOptions llvmTargetOptions;

  bool getLinkEmbedded() const { return linkEmbedded; }

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

  // Enables ukernels in the generated executables. May be `default`, `none`,
  // `all`, or a comma-separated list of specific unprefixed ukernels to
  // enable, e.g. `mmt4d`.
  std::string ukernels = DEFAULT_ENABLE_UKERNELS;

  // Link built-in ukernel bitcode libraries into generated executables.
  bool linkUkernelBitcode = DEFAULT_LINK_UKERNEL_BITCODE;

private:
  void populateDefaultsFromTargetMachine();

  std::string triple;
  std::string cpu;
  std::string cpuFeatures;

  // Build for the IREE embedded platform-agnostic ELF loader.
  // Note: this is ignored for target machines that do not support the ELF
  // loader, such as WebAssembly.
  bool linkEmbedded = DEFAULT_LINK_EMBEDDED;

  friend struct LLVMTargetOptions;
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

  // Ensures that command line flags are registered. Should be called on
  // startup.
  static void registerFlags();

  // Returns LLVMTargetOptions struct initialized with the iree-llvmcpu-* flags.
  static LLVMTargetOptions getFromFlags();

private:
  static void initializeFromFlags(LLVMTargetOptions &targetOptions);
  void initializeTargetInvariantFlags();
};

// Creates target machine form target options.
std::unique_ptr<llvm::TargetMachine>
createTargetMachine(const LLVMTarget &target);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMTARGETOPTIONS_H_
