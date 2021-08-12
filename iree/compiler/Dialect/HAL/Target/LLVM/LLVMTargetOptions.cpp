// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

LLVMTargetOptions getDefaultLLVMTargetOptions() {
  LLVMTargetOptions targetOptions;

  // Host target triple.
  targetOptions.targetTriple = llvm::sys::getDefaultTargetTriple();
  targetOptions.targetCPU = llvm::sys::getHostCPUName().str();
  {
    llvm::SubtargetFeatures features;
    llvm::StringMap<bool> hostFeatures;
    if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
      for (auto &feature : hostFeatures) {
        features.AddFeature(feature.first(), feature.second);
      }
    }
    targetOptions.targetCPUFeatures = features.getString();
  }

  // LLVM loop optimization options.
  targetOptions.pipelineTuningOptions.LoopInterleaving = true;
  targetOptions.pipelineTuningOptions.LoopVectorization = true;
  targetOptions.pipelineTuningOptions.LoopUnrolling = true;

  // LLVM SLP Auto vectorizer.
  targetOptions.pipelineTuningOptions.SLPVectorization = true;

  // LLVM -O3.
  // TODO(benvanik): add an option for this.
  targetOptions.optLevel = llvm::OptimizationLevel::O3;
  targetOptions.options.FloatABIType = llvm::FloatABI::Hard;

  return targetOptions;
}

LLVMTargetOptions getLLVMTargetOptionsFromFlags() {
  auto targetOptions = getDefaultLLVMTargetOptions();

  static llvm::cl::opt<std::string> clTargetTriple(
      "iree-llvm-target-triple", llvm::cl::desc("LLVM target machine triple"),
      llvm::cl::init(targetOptions.targetTriple));
  static llvm::cl::opt<std::string> clTargetCPU(
      "iree-llvm-target-cpu",
      llvm::cl::desc(
          "LLVM target machine CPU; use 'host' for your host native CPU"),
      llvm::cl::init("generic"));
  static llvm::cl::opt<std::string> clTargetCPUFeatures(
      "iree-llvm-target-cpu-features",
      llvm::cl::desc("LLVM target machine CPU features; use 'host' for your "
                     "host native CPU"),
      llvm::cl::init(""));

  static llvm::cl::opt<bool> llvmLoopInterleaving(
      "iree-llvm-loop-interleaving", llvm::cl::init(false),
      llvm::cl::desc("Enable LLVM loop interleaving opt"));
  static llvm::cl::opt<bool> llvmLoopVectorization(
      "iree-llvm-loop-vectorization", llvm::cl::init(true),
      llvm::cl::desc("Enable LLVM loop vectorization opt"));
  static llvm::cl::opt<bool> llvmLoopUnrolling(
      "iree-llvm-loop-unrolling", llvm::cl::init(false),
      llvm::cl::desc("Enable LLVM loop unrolling opt"));
  static llvm::cl::opt<bool> llvmSLPVectorization(
      "iree-llvm-slp-vectorization", llvm::cl::init(false),
      llvm::cl::desc("Enable LLVM SLP Vectorization opt"));

  targetOptions.targetTriple = clTargetTriple;
  if (clTargetCPU != "host") {
    targetOptions.targetCPU = clTargetCPU;
  }
  if (clTargetCPUFeatures != "host") {
    targetOptions.targetCPUFeatures = clTargetCPUFeatures;
  }

  // LLVM opt options.
  targetOptions.pipelineTuningOptions.LoopInterleaving = llvmLoopInterleaving;
  targetOptions.pipelineTuningOptions.LoopVectorization = llvmLoopVectorization;
  targetOptions.pipelineTuningOptions.LoopUnrolling = llvmLoopUnrolling;
  targetOptions.pipelineTuningOptions.SLPVectorization = llvmSLPVectorization;

  static llvm::cl::opt<SanitizerKind> clSanitizerKind(
      "iree-llvm-sanitize", llvm::cl::desc("Apply LLVM sanitize feature"),
      llvm::cl::init(SanitizerKind::kNone),
      llvm::cl::values(clEnumValN(SanitizerKind::kAddress, "address",
                                  "Address sanitizer support")));
  targetOptions.sanitizerKind = clSanitizerKind;

  static llvm::cl::opt<std::string> clTargetABI(
      "iree-llvm-target-abi",
      llvm::cl::desc("LLVM target machine ABI; specify for -mabi"),
      llvm::cl::init(""));
  targetOptions.options.MCOptions.ABIName = clTargetABI;

  static llvm::cl::opt<llvm::FloatABI::ABIType> clTargetFloatABI(
      "iree-llvm-target-float-abi",
      llvm::cl::desc("LLVM target codegen enables soft float abi e.g "
                     "-mfloat-abi=softfp"),
      llvm::cl::init(targetOptions.options.FloatABIType),
      llvm::cl::values(
          clEnumValN(llvm::FloatABI::Default, "default", "Default (softfp)"),
          clEnumValN(llvm::FloatABI::Soft, "soft",
                     "Software floating-point emulation"),
          clEnumValN(llvm::FloatABI::Hard, "hard",
                     "Hardware floating-point instructions")));
  targetOptions.options.FloatABIType = clTargetFloatABI;

  static llvm::cl::opt<bool> clDebugSymbols(
      "iree-llvm-debug-symbols",
      llvm::cl::desc("Generate and embed debug information (DWARF, PDB, etc)"),
      llvm::cl::init(targetOptions.debugSymbols));
  targetOptions.debugSymbols = clDebugSymbols;

  static llvm::cl::opt<std::string> clLinkerPath(
      "iree-llvm-linker-path",
      llvm::cl::desc("Tool used to link shared libraries produced by IREE."),
      llvm::cl::init(""));
  targetOptions.linkerPath = clLinkerPath;

  static llvm::cl::opt<std::string> clEmbeddedLinkerPath(
      "iree-llvm-embedded-linker-path",
      llvm::cl::desc("Tool used to link embedded ELFs produced by IREE (for "
                     "-iree-llvm-link-embedded)."),
      llvm::cl::init("ld.lld"));
  targetOptions.embeddedLinkerPath = clEmbeddedLinkerPath;

  static llvm::cl::opt<bool> clLinkEmbedded(
      "iree-llvm-link-embedded",
      llvm::cl::desc("Links binaries into a platform-agnostic ELF to be loaded "
                     "by the embedded IREE ELF loader"),
      llvm::cl::init(targetOptions.linkEmbedded));
  targetOptions.linkEmbedded = clLinkEmbedded;

  static llvm::cl::opt<bool> clLinkStatic(
      "iree-llvm-link-static",
      llvm::cl::desc(
          "Links system libraries into binaries statically to isolate them "
          "from platform dependencies needed at runtime"),
      llvm::cl::init(targetOptions.linkStatic));
  targetOptions.linkStatic = clLinkStatic;

  static llvm::cl::opt<bool> clKeepLinkerArtifacts(
      "iree-llvm-keep-linker-artifacts",
      llvm::cl::desc("Keep LLVM linker target artifacts (.so/.dll/etc)"),
      llvm::cl::init(targetOptions.keepLinkerArtifacts));
  targetOptions.keepLinkerArtifacts = clKeepLinkerArtifacts;

  static llvm::cl::opt<std::string> clStaticLibraryOutputPath(
      "iree-llvm-static-library-output-path",
      llvm::cl::desc(
          "Path to output static object (EX: '/path/to/static-library.o'). "
          "This will produce the static library at the specified path along "
          "with a similarly named '.h' file for static linking."),
      llvm::cl::init(targetOptions.staticLibraryOutput));
  targetOptions.staticLibraryOutput = clStaticLibraryOutputPath;

  static llvm::cl::opt<bool> clListTargets(
      "iree-llvm-list-targets",
      llvm::cl::desc("Lists all registered targets that the LLVM backend can "
                     "generate code for."),
      llvm::cl::init(false));
  if (clListTargets) {
    llvm::TargetRegistry::printRegisteredTargetsForVersion(llvm::outs());
    exit(0);
  }

  // Force the triple to something compatible with embedded linking when
  // enabled.
  if (targetOptions.linkEmbedded) {
    llvm::Triple triple(targetOptions.targetTriple);
    triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
    triple.setEnvironment(llvm::Triple::EnvironmentType::EABI);
    triple.setOS(llvm::Triple::OSType::UnknownOS);
    triple.setObjectFormat(llvm::Triple::ObjectFormatType::ELF);
    targetOptions.targetTriple = triple.str();
  }

  return targetOptions;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
