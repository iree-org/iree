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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
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
  targetOptions.optLevel = llvm::PassBuilder::OptimizationLevel::O3;
  targetOptions.options.FloatABIType = llvm::FloatABI::Hard;

  return targetOptions;
}

LLVMTargetOptions getLLVMTargetOptionsFromFlags() {
  auto llvmTargetOptions = getDefaultLLVMTargetOptions();

  static llvm::cl::opt<std::string> clTargetTriple(
      "iree-llvm-target-triple", llvm::cl::desc("LLVM target machine triple"),
      llvm::cl::init(llvmTargetOptions.targetTriple));
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

  llvmTargetOptions.targetTriple = clTargetTriple;
  if (clTargetCPU != "host") {
    llvmTargetOptions.targetCPU = clTargetCPU;
  }
  if (clTargetCPUFeatures != "host") {
    llvmTargetOptions.targetCPUFeatures = clTargetCPUFeatures;
  }

  // LLVM opt options.
  llvmTargetOptions.pipelineTuningOptions.LoopInterleaving =
      llvmLoopInterleaving;
  llvmTargetOptions.pipelineTuningOptions.LoopVectorization =
      llvmLoopVectorization;
  llvmTargetOptions.pipelineTuningOptions.LoopUnrolling = llvmLoopUnrolling;
  llvmTargetOptions.pipelineTuningOptions.SLPVectorization =
      llvmSLPVectorization;

  static llvm::cl::opt<SanitizerKind> clSanitizerKind(
      "iree-llvm-sanitize", llvm::cl::desc("Apply LLVM sanitize feature"),
      llvm::cl::init(SanitizerKind::kNone),
      llvm::cl::values(clEnumValN(SanitizerKind::kAddress, "address",
                                  "Address sanitizer support")));
  llvmTargetOptions.sanitizerKind = clSanitizerKind;

  static llvm::cl::opt<std::string> clTargetABI(
      "iree-llvm-target-abi",
      llvm::cl::desc("LLVM target machine ABI; specify for -mabi"),
      llvm::cl::init(""));
  llvmTargetOptions.options.MCOptions.ABIName = clTargetABI;

  static llvm::cl::opt<llvm::FloatABI::ABIType> clTargetFloatABI(
      "iree-llvm-target-float-abi",
      llvm::cl::desc("LLVM target codegen enables soft float abi e.g "
                     "-mfloat-abi=softfp"),
      llvm::cl::init(llvmTargetOptions.options.FloatABIType),
      llvm::cl::values(
          clEnumValN(llvm::FloatABI::Default, "default", "Default (softfp)"),
          clEnumValN(llvm::FloatABI::Soft, "soft",
                     "Software floating-point emulation"),
          clEnumValN(llvm::FloatABI::Hard, "hard",
                     "Hardware floating-point instructions")));
  llvmTargetOptions.options.FloatABIType = clTargetFloatABI;

  static llvm::cl::opt<bool> clDebugSymbols(
      "iree-llvm-debug-symbols",
      llvm::cl::desc("Generate and embed debug information (DWARF, PDB, etc)"),
      llvm::cl::init(llvmTargetOptions.debugSymbols));
  llvmTargetOptions.debugSymbols = clDebugSymbols;

  static llvm::cl::opt<bool> clLinkStatic(
      "iree-llvm-link-static",
      llvm::cl::desc(
          "Links system libraries into binaries statically to isolate them "
          "from platform dependencies needed at runtime"),
      llvm::cl::init(llvmTargetOptions.linkStatic));
  llvmTargetOptions.linkStatic = clLinkStatic;

  static llvm::cl::opt<bool> clKeepLinkerArtifacts(
      "iree-llvm-keep-linker-artifacts",
      llvm::cl::desc("Keep LLVM linker target artifacts (.so/.dll/etc)"),
      llvm::cl::init(llvmTargetOptions.keepLinkerArtifacts));
  llvmTargetOptions.keepLinkerArtifacts = clKeepLinkerArtifacts;

  return llvmTargetOptions;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
