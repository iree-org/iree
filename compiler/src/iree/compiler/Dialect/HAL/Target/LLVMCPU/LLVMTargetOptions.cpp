// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMTargetOptions.h"

#include <mutex>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/X86TargetParser.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

bool resolveCPUAndCPUFeatures(llvm::StringRef inCpu,
                              llvm::StringRef inCpuFeatures,
                              const llvm::Triple &triple, std::string &outCpu,
                              std::string &outCpuFeatures) {
  // Resolve "host"
  if (inCpu == "host" || inCpuFeatures == "host") {
    // If either Cpu or CpuFeatures is "host", the other must be either also
    // host or the default value.
    bool isCpuHostOrDefault =
        inCpu.empty() || inCpu == "host" || inCpu == "generic";
    bool isCpuFeaturesHostOrDefault =
        inCpuFeatures.empty() || inCpuFeatures == "host";
    if (!(isCpuHostOrDefault && isCpuFeaturesHostOrDefault)) {
      llvm::errs()
          << "error: If either cpu or CpuFeatures is `host`, the other must "
             "be either also `host` or the default value\n";
      return false;
    }
    outCpu = triple.isX86() ? llvm::sys::getHostCPUName().str() : "";
    llvm::SubtargetFeatures features;
    llvm::StringMap<bool> hostFeatures;
    if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
      for (auto &feature : hostFeatures) {
        features.AddFeature(feature.first(), feature.second);
      }
    }
    outCpuFeatures = features.getString();
  } else {
    outCpu = inCpu;
    outCpuFeatures = inCpuFeatures;
  }

  // Target-specific CPU feature tweaks that we need unconditionally.
  if (triple.isAArch64()) {
    llvm::SubtargetFeatures targetCpuFeatures(outCpuFeatures);
    // x18 is platform-reserved per the Aarch64 procedure call specification.
    targetCpuFeatures.AddFeature("reserve-x18", true);
    outCpuFeatures = targetCpuFeatures.getString();
  }

  // If CPU is non-host and non-generic then we need to populate the
  // corresponding features.
  if (outCpu.empty() || inCpu == "host" || inCpu == "generic" ||
      inCpu.starts_with("generic-")) {
    return true;
  }
  if (triple.isX86()) {
    llvm::SubtargetFeatures targetCpuFeatures(outCpuFeatures);
    llvm::SmallVector<llvm::StringRef> cpuFeatureList;
    llvm::X86::getFeaturesForCPU(outCpu, cpuFeatureList);
    for (auto &feature : cpuFeatureList) {
      targetCpuFeatures.AddFeature(feature);
    }
    outCpuFeatures = targetCpuFeatures.getString();
  } else {
    llvm::errs()
        << "error: Resolution of target CPU to target CPU features is not "
           "implemented on "
           "this target architecture. Pass explicit CPU features "
           "instead of a CPU "
           "on this architecture, or implement that.\n";
    return false;
  }
  return true;
}

} // namespace

LLVMTarget::LLVMTarget() {
  // LLVM loop optimization options.
  pipelineTuningOptions.LoopInterleaving = DEFAULT_LOOP_INTERLEAVING;
  pipelineTuningOptions.LoopVectorization = DEFAULT_LOOP_VECTORIZATION;
  pipelineTuningOptions.LoopUnrolling = DEFAULT_LOOP_UNROLLING;

  // LLVM SLP Auto vectorizer.
  pipelineTuningOptions.SLPVectorization = DEFAULT_SLP_VECTORIZATION;

  // LLVM optimization levels.
  // TODO(benvanik): add an option for this.
  optimizerOptLevel = llvm::OptimizationLevel::O2;
  codeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  llvmTargetOptions.FloatABIType = DEFAULT_FLOAT_ABI;

  // Force `-ffunction-sections` so we can strip unused code.
  llvmTargetOptions.FunctionSections = true;
  llvmTargetOptions.DataSections = true;
  llvmTargetOptions.UniqueSectionNames = true;
}

std::optional<LLVMTarget> LLVMTarget::create(std::string_view triple,
                                             std::string_view cpu,
                                             std::string_view cpuFeatures,
                                             bool requestLinkEmbedded) {
  LLVMTarget target;
  target.linkEmbedded = requestLinkEmbedded;

  target.triple = triple;
  llvm::Triple targetTriple(target.triple);
  // Special casing if linkEmbedded.
  if (targetTriple.isWasm()) {
    // The embedded ELF loader is not supported on WebAssembly, so force it off.
    target.linkEmbedded = false;
  }
  if (target.linkEmbedded) {
    // Force the triple to something compatible with embedded linking.
    targetTriple.setVendor(llvm::Triple::VendorType::UnknownVendor);
    targetTriple.setEnvironment(llvm::Triple::EnvironmentType::EABI);
    targetTriple.setOS(llvm::Triple::OSType::UnknownOS);
    targetTriple.setObjectFormat(llvm::Triple::ObjectFormatType::ELF);
    target.triple = targetTriple.str();
  }
  if (!resolveCPUAndCPUFeatures(cpu, cpuFeatures, llvm::Triple(triple),
                                target.cpu, target.cpuFeatures)) {
    // Something bad happened, and our target might not be what the user expects
    // but we need to continue to avoid breaking existing users. Hopefully
    // resolveCPUAndCPUFeatures logged a helpful error already.
  }
  return target;
}

std::optional<LLVMTarget> LLVMTarget::createForHost() {
  return LLVMTarget::create(llvm::sys::getProcessTriple(), /*cpu=*/"host",
                            /*cpuFeatures=*/"host",
                            /*requestLinkEmbedded=*/true);
}

void LLVMTarget::print(llvm::raw_ostream &os) const {
  os << "LLVMTarget{\n"
     << "  triple=" << triple << ", cpu=" << cpu
     << ", cpuFeatures=" << cpuFeatures << "\n"
     << "  linkEmbedded=" << linkEmbedded << "\n"
     << "  debugSymbols=" << debugSymbols << "\n"
     << "  sanitizer=" << static_cast<int>(sanitizerKind) << "\n"
     << "  staticLibraryOutput=" << staticLibraryOutput << "\n"
     << "  linkStatic=" << linkStatic << "\n"
     << "  pipelineTuningOptions={\n"
     << "    LoopInterleaving=" << pipelineTuningOptions.LoopInterleaving
     << "\n"
     << "    LoopVectorization=" << pipelineTuningOptions.LoopVectorization
     << "\n"
     << "    LoopUnrolling=" << pipelineTuningOptions.LoopUnrolling << "\n"
     << "    SLPVectorization=" << pipelineTuningOptions.SLPVectorization
     << "\n"
     << "  }, llvmTargetOptions={\n"
     << "    FloatABIType=" << static_cast<int>(llvmTargetOptions.FloatABIType)
     << "\n"
     << "  }\n"
     << "}\n";
}

void LLVMTarget::storeToConfigAttrs(MLIRContext *context,
                                    SmallVector<NamedAttribute> &config) const {
  Builder b(context);
  auto addString = [&](StringRef name, StringRef value) {
    config.emplace_back(b.getStringAttr(name), b.getStringAttr(value));
  };
  auto addBool = [&](StringRef name, bool value) {
    config.emplace_back(b.getStringAttr(name), b.getBoolAttr(value));
  };

  addString("target_triple", triple);
  addString("cpu", cpu);
  addString("cpu_features", cpuFeatures);
  if (linkEmbedded != DEFAULT_LINK_EMBEDDED) {
    addBool("link_embedded", linkEmbedded);
  }
  if (debugSymbols != DEFAULT_DEBUG_SYMBOLS) {
    addBool("debug_symbols", debugSymbols);
  }
  if (linkStatic != DEFAULT_LINK_STATIC) {
    addBool("link_static", linkStatic);
  }
  if (sanitizerKind != DEFAULT_SANITIZER_KIND) {
    switch (sanitizerKind) {
    case SanitizerKind::kNone:
      addString("sanitizer", "none");
      break;
    case SanitizerKind::kAddress:
      addString("sanitizer", "address");
      break;
    case SanitizerKind::kThread:
      addString("sanitizer", "thread");
      break;
    }
  }
  if (!staticLibraryOutput.empty()) {
    addString("static_library_output", staticLibraryOutput);
  }
  if (pipelineTuningOptions.LoopInterleaving != DEFAULT_LOOP_INTERLEAVING)
    addBool("loop_interleaving", DEFAULT_LOOP_INTERLEAVING);
  if (pipelineTuningOptions.LoopVectorization != DEFAULT_LOOP_VECTORIZATION)
    addBool("loop_vectorization", DEFAULT_LOOP_VECTORIZATION);
  if (pipelineTuningOptions.LoopUnrolling != DEFAULT_LOOP_UNROLLING)
    addBool("loop_unrolling", DEFAULT_LOOP_UNROLLING);
  if (pipelineTuningOptions.SLPVectorization != DEFAULT_SLP_VECTORIZATION)
    addBool("slp_vectorization", DEFAULT_SLP_VECTORIZATION);
  if (!llvmTargetOptions.MCOptions.ABIName.empty())
    addString("target_abi", llvmTargetOptions.MCOptions.ABIName);
  if (llvmTargetOptions.FloatABIType != DEFAULT_FLOAT_ABI) {
    switch (llvmTargetOptions.FloatABIType) {
    case llvm::FloatABI::Default:
      addString("float_abi", "default");
      break;
    case llvm::FloatABI::Soft:
      addString("float_abi", "soft");
      break;
    case llvm::FloatABI::Hard:
      addString("float_abi", "hard");
      break;
    }
  }
}

std::optional<LLVMTarget>
LLVMTarget::loadFromConfigAttr(Location loc, DictionaryAttr config,
                               const LLVMTarget &defaultTarget) {
  bool hasFailures = false;
  auto getString = [&](StringRef name, StringRef fallback,
                       bool required) -> StringRef {
    Attribute attr = config.get(name);
    if (auto sattr = llvm::dyn_cast_if_present<StringAttr>(attr)) {
      return sattr.strref();
    } else {
      if (required) {
        hasFailures = true;
        emitError(loc) << "executable config '" << name
                       << "' required but not present on attribute";
      }
      return fallback;
    }
  };
  auto getOptionalString = [&](StringRef name) -> std::optional<StringRef> {
    Attribute attr = config.get(name);
    if (auto sattr = llvm::dyn_cast_if_present<StringAttr>(attr)) {
      return sattr.strref();
    } else if (attr) {
      hasFailures = true;
      emitError(loc) << "executable config '" << name
                     << "' requires string but got " << attr;
    }
    return {};
  };
  auto getBoolValue = [&](StringRef name, bool fallback) -> bool {
    Attribute attr = config.get(name);
    if (auto battr = llvm::dyn_cast_if_present<BoolAttr>(attr)) {
      return battr.getValue();
    } else if (attr) {
      hasFailures = true;
      emitError(loc) << "executable config '" << name
                     << "' requires bool but got " << attr;
    }
    return fallback;
  };

  LLVMTarget target;

  // Constructor arguments.
  auto triple = getOptionalString("target_triple");
  auto cpu = getOptionalString("cpu");
  auto cpuFeatures = getOptionalString("cpu_features");
  bool linkEmbedded = getBoolValue("link_embedded", DEFAULT_LINK_EMBEDDED);
  if (triple || cpu || cpuFeatures) {
    if (!triple) {
      emitError(loc) << "executable config 'cpu' or 'cpu_features' must be "
                        "accompanied by 'target_triple'";
      return {};
    }
    std::optional<LLVMTarget> maybeTarget =
        LLVMTarget::create(*triple, cpu ? *cpu : "generic",
                           cpuFeatures ? *cpuFeatures : "", linkEmbedded);
    if (!maybeTarget) {
      return {};
    }
    target.copy(*maybeTarget);
  } else {
    target.copy(defaultTarget);
  }

  // Loose items.
  target.debugSymbols = getBoolValue("debug_symbols", DEFAULT_DEBUG_SYMBOLS);
  target.linkStatic = getBoolValue("link_static", DEFAULT_LINK_STATIC);
  auto sanitizer = getOptionalString("sanitizer");
  if (sanitizer) {
    if (sanitizer == "none")
      target.sanitizerKind = SanitizerKind::kNone;
    else if (sanitizer == "address")
      target.sanitizerKind = SanitizerKind::kAddress;
    else if (sanitizer == "thread")
      target.sanitizerKind = SanitizerKind::kThread;
    else {
      emitError(loc) << "executable config unexpected value for 'sanitizer': "
                     << *sanitizer;
      return {};
    }
  }
  target.staticLibraryOutput = getString("static_library_output", "", false);
  target.pipelineTuningOptions.LoopInterleaving = getBoolValue(
      "loop_interleaving", target.pipelineTuningOptions.LoopInterleaving);
  target.pipelineTuningOptions.LoopVectorization = getBoolValue(
      "loop_vectorization", target.pipelineTuningOptions.LoopVectorization);
  target.pipelineTuningOptions.LoopUnrolling = getBoolValue(
      "loop_unrolling", target.pipelineTuningOptions.LoopUnrolling);
  target.pipelineTuningOptions.SLPVectorization = getBoolValue(
      "slp_vectorization", target.pipelineTuningOptions.SLPVectorization);
  auto targetAbi = getOptionalString("target_abi");
  if (targetAbi)
    target.llvmTargetOptions.MCOptions.ABIName = *targetAbi;
  auto floatAbi = getOptionalString("float_abi");
  if (floatAbi) {
    if (floatAbi == "default")
      target.llvmTargetOptions.FloatABIType = llvm::FloatABI::Default;
    else if (floatAbi == "soft")
      target.llvmTargetOptions.FloatABIType = llvm::FloatABI::Default;
    else if (floatAbi == "hard")
      target.llvmTargetOptions.FloatABIType = llvm::FloatABI::Default;
    else {
      emitError(loc) << "executable config unexpected value for 'float_abi'";
      return {};
    }
  }

  if (hasFailures) {
    return {};
  }
  return target;
}

void LLVMTargetOptions::initializeTargetInvariantFlags() {
  static llvm::cl::opt<std::string> clSystemLinkerPath(
      "iree-llvmcpu-system-linker-path",
      llvm::cl::desc("Tool used to link system shared libraries produced by "
                     "IREE (for --iree-llvmcpu-link-embedded=false)."),
      llvm::cl::init(""));
  systemLinkerPath = clSystemLinkerPath;

  static llvm::cl::opt<std::string> clEmbeddedLinkerPath(
      "iree-llvmcpu-embedded-linker-path",
      llvm::cl::desc("Tool used to link embedded ELFs produced by IREE (for "
                     "--iree-llvmcpu-link-embedded=true)."),
      llvm::cl::init(""));
  embeddedLinkerPath = clEmbeddedLinkerPath;

  static llvm::cl::opt<std::string> clWasmLinkerPath(
      "iree-llvmcpu-wasm-linker-path",
      llvm::cl::desc("Tool used to link WebAssembly modules produced by "
                     "IREE (for --iree-llvmcpu-target-triple=wasm32-*)."),
      llvm::cl::init(""));
  wasmLinkerPath = clWasmLinkerPath;

  static llvm::cl::opt<bool> clKeepLinkerArtifacts(
      "iree-llvmcpu-keep-linker-artifacts",
      llvm::cl::desc("Keep LLVM linker target artifacts (.so/.dll/etc)"),
      llvm::cl::init(keepLinkerArtifacts));
  keepLinkerArtifacts = clKeepLinkerArtifacts;
}

LLVMTargetOptions LLVMTargetOptions::getHostOptions() {
  LLVMTargetOptions targetOptions;
  std::optional<LLVMTarget> maybeTarget = LLVMTarget::createForHost();
  if (!maybeTarget)
    return {};
  targetOptions.target = *maybeTarget;
  targetOptions.initializeTargetInvariantFlags();
  return targetOptions;
}

LLVMTargetOptions LLVMTargetOptions::getFromFlags() {
  LLVMTargetOptions targetOptions;
  targetOptions.initializeTargetInvariantFlags();

  // Target parameters.
  static llvm::cl::opt<std::string> clTargetTriple(
      "iree-llvmcpu-target-triple",
      llvm::cl::desc("LLVM target machine triple"),
      llvm::cl::init(llvm::sys::getProcessTriple()));
  static llvm::cl::opt<std::string> clTargetCPU(
      "iree-llvmcpu-target-cpu",
      llvm::cl::desc(
          "LLVM target machine CPU; use 'host' for your host native CPU"),
      llvm::cl::init("generic"));
  static llvm::cl::opt<std::string> clTargetCPUFeatures(
      "iree-llvmcpu-target-cpu-features",
      llvm::cl::desc("LLVM target machine CPU features; use 'host' for your "
                     "host native CPU"),
      llvm::cl::init(""));
  static llvm::cl::opt<bool> clLinkEmbedded(
      "iree-llvmcpu-link-embedded",
      llvm::cl::desc("Links binaries into a platform-agnostic ELF to be loaded "
                     "by the embedded IREE ELF loader"),
      llvm::cl::init(LLVMTarget::DEFAULT_LINK_EMBEDDED));
  std::optional<LLVMTarget> maybeTarget =
      LLVMTarget::create(clTargetTriple, clTargetCPU, clTargetCPUFeatures,
                         /*requestLinkEmbedded=*/clLinkEmbedded);
  if (maybeTarget) {
    targetOptions.target = *maybeTarget;
  } else {
    llvm::errs() << "Inconsistency in iree-llvmcpu-target-cpu-* command-line "
                    "flags. The target CPU is not properly defined.\n";
  }
  LLVMTarget &target = targetOptions.target;

  static llvm::cl::opt<bool> llvmLoopInterleaving(
      "iree-llvmcpu-loop-interleaving",
      llvm::cl::init(LLVMTarget::DEFAULT_LOOP_INTERLEAVING),
      llvm::cl::desc("Enable LLVM loop interleaving opt"));
  static llvm::cl::opt<bool> llvmLoopVectorization(
      "iree-llvmcpu-loop-vectorization",
      llvm::cl::init(LLVMTarget::DEFAULT_LOOP_VECTORIZATION),
      llvm::cl::desc("Enable LLVM loop vectorization opt"));
  static llvm::cl::opt<bool> llvmLoopUnrolling(
      "iree-llvmcpu-loop-unrolling",
      llvm::cl::init(LLVMTarget::DEFAULT_LOOP_UNROLLING),
      llvm::cl::desc("Enable LLVM loop unrolling opt"));
  static llvm::cl::opt<bool> llvmSLPVectorization(
      "iree-llvmcpu-slp-vectorization",
      llvm::cl::init(LLVMTarget::DEFAULT_SLP_VECTORIZATION),
      llvm::cl::desc("Enable LLVM SLP Vectorization opt"));

  // LLVM opt options.
  target.pipelineTuningOptions.LoopInterleaving = llvmLoopInterleaving;
  target.pipelineTuningOptions.LoopVectorization = llvmLoopVectorization;
  target.pipelineTuningOptions.LoopUnrolling = llvmLoopUnrolling;
  target.pipelineTuningOptions.SLPVectorization = llvmSLPVectorization;

  static llvm::cl::opt<SanitizerKind> clSanitizerKind(
      "iree-llvmcpu-sanitize", llvm::cl::desc("Apply LLVM sanitize feature"),
      llvm::cl::init(SanitizerKind::kNone),
      llvm::cl::values(clEnumValN(SanitizerKind::kAddress, "address",
                                  "Address sanitizer support"),
                       clEnumValN(SanitizerKind::kThread, "thread",
                                  "Thread sanitizer support")));
  target.sanitizerKind = clSanitizerKind;

  static llvm::cl::opt<std::string> clTargetABI(
      "iree-llvmcpu-target-abi",
      llvm::cl::desc("LLVM target machine ABI; specify for -mabi"),
      llvm::cl::init(""));
  target.llvmTargetOptions.MCOptions.ABIName = clTargetABI;

  static llvm::cl::opt<llvm::FloatABI::ABIType> clTargetFloatABI(
      "iree-llvmcpu-target-float-abi",
      llvm::cl::desc("LLVM target codegen enables soft float abi e.g "
                     "-mfloat-abi=softfp"),
      llvm::cl::init(target.llvmTargetOptions.FloatABIType),
      llvm::cl::values(
          clEnumValN(llvm::FloatABI::Default, "default", "Default (softfp)"),
          clEnumValN(llvm::FloatABI::Soft, "soft",
                     "Software floating-point emulation"),
          clEnumValN(llvm::FloatABI::Hard, "hard",
                     "Hardware floating-point instructions")));
  target.llvmTargetOptions.FloatABIType = clTargetFloatABI;

  static llvm::cl::opt<bool> clDebugSymbols(
      "iree-llvmcpu-debug-symbols",
      llvm::cl::desc("Generate and embed debug information (DWARF, PDB, etc)"),
      llvm::cl::init(target.debugSymbols));
  target.debugSymbols = clDebugSymbols;

  static llvm::cl::opt<bool> clLinkStatic(
      "iree-llvmcpu-link-static",
      llvm::cl::desc(
          "Links system libraries into binaries statically to isolate them "
          "from platform dependencies needed at runtime"),
      llvm::cl::init(target.linkStatic));
  target.linkStatic = clLinkStatic;

  static llvm::cl::opt<std::string> clStaticLibraryOutputPath(
      "iree-llvmcpu-static-library-output-path",
      llvm::cl::desc(
          "Path to output static object (EX: '/path/to/static-library.o'). "
          "This will produce the static library at the specified path along "
          "with a similarly named '.h' file for static linking."),
      llvm::cl::init(target.staticLibraryOutput));
  target.staticLibraryOutput = clStaticLibraryOutputPath;

  static llvm::cl::opt<bool> clListTargets(
      "iree-llvmcpu-list-targets",
      llvm::cl::desc("Lists all registered targets that the LLVM backend can "
                     "generate code for."),
      llvm::cl::init(false), llvm::cl::ValueDisallowed,
      llvm::cl::callback([&](const bool &) {
        llvm::TargetRegistry::printRegisteredTargetsForVersion(llvm::outs());
        exit(0);
      }));

  return targetOptions;
}

} // namespace mlir::iree_compiler::IREE::HAL
