// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMTargetOptions.h"

#include <mutex>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
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
  auto target =
      LLVMTarget::create(llvm::sys::getProcessTriple(), /*cpu=*/"host",
                         /*cpuFeatures=*/"host",
                         /*requestLinkEmbedded=*/true);
  if (target)
    target->populateDefaultsFromTargetMachine();
  return target;
}

void LLVMTarget::print(llvm::raw_ostream &os) const {
  os << "LLVMTarget{\n"
     << "  triple=" << triple << ", cpu=" << cpu
     << ", cpuFeatures=" << cpuFeatures << "\n"
     << "  dataLayout=" << dataLayout << "\n"
     << "  vectorWidthInBytes=" << vectorWidthInBytes << "\n"
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
     << "  ukernels=" << ukernels << "\n"
     << "  linkUkernelBitcode=" << linkUkernelBitcode << "\n"
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
  auto addInt64 = [&](StringRef name, int64_t value) {
    config.emplace_back(b.getStringAttr(name), b.getI64IntegerAttr(value));
  };

  addString("target_triple", triple);
  addString("cpu", cpu);
  addString("cpu_features", cpuFeatures);
  if (!dataLayout.empty()) {
    addString("data_layout", dataLayout);
  }
  if (vectorWidthInBytes != DEFAULT_VECTOR_WIDTH_IN_BYTES) {
    addInt64("native_vector_size", vectorWidthInBytes);
  }
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
    addBool("loop_interleaving", pipelineTuningOptions.LoopInterleaving);
  if (pipelineTuningOptions.LoopVectorization != DEFAULT_LOOP_VECTORIZATION)
    addBool("loop_vectorization", pipelineTuningOptions.LoopVectorization);
  if (pipelineTuningOptions.LoopUnrolling != DEFAULT_LOOP_UNROLLING)
    addBool("loop_unrolling", pipelineTuningOptions.LoopUnrolling);
  if (pipelineTuningOptions.SLPVectorization != DEFAULT_SLP_VECTORIZATION)
    addBool("slp_vectorization", pipelineTuningOptions.SLPVectorization);
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
  if (ukernels.compare(DEFAULT_ENABLE_UKERNELS) != 0)
    addString("ukernels", ukernels);
  if (linkUkernelBitcode != DEFAULT_LINK_UKERNEL_BITCODE)
    addBool("link_ukernel_bitcode", linkUkernelBitcode);
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
  auto getBool = [&](StringRef name, bool fallback) -> bool {
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
  auto getInt64 = [&](StringRef name, int64_t fallback) -> int64_t {
    Attribute attr = config.get(name);
    if (auto iattr = llvm::dyn_cast_if_present<IntegerAttr>(attr)) {
      return iattr.getValue().getSExtValue();
    } else if (attr) {
      hasFailures = true;
      emitError(loc) << "executable config '" << name
                     << "' requires i64 but got " << attr;
    }
    return fallback;
  };

  LLVMTarget target;

  // Constructor arguments.
  auto triple = getOptionalString("target_triple");
  auto cpu = getOptionalString("cpu");
  auto cpuFeatures = getOptionalString("cpu_features");
  bool linkEmbedded = getBool("link_embedded", DEFAULT_LINK_EMBEDDED);
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

  target.dataLayout = getString("data_layout", DEFAULT_DATA_LAYOUT, false);
  target.vectorWidthInBytes =
      getInt64("native_vector_size", DEFAULT_VECTOR_WIDTH_IN_BYTES);

  target.debugSymbols = getBool("debug_symbols", DEFAULT_DEBUG_SYMBOLS);
  target.linkStatic = getBool("link_static", DEFAULT_LINK_STATIC);
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

  target.pipelineTuningOptions.LoopInterleaving = getBool(
      "loop_interleaving", target.pipelineTuningOptions.LoopInterleaving);
  target.pipelineTuningOptions.LoopVectorization = getBool(
      "loop_vectorization", target.pipelineTuningOptions.LoopVectorization);
  target.pipelineTuningOptions.LoopUnrolling =
      getBool("loop_unrolling", target.pipelineTuningOptions.LoopUnrolling);
  target.pipelineTuningOptions.SLPVectorization = getBool(
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

  target.ukernels = getString("ukernels", target.ukernels, false);
  target.linkUkernelBitcode =
      getBool("link_ukernel_bitcode", target.linkUkernelBitcode);

  if (hasFailures) {
    return {};
  }
  target.populateDefaultsFromTargetMachine();
  return target;
}

void LLVMTarget::populateDefaultsFromTargetMachine() {
  // We may need the target machine for certain default values.
  std::unique_ptr<llvm::TargetMachine> cachedTargetMachine;
  auto getTargetMachine = [&]() {
    if (!cachedTargetMachine) {
      cachedTargetMachine = createTargetMachine(*this);
      // TODO(#13988): proper error propagation. This is a common user scenario.
      assert(cachedTargetMachine && "createTargetMachine failed");
    }
    return cachedTargetMachine.get();
  };

  if (dataLayout.empty()) {
    auto targetDataLayout = getTargetMachine()->createDataLayout();
    dataLayout = targetDataLayout.getStringRepresentation();
  }

  if (vectorWidthInBytes == DEFAULT_VECTOR_WIDTH_IN_BYTES) {
    auto targetMachine = getTargetMachine();
    auto targetFeatures = targetMachine->getTargetFeatureString();

    // The only way to get the real TTI is to create a function using it.
    // LLVM's TargetMachine and related APIs are terrible. Absolutely yuck.
    // Note that we use the data layout set above to either what the user
    // specified or what the target machine returned.
    //
    // If anyone comes across this: it'd be great if getTargetTransformInfo
    // could be called without requiring a function.
    llvm::LLVMContext llvmContext;
    auto llvmModule =
        std::make_unique<llvm::Module>("dummy_module", llvmContext);
    llvmModule->setDataLayout(dataLayout);
    llvm::Function *dummyFunc = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), false),
        llvm::GlobalValue::ExternalLinkage, "dummy_func", *llvmModule);
    if (targetFeatures.contains("avx512")) {
      // Always override the vector with to 512 on systems with avx512.
      // @dcaballe says:
      // > in ML the frequency throttling that happens when using 512-bit
      // > register doesn't have an overall negative impact in performance due
      // > to the high computational density of the workloads, even on skylake
      // > where the throttling was really bad
      dummyFunc->addFnAttr("prefer-vector-width", "512");
    }
    auto targetTTI = targetMachine->getTargetTransformInfo(*dummyFunc);

    // Query the vector width from TTI.
    unsigned ttiVectorWidthInBytes =
        targetTTI.getRegisterBitWidth(
            llvm::TargetTransformInfo::RGK_FixedWidthVector) /
        8;
    vectorWidthInBytes = ttiVectorWidthInBytes > 1 ? ttiVectorWidthInBytes : 16;
  }
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
  targetOptions.target.populateDefaultsFromTargetMachine();
  return targetOptions;
}

// static
void LLVMTargetOptions::initializeFromFlags(LLVMTargetOptions &targetOptions) {
  targetOptions.initializeTargetInvariantFlags();

  // Target parameters.
  static llvm::cl::opt<std::string> clTargetTriple(
      "iree-llvmcpu-target-triple",
      llvm::cl::desc("LLVM target machine triple."),
      llvm::cl::init(llvm::sys::getProcessTriple()));
  static llvm::cl::opt<std::string> clTargetCPU(
      "iree-llvmcpu-target-cpu",
      llvm::cl::desc(
          "LLVM target machine CPU; use 'host' for your host native CPU."),
      llvm::cl::init("generic"));
  static llvm::cl::opt<std::string> clTargetCPUFeatures(
      "iree-llvmcpu-target-cpu-features",
      llvm::cl::desc("LLVM target machine CPU features; use 'host' for your "
                     "host native CPU."),
      llvm::cl::init(""));
  static llvm::cl::opt<bool> clLinkEmbedded(
      "iree-llvmcpu-link-embedded",
      llvm::cl::desc("Links binaries into a platform-agnostic ELF to be loaded "
                     "by the embedded IREE ELF loader."),
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

  static llvm::cl::opt<std::string> clTargetDataLayout(
      "iree-llvmcpu-target-data-layout",
      llvm::cl::desc("LLVM target machine data layout override."),
      llvm::cl::init(""));
  target.dataLayout = clTargetDataLayout;
  static llvm::cl::opt<unsigned> clTargetVectorWidthInBytes(
      "iree-llvmcpu-target-vector-width-in-bytes",
      llvm::cl::desc("Overrides the native vector register width (in bytes) of "
                     "the target."),
      llvm::cl::init(0));
  target.vectorWidthInBytes = clTargetVectorWidthInBytes;

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

  static llvm::cl::opt<std::string> clEnableUkernels(
      "iree-llvmcpu-enable-ukernels",
      llvm::cl::desc("Enables ukernels in the llvmcpu backend. May be "
                     "`default`, `none`, `all`, or a comma-separated list of "
                     "specific unprefixed ukernels to enable, e.g. `mmt4d`."),
      llvm::cl::init("default"));
  target.ukernels = clEnableUkernels;
  static llvm::cl::opt<bool> clLinkUKernelBitcode(
      "iree-llvmcpu-link-ukernel-bitcode",
      llvm::cl::desc(
          "Link ukernel bitcode libraries into generated executables"),
      llvm::cl::init(target.linkUkernelBitcode));
  target.linkUkernelBitcode = clLinkUKernelBitcode;

  static llvm::cl::opt<bool> clListTargets(
      "iree-llvmcpu-list-targets",
      llvm::cl::desc("Lists all registered targets that the LLVM backend can "
                     "generate code for."),
      llvm::cl::init(false), llvm::cl::ValueDisallowed,
      llvm::cl::callback([&](const bool &) {
        llvm::TargetRegistry::printRegisteredTargetsForVersion(llvm::outs());
        exit(0);
      }));
}

// static
void LLVMTargetOptions::registerFlags() {
  LLVMTargetOptions targetOptions;
  initializeFromFlags(targetOptions);
}

// static
LLVMTargetOptions LLVMTargetOptions::getFromFlags() {
  LLVMTargetOptions targetOptions;
  initializeFromFlags(targetOptions);
  targetOptions.target.populateDefaultsFromTargetMachine();
  return targetOptions;
}

std::unique_ptr<llvm::TargetMachine>
createTargetMachine(const LLVMTarget &target) {
  std::string errorMessage;
  auto llvmTarget =
      llvm::TargetRegistry::lookupTarget(target.getTriple(), errorMessage);
  if (!llvmTarget)
    return nullptr;
  std::unique_ptr<llvm::TargetMachine> machine(llvmTarget->createTargetMachine(
      target.getTriple(), target.getCpu() /* cpu e.g k8 */,
      target.getCpuFeatures() /* cpu features e.g avx512f */,
      target.llvmTargetOptions, llvm::Reloc::Model::PIC_, {},
      target.codeGenOptLevel,
      /*JIT=*/false));
  return machine;
}

} // namespace mlir::iree_compiler::IREE::HAL
