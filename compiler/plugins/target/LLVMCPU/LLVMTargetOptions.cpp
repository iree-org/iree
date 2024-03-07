// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"

#include <mutex>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
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

static void initializeLLVMTargets() {
// Dynamically do preprocessor dispatch to initialize only targets that we
// care about if they are enabled. Unfortunately, the way the LLVM macros
// for this are set up and the inability to do a conditional within a macro
// means that we have to syntactically have a macro for every possible
// target we care about. There are more robust ways to do this but they all
// require build support, which is a pain to manage across platforms.
//
// See comments below.
#define LLVM_INITIALIZE_GENERIC(TargetName)                                    \
  LLVMInitialize##TargetName##Target();                                        \
  LLVMInitialize##TargetName##TargetMC();                                      \
  LLVMInitialize##TargetName##TargetInfo();                                    \
  LLVMInitialize##TargetName##AsmPrinter();                                    \
  LLVMInitialize##TargetName##AsmParser();

// CPU targets that we care about and have hard-linked against are here.
// They delegate to the generic initialize above. These must all be added
// to the build file or you will get undefined symbol errors at link time.
#define LLVM_INITIALIZE_TARGET_AArch64() LLVM_INITIALIZE_GENERIC(AArch64)
#define LLVM_INITIALIZE_TARGET_ARM() LLVM_INITIALIZE_GENERIC(ARM)
#define LLVM_INITIALIZE_TARGET_RISCV() LLVM_INITIALIZE_GENERIC(RISCV)
#define LLVM_INITIALIZE_TARGET_X86() LLVM_INITIALIZE_GENERIC(X86)
#define LLVM_INITIALIZE_TARGET_WebAssembly()                                   \
  LLVM_INITIALIZE_GENERIC(WebAssembly)

// We must no-op the name of each target we don't care about. This is annoying,
// but targets aren't created every day and isn't the end of the world. The
// error messages when missing are quite clear and you just add a line here.
#define LLVM_INITIALIZE_TARGET_AMDGPU()
#define LLVM_INITIALIZE_TARGET_AVR()
#define LLVM_INITIALIZE_TARGET_BPF()
#define LLVM_INITIALIZE_TARGET_Hexagon()
#define LLVM_INITIALIZE_TARGET_Lanai()
#define LLVM_INITIALIZE_TARGET_LoongArch()
#define LLVM_INITIALIZE_TARGET_Mips()
#define LLVM_INITIALIZE_TARGET_MSP430()
#define LLVM_INITIALIZE_TARGET_NVPTX()
#define LLVM_INITIALIZE_TARGET_PowerPC()
#define LLVM_INITIALIZE_TARGET_Sparc()
#define LLVM_INITIALIZE_TARGET_SystemZ()
#define LLVM_INITIALIZE_TARGET_VE()
#define LLVM_INITIALIZE_TARGET_XCore()

#define LLVM_TARGET(TargetName) LLVM_INITIALIZE_TARGET_##TargetName()
#include "llvm/Config/Targets.def"
}

//===----------------------------------------------------------------------===//
//    __    __   ___________    ____     ____    ____  ______    __    __     //
//   |  |  |  | |   ____\   \  /   /     \   \  /   / /  __  \  |  |  |  |    //
//   |  |__|  | |  |__   \   \/   /       \   \/   / |  |  |  | |  |  |  |    //
//   |   __   | |   __|   \_    _/         \_    _/  |  |  |  | |  |  |  |    //
//   |  |  |  | |  |____    |  |   __        |  |    |  `--'  | |  `--'  |    //
//   |__|  |__| |_______|   |__|  (_ )       |__|     \______/   \______/     //
//                                 |/                                         //
//===----------------------------------------------------------------------===//
//
// Beware adding command-line flags here: IREE is a cross-compiler and can
// compile for multiple targets in a single invocation. Global flags added here
// apply to all targets with no way to override them from hosting applications
// that may need to programmatically set them per target and that's bad.
//
// Flags *must* be added to the LLVMTarget if they are target-specific and
// LLVMTargetOptions if they are apply to the whole backend.

void LLVMCPUTargetCLOptions::bindOptions(OptionsBinder &binder) {
  // Initialize LLVM targets prior to the iree-llvmcpu-list-targets CLI opt.
  initializeLLVMTargets();

  static llvm::cl::OptionCategory category("LLVMCPU HAL Target");

  // General flags.
  binder.opt<bool>(
      "iree-llvmcpu-list-targets", listTargets, llvm::cl::cat(category),
      llvm::cl::desc("Lists all registered targets that the LLVM backend can "
                     "generate code for."),
      llvm::cl::ValueDisallowed, llvm::cl::callback([&](const bool &) {
        llvm::TargetRegistry::printRegisteredTargetsForVersion(llvm::outs());
        exit(0);
      }));

  // Target invariant flags.
  binder.opt<std::string>(
      "iree-llvmcpu-system-linker-path", systemLinkerPath,
      llvm::cl::cat(category),
      llvm::cl::desc("Tool used to link system shared libraries produced by "
                     "IREE (for --iree-llvmcpu-link-embedded=false)."));
  binder.opt<std::string>(
      "iree-llvmcpu-embedded-linker-path", embeddedLinkerPath,
      llvm::cl::cat(category),
      llvm::cl::desc("Tool used to link embedded ELFs produced by IREE (for "
                     "--iree-llvmcpu-link-embedded=true)."));
  binder.opt<std::string>(
      "iree-llvmcpu-wasm-linker-path", wasmLinkerPath, llvm::cl::cat(category),
      llvm::cl::desc("Tool used to link WebAssembly modules produced by "
                     "IREE (for --iree-llvmcpu-target-triple=wasm32-*)."));
  binder.opt<bool>(
      "iree-llvmcpu-keep-linker-artifacts", keepLinkerArtifacts,
      llvm::cl::cat(category),
      llvm::cl::desc("Keep LLVM linker target artifacts (.so/.dll/etc)"));

  // Default device options.
  binder.opt<std::string>("iree-llvmcpu-target-triple", targetTriple,
                          llvm::cl::cat(category),
                          llvm::cl::desc("LLVM target machine triple."));
  binder.opt<std::string>(
      "iree-llvmcpu-target-cpu", targetCPU, llvm::cl::cat(category),
      llvm::cl::desc(
          "LLVM target machine CPU; use 'host' for your host native CPU."));
  binder.opt<std::string>(
      "iree-llvmcpu-target-cpu-features", targetCPUFeatures,
      llvm::cl::cat(category),
      llvm::cl::desc("LLVM target machine CPU features; use 'host' for your "
                     "host native CPU."));
  binder.opt<bool>(
      "iree-llvmcpu-link-embedded", linkEmbedded, llvm::cl::cat(category),
      llvm::cl::desc("Links binaries into a platform-agnostic ELF to be "
                     "loaded by the embedded IREE ELF loader."));
  binder.opt<bool>(
      "iree-llvmcpu-link-static", linkStatic, llvm::cl::cat(category),
      llvm::cl::desc(
          "Links system libraries into binaries statically to isolate them "
          "from platform dependencies needed at runtime"));
  binder.opt<std::string>(
      "iree-llvmcpu-static-library-output-path", staticLibraryOutputPath,
      llvm::cl::cat(category),
      llvm::cl::desc(
          "Path to output static object (EX: '/path/to/static-library.o'). "
          "This will produce the static library at the specified path along "
          "with a similarly named '.h' file for static linking."));
  binder.opt<bool>(
      "iree-llvmcpu-debug-symbols", debugSymbols, llvm::cl::cat(category),
      llvm::cl::desc("Generate and embed debug information (DWARF, PDB, etc)"));
  binder.opt<bool>("iree-llvmcpu-loop-interleaving", llvmLoopInterleaving,
                   llvm::cl::cat(category),
                   llvm::cl::desc("Enable LLVM loop interleaving opt"));
  binder.opt<bool>("iree-llvmcpu-loop-vectorization", llvmLoopVectorization,
                   llvm::cl::cat(category),
                   llvm::cl::desc("Enable LLVM loop vectorization opt"));
  binder.opt<bool>("iree-llvmcpu-loop-unrolling", llvmLoopUnrolling,
                   llvm::cl::cat(category),
                   llvm::cl::desc("Enable LLVM loop unrolling opt"));
  binder.opt<bool>("iree-llvmcpu-slp-vectorization", llvmSLPVectorization,
                   llvm::cl::cat(category),
                   llvm::cl::desc("Enable LLVM SLP Vectorization opt"));
  binder.opt<SanitizerKind>(
      "iree-llvmcpu-sanitize", sanitizerKind, llvm::cl::cat(category),
      llvm::cl::desc("Apply LLVM sanitize feature"),
      llvm::cl::values(clEnumValN(SanitizerKind::kAddress, "address",
                                  "Address sanitizer support"),
                       clEnumValN(SanitizerKind::kThread, "thread",
                                  "Thread sanitizer support")));
  binder.opt<std::string>(
      "iree-llvmcpu-target-abi", targetABI, llvm::cl::cat(category),
      llvm::cl::desc("LLVM target machine ABI; specify for -mabi"));
  binder.opt<llvm::FloatABI::ABIType>(
      "iree-llvmcpu-target-float-abi", targetFloatABI, llvm::cl::cat(category),
      llvm::cl::desc("LLVM target codegen enables soft float abi e.g "
                     "-mfloat-abi=softfp"),
      llvm::cl::values(
          clEnumValN(llvm::FloatABI::Default, "default", "Default (softfp)"),
          clEnumValN(llvm::FloatABI::Soft, "soft",
                     "Software floating-point emulation"),
          clEnumValN(llvm::FloatABI::Hard, "hard",
                     "Hardware floating-point instructions")));
  binder.opt<std::string>(
      "iree-llvmcpu-target-data-layout", targetDataLayout,
      llvm::cl::cat(category),
      llvm::cl::desc("LLVM target machine data layout override."));
  binder.opt<unsigned>("iree-llvmcpu-target-vector-width-in-bytes",
                       targetVectorWidthInBytes, llvm::cl::cat(category),
                       llvm::cl::desc("Overrides the native vector register "
                                      "width (in bytes) of the target."));
  binder.opt<std::string>(
      "iree-llvmcpu-enable-ukernels", enableUkernels, llvm::cl::cat(category),
      llvm::cl::desc("Enables ukernels in the llvmcpu backend. May be "
                     "`default`, `none`, `all`, or a comma-separated list of "
                     "specific unprefixed ukernels to enable, e.g. `mmt4d`."));
  binder.opt<bool>(
      "iree-llvmcpu-link-ukernel-bitcode", linkUKernelBitcode,
      llvm::cl::cat(category),
      llvm::cl::desc(
          "Link ukernel bitcode libraries into generated executables"));
}

LLVMTargetOptions LLVMCPUTargetCLOptions::getTargetOptions() {
  LLVMTargetOptions targetOptions;
  targetOptions.systemLinkerPath = systemLinkerPath;
  targetOptions.embeddedLinkerPath = embeddedLinkerPath;
  targetOptions.wasmLinkerPath = wasmLinkerPath;
  targetOptions.keepLinkerArtifacts = keepLinkerArtifacts;

  if (targetTriple.empty()) {
    targetTriple = llvm::sys::getProcessTriple();
  }

  std::optional<LLVMTarget> maybeTarget = LLVMTarget::create(
      targetTriple, targetCPU, targetCPUFeatures, linkEmbedded);
  if (maybeTarget) {
    targetOptions.target = *maybeTarget;
  } else {
    llvm::errs() << "Inconsistency in iree-llvmcpu-target-cpu-* command-line"
                    "flags. The target CPU is not properly defined.\n";
  }
  LLVMTarget &target = targetOptions.target;
  target.linkStatic = linkStatic;
  target.staticLibraryOutput = staticLibraryOutputPath;
  target.debugSymbols = debugSymbols;
  target.pipelineTuningOptions.LoopInterleaving = llvmLoopInterleaving;
  target.pipelineTuningOptions.LoopVectorization = llvmLoopVectorization;
  target.pipelineTuningOptions.LoopUnrolling = llvmLoopUnrolling;
  target.pipelineTuningOptions.SLPVectorization = llvmSLPVectorization;
  target.sanitizerKind = sanitizerKind;
  target.llvmTargetOptions.MCOptions.ABIName = targetABI;
  target.llvmTargetOptions.FloatABIType = targetFloatABI;
  target.dataLayout = targetDataLayout;
  target.vectorWidthInBytes = targetVectorWidthInBytes;
  target.ukernels = enableUkernels;
  target.linkUkernelBitcode = linkUKernelBitcode;

  target.populateDefaultsFromTargetMachine();
  return targetOptions;
}

} // namespace mlir::iree_compiler::IREE::HAL
