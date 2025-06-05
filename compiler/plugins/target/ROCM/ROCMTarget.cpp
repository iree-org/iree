// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ROCMTargetUtils.h"

#include <cstdint>

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.h"
#include "compiler/plugins/target/ROCM/builtins/ukernel/iree_uk_amdgpu_bitcode.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMLinkerUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/amdgpu_executable_def_builder.h"
#include "iree/schemas/hip_executable_def_builder.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

enum class ContainerType {
  // Automatically detect the container type from the target ABI attribute.
  Auto,
  // HIP ExecutableDef flatbuffer.
  HIP,
  // AMDGPU ExecutableDef flatbuffer.
  AMDGPU,
  // Raw HSACO image (ELF).
  HSACO,
};

// TODO(#18792): rename flags back to iree-rocm- as they are not HIP-specific.
struct ROCMOptions {
  std::string target = "";
  std::string targetFeatures = "";
  ContainerType containerType = ContainerType::Auto;
  std::string bitcodeDirectory = getDefaultBitcodeDirectory();
  int wavesPerEu = 0;
  std::string enableROCMUkernels = "none";
  std::string encodingLayoutResolver = GPU::kNoEncodingLayoutResolverName;
  bool slpVectorization = true;
  bool globalISel = false;

  /// List of LLVM opt pass pluggins to be loaded during GPU code
  /// generation. The pluggins are paths to dynamic libraries that
  /// are added to the LLVM pass manager.
  SmallVector<std::string> passPlugins;

  void bindOptions(OptionsBinder &binder) {
    using namespace llvm;
    static cl::OptionCategory category("HIP HAL Target");

    binder.opt<std::string>(
        "iree-hip-target", target, cl::cat(category),
        cl::desc(
            // clang-format off
            "HIP target as expected by LLVM AMDGPU backend; e.g., "
            "'gfx90a'/'gfx942' for targeting MI250/MI300 GPUs. "
            "Additionally this also supports architecture code names like "
            "'cdna3'/'rdna3' or some product names like 'mi300x'/'rtx7900xtx' "
            "for a better experience. See "
            "https://iree.dev/guides/deployment-configurations/gpu-rocm/ "
            "for more details."
            // clang-format on
            ));

    binder.opt<std::string>(
        "iree-hip-target-features", targetFeatures, cl::cat(category),
        cl::desc("HIP target features as expected by LLVM AMDGPU backend; "
                 "e.g., '+sramecc,+xnack'."));

    binder.opt<ContainerType>(
        "iree-rocm-container-type", containerType,
        llvm::cl::desc("Serialized executable container type."),
        llvm::cl::cat(category),
        llvm::cl::values(clEnumValN(ContainerType::Auto, "auto",
                                    "Automatically detect the container type "
                                    "from the target ABI attribute."),
                         clEnumValN(ContainerType::HIP, "hip",
                                    "HIP ExecutableDef flatbuffer."),
                         clEnumValN(ContainerType::AMDGPU, "amdgpu",
                                    "AMDGPU ExecutableDef flatbuffer."),
                         clEnumValN(ContainerType::HSACO, "hsaco",
                                    "Raw HSACO image (ELF).")));

    binder.opt<std::string>("iree-hip-bc-dir", bitcodeDirectory,
                            cl::cat(category),
                            cl::desc("Directory of HIP Bitcode."));

    binder.opt<int>("iree-hip-waves-per-eu", wavesPerEu, cl::cat(category),
                    cl::desc("Optimization hint specifying minimum "
                             "number of waves per execution unit."));

    binder.opt<std::string>(
        "iree-hip-enable-ukernels", enableROCMUkernels, cl::cat(category),
        cl::desc("Enables microkernels in the HIP compiler backend. May be "
                 "`default`, `none`, `all`, or a comma-separated list of "
                 "specific unprefixed microkernels to enable, e.g. `mmt4d`."));
    binder.opt<std::string>(
        "iree-hip-encoding-layout-resolver", encodingLayoutResolver,
        cl::cat(category),
        cl::desc("Selects the way that encodings will be "
                 "resolved. Options are: `none` (resolve to "
                 "identity layout), `pad` (additional padding "
                 "on allocations to maximize cache bandwidth), "
                 "and `data-tiling` (enable data tiled layouts)"));

    binder.list<std::string>(
        "iree-hip-pass-plugin-path", passPlugins,
        cl::desc("LLVM pass plugins are out of tree libraries that implement "
                 "LLVM opt passes. The library paths passed in this flag are "
                 "to be passed to the target backend compiler during HIP "
                 "executable serialization"),
        cl::ZeroOrMore, cl::cat(category));

    binder.opt<bool>("iree-hip-llvm-slp-vec", slpVectorization,
                     cl::cat(category),
                     cl::desc("Enable slp vectorization in llvm opt."));
    binder.opt<bool>("iree-hip-llvm-global-isel", globalISel, cl::cat(category),
                     cl::desc("Enable global instruction selection in llvm."));
  }

  LogicalResult verify(mlir::Builder &builder) const {
    if (target.empty()) {
      return emitError(builder.getUnknownLoc())
             << "HIP target not set; did you forget to pass "
                "'--iree-hip-target'?";
    }
    if (GPU::normalizeHIPTarget(target).empty()) {
      return emitError(builder.getUnknownLoc(), "Unknown HIP target '")
             << target << "'";
    }
    SmallVector<StringRef> features;
    llvm::SplitString(targetFeatures, features, ",");
    for (StringRef f : features) {
      if (!(f.starts_with("+") || f.starts_with("-"))) {
        return emitError(builder.getUnknownLoc(),
                         "HIP target feature must be prefixed with '+' or "
                         "'-'; but seen '")
               << f << "'";
      }
      StringRef feature = f.substr(1);
      if (feature != "sramecc" && feature != "xnack") {
        // We only support these two features to be set explicitly. Features
        // like wavefrontsize is controlled and tuned by the compiler.
        return emitError(builder.getUnknownLoc(),
                         "HIP target feature can only be 'sramecc' or "
                         "'xnack'; but seen '")
               << feature << "'";
      }
    }
    return success();
  }

private:
  static std::string getDefaultBitcodeDirectory() {
    return mlir::iree_compiler::findPlatformLibDirectory("rocm");
  }
};

// Returns the ABI or an empty string if unspecified.
static StringRef getABI(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (targetAttr) {
    if (auto config = targetAttr.getConfiguration()) {
      auto abiAttr = targetAttr.getConfiguration().getAs<StringAttr>("abi");
      return abiAttr ? abiAttr.getValue() : "";
    }
  }
  return "";
}

static void dumpModuleToPath(StringRef path, StringRef baseName,
                             StringRef suffix, StringRef extension,
                             llvm::Module &module, StringRef header = {}) {
  llvm::SmallVector<char, 0> data;
  llvm::raw_svector_ostream ostream(data);
  ostream << header;
  module.print(ostream, nullptr);
  dumpDataToPath(path, baseName, suffix, extension,
                 StringRef(data.data(), data.size()));
}

static std::string translateModuleToObj(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetObj;
  {
    llvm::raw_string_ostream stream(targetObj);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);
    codegenPasses.run(module);
  }
  return targetObj;
}

static std::string translateModuleToISA(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CodeGenFileType::AssemblyFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

} // namespace

class ROCMTargetBackend final : public TargetBackend {
public:
  ROCMTargetBackend(const ROCMOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "hip"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    if (auto target = getExecutableTarget(deviceID, context)) {
      executableTargetAttrs.push_back(target);
    }
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(StringRef deviceID, MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute, 4> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(name, value);
    };

    if (failed(options.verify(b))) {
      return nullptr;
    }

    addConfig("abi", b.getStringAttr(deviceID));
    std::string format;
    if (deviceID == "amdgpu") {
      format = options.target;
    } else {
      format = "rocm-hsaco-fb"; // legacy HIP
    }

    if (auto target = GPU::getHIPTargetDetails(
            options.target, options.targetFeatures, context)) {
      addConfig(kGPUTargetAttrName, target);
      if (options.encodingLayoutResolver !=
          GPU::kNoEncodingLayoutResolverName) {
        if (Attribute encoding = GPU::getHIPTargetEncodingLayoutAttr(
                target, options.encodingLayoutResolver)) {
          addConfig(IREE::Encoding::kEncodingResolverAttrName, encoding);
        }
      }

      // Look for a default tuning spec.
      auto rocmDialect = context->getOrLoadDialect<IREE::ROCM::ROCMDialect>();
      // First check for a spec based on the sku.
      std::optional<std::string> maybeSpecName = std::nullopt;
      if (IREE::GPU::TargetChipAttr chip = target.getChip()) {
        if (StringAttr sku = chip.getSku()) {
          std::string specName =
              llvm::formatv("iree_default_tuning_spec_{}.mlir", sku.strref());
          if (!rocmDialect->hasBuiltin(specName)) {
            maybeSpecName = specName;
          }
        }
      }

      // Then, if none was found, look for one based on the target arch.
      if (!maybeSpecName) {
        std::string specName =
            llvm::formatv("iree_default_tuning_spec_{}.mlir", target.getArch());
        if (rocmDialect->hasBuiltin(specName)) {
          maybeSpecName = specName;
        }
      }

      if (maybeSpecName) {
        addConfig("iree_codegen.default_tuning_spec",
                  IREE::ROCM::BuiltinTuningModuleAttr::get(
                      context, maybeSpecName.value()));
      }
    }

    addConfig("ukernels", b.getStringAttr(options.enableROCMUkernels));
    if (options.wavesPerEu > 0) {
      addConfig("waves_per_eu", b.getI64IntegerAttr(options.wavesPerEu));
    }

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("rocm"), b.getStringAttr(format),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerROCDLDialectTranslation(registry);
    registry.insert<IREE::Codegen::IREECodegenDialect>();
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<IREE::GPU::IREEGPUDialect>();
    registry.insert<IREE::ROCM::ROCMDialect>();
    // Configuration may load and manipulate transform dialect libraries.
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) override {
    buildLLVMGPUCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    buildLLVMGPUCodegenPassPipeline(passManager, true);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    buildLLVMGPULinkingPassPipeline(passManager, "rocm");
  }

  // Performs optimizations on |module| (including LTO-style whole-program
  // ones). Inspired by code section in
  // https://github.com/iree-org/iree/blob/main/compiler/plugins/target/CUDA/CUDATarget.cpp
  static void optimizeModule(llvm::Module &module,
                             llvm::TargetMachine &targetMachine,
                             ArrayRef<std::string> passPlugins,
                             bool slpVectorization,
                             std::string &outPassesString) {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    fam.registerPass([&] { return targetMachine.getTargetIRAnalysis(); });

    llvm::PipelineTuningOptions pto;
    pto.SLPVectorization = slpVectorization;

    llvm::PassInstrumentationCallbacks pic;

    llvm::StandardInstrumentations si(module.getContext(), false);
    si.registerCallbacks(pic, &mam);

    llvm::PassBuilder pb(&targetMachine, pto, std::nullopt, &pic);
    llvm::ModulePassManager mpm;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    for (const std::string &pluginFileName : passPlugins) {
      llvm::Expected<llvm::PassPlugin> pp =
          llvm::PassPlugin::Load(pluginFileName);
      if (pp) {
        pp->registerPassBuilderCallbacks(pb);
      } else {
        std::string error = "unable to load plugin " + pluginFileName + ": " +
                            llvm::toString(pp.takeError());
        llvm::report_fatal_error(error.c_str());
      }
    }

    llvm::OptimizationLevel ol = llvm::OptimizationLevel::O2;

    mpm.addPass(llvm::VerifierPass());
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
    mpm.addPass(llvm::VerifierPass());
    llvm::raw_string_ostream os(outPassesString);
    mpm.printPipeline(os, [&pic](StringRef className) {
      auto passName = pic.getPassNameForClassName(className);
      return passName.empty() ? className : passName;
    });
    mpm.run(module, mam);
  }

  LogicalResult
  validateFinalizedModule(IREE::HAL::ExecutableVariantOp variantOp,
                          llvm::Module &module) {
    for (llvm::Function &func : module.functions()) {
      if (func.isDeclaration() && !func.isIntrinsic() && !func.use_empty()) {
        llvm::User *liveUser = *func.user_begin();
        return variantOp.emitError()
               << "found an unresolved external function '" << func.getName()
               << "' in the final bitcode. A remaining live user is\n"
               << llvm::formatv("{}", *liveUser);
      }
    }
    return success();
  }

  LogicalResult
  serializeExecutable(const SerializationOptions &serializationOptions,
                      IREE::HAL::ExecutableVariantOp variantOp,
                      OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto targetAttr = variantOp.getTargetAttr();
    StringRef targetArch = options.target;
    StringRef targetFeatures = options.targetFeatures;
    if (auto attr = getGPUTargetAttr(targetAttr)) {
      targetArch = attr.getArch();
      targetFeatures = attr.getFeatures();
    }

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    const std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // Collect all the entry point names.
    auto exportOps = llvm::to_vector_of<IREE::HAL::ExecutableExportOp>(
        variantOp.getExportOps());
    std::optional<uint32_t> subgroupSize;
    for (IREE::HAL::ExecutableExportOp exportOp : exportOps) {
      // TODO: put this either on the variant or propagate as a function
      // attribute instead - today this *must* be consistent across all exports
      // and it shouldn't need to be.
      if (auto setSubgroupSize = exportOp.getSubgroupSizeAsUInt()) {
        if (setSubgroupSize.value() != 32 && setSubgroupSize.value() != 64) {
          return variantOp.emitError()
                 << "invalid subgroup size " << setSubgroupSize.value();
        }
        if (subgroupSize.has_value() &&
            setSubgroupSize.value() != subgroupSize.value()) {
          return variantOp.emitError()
                 << "multiple exports with different subgroup sizes; this is a "
                    "limitation of the IREE compilation process and should be "
                    "fixed";
        }
        subgroupSize = setSubgroupSize.value();
      }
    }

    std::string targetHSACO;
    if (variantOp.isExternal()) {
      if (!variantOp.getObjects().has_value()) {
        return variantOp.emitOpError()
               << "no objects defined for external variant";
      }

      if (variantOp.getObjects()->getValue().size() != 1) {
        // For now we assume there will be exactly one object file.
        // In the future we will want to perform a linking step here and ideally
        // support _also_ linking in the codegen results.
        return variantOp.emitOpError() << "only one object reference is "
                                          "supported for external variants";
      }

      // Read the HSACO from the object file.
      auto objectAttr = llvm::cast<IREE::HAL::ExecutableObjectAttr>(
          variantOp.getObjects()->getValue().front());
      if (auto data = objectAttr.loadData()) {
        targetHSACO = data.value();
      } else {
        return variantOp.emitOpError()
               << "object file could not be loaded: " << objectAttr;
      }
    } else {
      auto maybeChipset = amdgpu::Chipset::parse(targetArch);
      if (failed(maybeChipset)) {
        return variantOp.emitOpError()
               << "could not parse AMDGPU chipset name '" << targetArch << "'";
      }
      amdgpu::Chipset chipset = *maybeChipset;
      // Perform the translation in a separate context to avoid any
      // multi-threading issues.
      llvm::LLVMContext context;
      std::unique_ptr<llvm::Module> llvmModule =
          mlir::translateModuleToLLVMIR(innerModuleOp, context, libraryName);
      if (!llvmModule) {
        return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                        "dialect to the native llvm::Module";
      }

      for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
        llvm::Function *llvmFunc = llvmModule->getFunction(func.getName());
        if (llvmFunc->isDeclaration())
          continue;

        // Override flags as given by target func attrs.
        if (auto funcAttrs =
                func->getAttrOfType<DictionaryAttr>("llvm_func_attrs")) {
          for (NamedAttribute funcAttr : funcAttrs) {
            auto value = dyn_cast<StringAttr>(funcAttr.getValue());
            if (!value) {
              return variantOp->emitError()
                     << "llvm_func_attrs attribute must be a dictionary of "
                        "strings. Attribute "
                     << funcAttr.getName() << " is not a StringAttr.";
            }
            llvmFunc->addFnAttr(funcAttr.getName(), value.getValue());
          }
        }
      }

      std::unique_ptr<llvm::TargetMachine> targetMachine;
      bool isWave64 = true;
      {
        llvm::Triple triple("amdgcn-amd-amdhsa");
        std::string error;
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget("", triple, error);
        if (!target) {
          return variantOp.emitError() << "cannot initialize target triple";
        }
        llvm::TargetOptions opt;
        opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
        opt.UnsafeFPMath = false;
        opt.NoInfsFPMath = false;
        opt.NoNaNsFPMath = true;
        // Be extra cautious while this is less tested, and prevent unknown
        // fallbacks from global isel.
        //
        // When GlobalISelAbort is set, any failure of GlobalISel,
        // whether due to being not yet implemented or incorrect IR will result
        // in an immediate abortion of compilation. This disables the fallback
        // path of AMDGPUPassConfig::addInstSelector and 2 legacy passes which
        // might work around unimplemented cases or errors in GlobalISel
        // resulting in a successful compilation but would make one assume
        // results are with GlobalISel when they are not.
        opt.EnableGlobalISel = options.globalISel;
        opt.GlobalISelAbort = options.globalISel
                                  ? llvm::GlobalISelAbortMode::Enable
                                  : llvm::GlobalISelAbortMode::Disable;
        SmallVector<std::string> features;
        if (chipset.majorVersion >= 10 && chipset.majorVersion <= 12) {
          switch (subgroupSize.value_or(64)) {
          case 32:
            isWave64 = false;
            features.emplace_back("+wavefrontsize32");
            break;
          default:
          case 64:
            features.emplace_back("+wavefrontsize64");
            break;
          }
        }

        // Mixed precision fma instructions have complicated semantics on
        // gf9+ GPUs and can lead to numeric issues as seen in
        // https://github.com/iree-org/iree/issues/18746 so we disable this
        // feature.
        if (targetArch.starts_with("gfx9")) {
          features.emplace_back("-fma-mix-insts");
        }

        if (!targetFeatures.empty()) {
          features.emplace_back(targetFeatures.str());
        }

        std::string featureStr = llvm::join(features, ",");

        targetMachine.reset(target->createTargetMachine(
            triple, targetArch, featureStr, opt, llvm::Reloc::Model::PIC_,
            std::nullopt, llvm::CodeGenOptLevel::Aggressive));

        if (!targetMachine) {
          return variantOp.emitError() << "cannot initialize target machine";
        }
      }

      llvmModule->setDataLayout(targetMachine->createDataLayout());

      // Code object version * 100.
      constexpr uint32_t abiVersion = 500;
      // Let the backend know what code object version we're compiling for. This
      // insulates us from changes to the default code object version that our
      // CI or users may not be prepared for.
      llvmModule->addModuleFlag(llvm::Module::Error,
                                "amdhsa_code_object_version", abiVersion);
      for (llvm::Function &f : llvmModule->functions())
        f.addFnAttr(llvm::Attribute::AlwaysInline);

      // Link user-provided modules.
      llvm::Linker linker(*llvmModule);
      if (failed(linkCmdlineBitcodeFiles(
              variantOp.getLoc(), linker, llvm::Linker::OverrideFromSrc,
              *targetMachine, llvmModule->getContext()))) {
        return failure();
      }

      // Link bitcode (*.bc) object attrs specified by the input program.
      // Note that this happens after the command-line files so that the command
      // line ones override the symbols coming from the embedded files.
      auto specializationCallback = [&](llvm::Module &userModule) {
        // TODO: inject __nvvm_reflect-style functions/globals for
        // bitcode specialization based on the targetMachine and configuration.
        // These could use any information we have on the IREE side as well as
        // the TargetMachine.
      };
      unsigned linkerFlags =
          llvm::Linker::LinkOnlyNeeded | llvm::Linker::OverrideFromSrc;
      if (failed(linkBitcodeObjects(variantOp.getLoc(), linker, linkerFlags,
                                    *targetMachine, variantOp.getObjectsAttr(),
                                    llvmModule->getContext(),
                                    specializationCallback))) {
        return mlir::emitError(variantOp.getLoc())
               << "failed linking in user objects for target triple '"
               << targetArch.str() << "'";
      }

      // Link module to HIP device library.
      if (options.bitcodeDirectory.empty()) {
        return variantOp.emitError()
               << "cannot find ROCM bitcode files. Check your installation "
                  "consistency and in the worst case, set "
                  "--iree-hip-bc-dir= to a path on your system.";
      }
      if (failed(linkHIPBitcodeIfNeeded(variantOp.getLoc(), llvmModule.get(),
                                        targetArch,
                                        options.bitcodeDirectory))) {
        return failure();
      }

      // Sets HIP platform globals based on the target architecture.
      if (failed(setHIPGlobals(variantOp.getLoc(), llvmModule.get(), chipset,
                               isWave64, abiVersion))) {
        return failure();
      }

      if (!serializationOptions.dumpIntermediatesPath.empty()) {
        dumpModuleToPath(serializationOptions.dumpIntermediatesPath,
                         serializationOptions.dumpBaseName, variantOp.getName(),
                         ".linked.ll", *llvmModule);
      }

      // For example 'gfx942'
      StringRef targetCPU = targetMachine->getTargetCPU();

      // For example 'amdgcn-amd-amdhsa'
      std::string targetTriple = targetMachine->getTargetTriple().str();

      // Run LLVM optimization passes.
      std::string passesString;
      optimizeModule(*llvmModule, *targetMachine, options.passPlugins,
                     options.slpVectorization, passesString);
      if (!serializationOptions.dumpIntermediatesPath.empty()) {

        // Additional context on '-mcpu' flag in PR comments, see for example:
        // https://github.com/iree-org/iree/pull/20716#issuecomment-2851650421
        std::string header =
            llvm::formatv(R"TXT(
; To reproduce the .optimized.ll from the .linked.ll, run:
; opt -S -mtriple={} -mcpu={} --passes='{}'
; The flag '-S' to emit LLVMIR.
; The behavior of some passes depends on '-mtriple' and '-mcpu'

)TXT",
                          targetTriple, targetCPU, passesString);

        dumpModuleToPath(serializationOptions.dumpIntermediatesPath,
                         serializationOptions.dumpBaseName, variantOp.getName(),
                         ".optimized.ll", *llvmModule, header);
      }

      if (failed(validateFinalizedModule(variantOp, *llvmModule))) {
        return failure();
      }

      // Dump the assembly output.
      if (!serializationOptions.dumpIntermediatesPath.empty()) {
        auto moduleCopy = llvm::CloneModule(*llvmModule);
        if (!moduleCopy) {
          llvm::errs() << "Error: cloning LLVM IR failed\n";
          return failure();
        }
        std::string targetISA =
            translateModuleToISA(*moduleCopy.get(), *targetMachine);
        dumpDataToPath(serializationOptions.dumpIntermediatesPath,
                       serializationOptions.dumpBaseName, variantOp.getName(),
                       ".rocmasm", targetISA);
      }

      // Serialize hsaco kernel into the binary that we will embed in the
      // final FlatBuffer.
      std::string targetObj = translateModuleToObj(*llvmModule, *targetMachine);
      targetHSACO = createHsaco(variantOp.getLoc(), targetObj, libraryName);
      if (targetHSACO.empty())
        return failure();
    }

    if (!serializationOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serializationOptions.dumpBinariesPath,
                     serializationOptions.dumpBaseName, variantOp.getName(),
                     ".hsaco", targetHSACO);
    }

    // Determine container type from the target ABI attribute.
    ContainerType containerType = options.containerType;
    if (containerType == ContainerType::Auto) {
      if (getABI(targetAttr) == "amdgpu") {
        containerType = ContainerType::AMDGPU;
      } else {
        containerType = ContainerType::HIP;
      }
    }

    // Wrap the HSACO ELF binary in the requested container type (if any).
    FailureOr<DenseIntElementsAttr> binaryContainer;
    switch (containerType) {
    case ContainerType::Auto: {
      // Resolved above; unreachable. Fall-through to error case.
      assert(false && "auto container type must have resolved earlier");
      break;
    }
    case ContainerType::AMDGPU: {
      binaryContainer = serializeAMDGPUBinaryContainer(
          serializationOptions, variantOp, exportOps, targetHSACO);
      break;
    }
    case ContainerType::HIP: {
      binaryContainer = serializeHIPBinaryContainer(
          serializationOptions, variantOp, exportOps, targetHSACO);
      break;
    }
    case ContainerType::HSACO: {
      SmallVector<uint8_t> image;
      image.resize(targetHSACO.size());
      std::memcpy(image.data(), targetHSACO.data(), image.size());
      binaryContainer = DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(targetHSACO.size())},
                          executableBuilder.getI8Type()),
          image);
      break;
    }
    }
    if (failed(binaryContainer) || !binaryContainer.value()) {
      return failure();
    }

    // Add the binary data to the target executable.
    executableBuilder.create<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(), binaryContainer.value());

    return success();
  }

protected:
  FailureOr<DenseIntElementsAttr> serializeAMDGPUBinaryContainer(
      const SerializationOptions &serializationOptions,
      IREE::HAL::ExecutableVariantOp variantOp,
      ArrayRef<IREE::HAL::ExecutableExportOp> exportOps,
      StringRef hsacoModule) {
    iree_compiler::FlatbufferBuilder builder;
    iree_hal_amdgpu_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serializationOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    // Only a single module today.
    SmallVector<iree_hal_amdgpu_ModuleDef_ref_t> moduleRefs;
    {
      auto hsacoImageRef = flatbuffers_string_create(
          builder, hsacoModule.data(), hsacoModule.size());
      moduleRefs.push_back(
          iree_hal_amdgpu_ModuleDef_create(builder, hsacoImageRef));
    }
    auto modulesRef = builder.createOffsetVecDestructive(moduleRefs);

    // Generate optional per-export debug information.
    // May be empty if no debug information was requested.
    auto exportDebugInfos =
        createExportDefs(serializationOptions.debugLevel, exportOps, builder);

    SmallVector<iree_hal_amdgpu_ExportDef_ref_t> exportRefs;
    exportRefs.resize(exportOps.size(), 0);
    for (auto exportOp : exportOps) {
      auto ordinalAttr = exportOp.getOrdinalAttr();
      if (!ordinalAttr) {
        return mlir::emitError(exportOp.getLoc())
               << "could not compile rocm binary: export op is missing ordinal";
      }
      int64_t ordinal = ordinalAttr.getInt();

      // Symbol names include a `.kd` suffix as that's what HSA expects.
      auto symbolNameKd = (exportOp.getName() + ".kd").str();
      auto symbolNameRef = builder.createString(symbolNameKd);

      iree_hal_amdgpu_Dims_t workgroupSize = {0};
      if (auto workgroupSizeAttr = exportOp.getWorkgroupSize()) {
        auto workgroupSizeDims = workgroupSizeAttr->getValue();
        workgroupSize.x = cast<IntegerAttr>(workgroupSizeDims[0]).getInt();
        workgroupSize.y = cast<IntegerAttr>(workgroupSizeDims[1]).getInt();
        workgroupSize.z = cast<IntegerAttr>(workgroupSizeDims[2]).getInt();
      }

      auto layoutAttr = exportOp.getLayoutAttr();
      uint32_t constantCount = static_cast<uint32_t>(layoutAttr.getConstants());
      SmallVector<iree_hal_amdgpu_BindingBits_enum_t> bindingFlags;
      for (auto bindingAttr : layoutAttr.getBindings()) {
        iree_hal_amdgpu_BindingBits_enum_t flags = 0;
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::ReadOnly)) {
          flags |= iree_hal_amdgpu_BindingBits_READ_ONLY;
        }
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::Indirect)) {
          flags |= iree_hal_amdgpu_BindingBits_INDIRECT;
        }
        bindingFlags.push_back(flags);
      }
      auto bindingFlagsRef = iree_hal_amdgpu_BindingBits_vec_create(
          builder, bindingFlags.data(), bindingFlags.size());

      iree_hal_amdgpu_ExportDef_start(builder);
      iree_hal_amdgpu_ExportDef_symbol_name_add(builder, symbolNameRef);
      iree_hal_amdgpu_ExportDef_workgroup_size_add(builder, &workgroupSize);
      iree_hal_amdgpu_ExportDef_constant_count_add(builder, constantCount);
      iree_hal_amdgpu_ExportDef_binding_flags_add(builder, bindingFlagsRef);
      iree_hal_amdgpu_ExportDef_debug_info_add(builder,
                                               exportDebugInfos[ordinal]);
      exportRefs[ordinal] = iree_hal_amdgpu_ExportDef_end(builder);
    }
    auto exportsRef = builder.createOffsetVecDestructive(exportRefs);

    iree_hal_amdgpu_ExecutableDef_exports_add(builder, exportsRef);
    iree_hal_amdgpu_ExecutableDef_modules_add(builder, modulesRef);
    iree_hal_amdgpu_ExecutableDef_source_files_add(builder, sourceFilesRef);
    iree_hal_amdgpu_ExecutableDef_end_as_root(builder);

    return builder.getBufferAttr(variantOp.getContext());
  }

  FailureOr<DenseIntElementsAttr>
  serializeHIPBinaryContainer(const SerializationOptions &serializationOptions,
                              IREE::HAL::ExecutableVariantOp variantOp,
                              ArrayRef<IREE::HAL::ExecutableExportOp> exportOps,
                              StringRef hsacoModule) {
    iree_compiler::FlatbufferBuilder builder;
    iree_hal_hip_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serializationOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    // Only a single module today.
    SmallVector<iree_hal_hip_ModuleDef_ref_t> moduleRefs;
    {
      auto hsacoImageRef = flatbuffers_string_create(
          builder, hsacoModule.data(), hsacoModule.size());
      moduleRefs.push_back(
          iree_hal_hip_ModuleDef_create(builder, hsacoImageRef));
    }
    auto modulesRef = builder.createOffsetVecDestructive(moduleRefs);

    // Generate optional per-export debug information.
    // May be empty if no debug information was requested.
    auto exportDebugInfos =
        createExportDefs(serializationOptions.debugLevel, exportOps, builder);

    SmallVector<iree_hal_hip_ExportDef_ref_t> exportRefs;
    exportRefs.resize(exportOps.size(), 0);
    for (auto exportOp : exportOps) {
      auto ordinalAttr = exportOp.getOrdinalAttr();
      if (!ordinalAttr) {
        return mlir::emitError(exportOp.getLoc())
               << "could not compile rocm binary: export op is missing ordinal";
      }
      int64_t ordinal = ordinalAttr.getInt();

      auto kernelNameRef = builder.createString(exportOp.getName());

      iree_hal_hip_BlockDims_t blockDims = {0};
      if (auto workgroupSizeAttr = exportOp.getWorkgroupSize()) {
        auto workgroupSize = workgroupSizeAttr->getValue();
        blockDims.x = cast<IntegerAttr>(workgroupSize[0]).getInt();
        blockDims.y = cast<IntegerAttr>(workgroupSize[1]).getInt();
        blockDims.z = cast<IntegerAttr>(workgroupSize[2]).getInt();
      }

      uint32_t blockSharedMemorySize = 0;
      if (std::optional<APInt> workgroupLocalMemoryAttr =
              exportOp.getWorkgroupLocalMemory()) {
        blockSharedMemorySize = workgroupLocalMemoryAttr->getSExtValue();
      }

      auto layoutAttr = exportOp.getLayoutAttr();
      uint32_t constantCount = static_cast<uint32_t>(layoutAttr.getConstants());
      SmallVector<iree_hal_hip_BindingBits_enum_t> bindingFlags;
      for (auto bindingAttr : layoutAttr.getBindings()) {
        iree_hal_hip_BindingBits_enum_t flags = 0;
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::ReadOnly)) {
          flags |= iree_hal_hip_BindingBits_READ_ONLY;
        }
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::Indirect)) {
          flags |= iree_hal_hip_BindingBits_INDIRECT;
        }
        bindingFlags.push_back(flags);
      }
      auto bindingFlagsRef = iree_hal_hip_BindingBits_vec_create(
          builder, bindingFlags.data(), bindingFlags.size());

      iree_hal_hip_ExportDef_start(builder);
      iree_hal_hip_ExportDef_module_ordinal_add(builder, 0); // always 0 today
      iree_hal_hip_ExportDef_kernel_name_add(builder, kernelNameRef);
      iree_hal_hip_ExportDef_block_dims_add(builder, &blockDims);
      iree_hal_hip_ExportDef_block_shared_memory_size_add(
          builder, blockSharedMemorySize);
      iree_hal_hip_ExportDef_constant_count_add(builder, constantCount);
      iree_hal_hip_ExportDef_binding_flags_add(builder, bindingFlagsRef);
      iree_hal_hip_ExportDef_debug_info_add(builder, exportDebugInfos[ordinal]);
      exportRefs[ordinal] = iree_hal_hip_ExportDef_end(builder);
    }
    auto exportsRef = builder.createOffsetVecDestructive(exportRefs);

    iree_hal_hip_ExecutableDef_exports_add(builder, exportsRef);
    iree_hal_hip_ExecutableDef_modules_add(builder, modulesRef);
    iree_hal_hip_ExecutableDef_source_files_add(builder, sourceFilesRef);
    iree_hal_hip_ExecutableDef_end_as_root(builder);

    return builder.getBufferAttr(variantOp.getContext());
  }

private:
  const ROCMOptions &options;
};

class AMDGPUTargetDevice final : public TargetDevice {
public:
  AMDGPUTargetDevice(const ROCMOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto deviceConfigAttr = b.getDictionaryAttr({});
    auto executableConfigAttr = b.getDictionaryAttr({});

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("rocm")->getDefaultExecutableTargets(
        context, "amdgpu", executableConfigAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("amdgpu"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
  }

private:
  const ROCMOptions &options;
};

class HIPTargetDevice final : public TargetDevice {
public:
  HIPTargetDevice(const ROCMOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto deviceConfigAttr = b.getDictionaryAttr({});
    auto executableConfigAttr = b.getDictionaryAttr({});

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("rocm")->getDefaultExecutableTargets(
        context, "hip", executableConfigAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("hip"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
  }

private:
  const ROCMOptions &options;
};

namespace {

struct ROCMSession final
    : PluginSession<ROCMSession, ROCMOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void onRegisterDialects(DialectRegistry &registry) {
    registry.insert<IREE::ROCM::ROCMDialect>();
  }
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"amdgpu", ...
    targets.add("amdgpu", [&]() {
      return std::make_shared<AMDGPUTargetDevice>(options);
    });
    // #hal.device.target<"hip", ...
    targets.add("hip",
                [&]() { return std::make_shared<HIPTargetDevice>(options); });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.executable.target<"rocm", ...
    targets.add("rocm", [&]() {
      LLVMInitializeAMDGPUTarget();
      LLVMInitializeAMDGPUTargetMC();
      LLVMInitializeAMDGPUTargetInfo();
      LLVMInitializeAMDGPUAsmParser();
      LLVMInitializeAMDGPUAsmPrinter();
      return std::make_shared<ROCMTargetBackend>(options);
    });
  }
};

// Iterate over ukernel bitcode embedded-data files, and insert them into the
// EmbeddedDataDirectory singleton.
static void addAMDGPUUkernelBitcodeToGlobalEmbeddedDataDirectory() {
  EmbeddedDataDirectory::withGlobal([](EmbeddedDataDirectory &dir) {
    const iree_file_toc_t *toc = iree_uk_amdgpu_bitcode_create();
    for (size_t i = 0; i < iree_uk_amdgpu_bitcode_size(); ++i) {
      dir.addFile(toc[i].name, llvm::StringRef{toc[i].data, toc[i].size});
    }
  });
}

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_rocm(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::ROCMSession>(
      "hal_target_rocm");
  mlir::iree_compiler::IREE::HAL::
      addAMDGPUUkernelBitcodeToGlobalEmbeddedDataDirectory();
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::HAL::ROCMOptions);
