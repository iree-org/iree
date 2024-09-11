// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ROCMTargetUtils.h"

#include <cstdint>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMLinkerUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
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
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

struct ROCmOptions {
  std::string target = "gfx908";
  std::string targetFeatures = "";
  std::string bitcodeDirectory = getDefaultBitcodeDirectory();
  int wavesPerEu = 0;
  std::string enableROCMUkernels = "none";
  bool legacySync = true;

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
    binder.opt<bool>("iree-hip-legacy-sync", legacySync, cl::cat(category),
                     cl::desc("Enables 'legacy-sync' mode, which is required "
                              "for inline execution."));
  }

  LogicalResult verify(mlir::Builder &builder) const {
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

// Extracts the amdgpu chipset version from the chip architecture in the
// executable target attribute.
static FailureOr<amdgpu::Chipset>
getChipsetVersion(ExecutableTargetAttr targetAttr) {
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(targetAttr);
  if (!gpuTarget)
    return failure();

  return amdgpu::Chipset::parse(gpuTarget.getArch());
}

// Set attributes on `funcOp` in order to use upstream's translation of
// ROCDL dialect attributes to LLVM. Primarily this is `rocdl.kernel`
// (sets the calling convention and workgroup size uniformity) but this will
// also set both forms of workgroup size metadata from `exportOp` (if it is set)
// and will set the waves_per_eq flag where relevant. Finally, it will mark
// kernel arguments `inreg` to enable argument preloading on supported
// architectures.
static void annotateKernelForTranslation(LLVM::LLVMFuncOp funcOp,
                                         ExecutableExportOp exportOp,
                                         ExecutableTargetAttr targetAttr,
                                         OpBuilder &builder) {
  auto *rocdlDialect =
      funcOp.getContext()->getLoadedDialect<ROCDL::ROCDLDialect>();
  UnitAttr unitAttr = builder.getUnitAttr();
  rocdlDialect->getKernelAttrHelper().setAttr(funcOp, unitAttr);
  std::optional<ArrayAttr> workgroupSizeAttr = exportOp.getWorkgroupSize();
  if (workgroupSizeAttr && workgroupSizeAttr->size() <= 3) {
    std::array<int32_t, 3> wgSizes;
    int32_t flatWgSize = 1;
    for (auto [value, attr] : llvm::zip_equal(
             wgSizes, workgroupSizeAttr->getAsRange<IntegerAttr>())) {
      value = attr.getInt();
      flatWgSize *= value;
    }
    rocdlDialect->getReqdWorkGroupSizeAttrHelper().setAttr(
        funcOp, builder.getDenseI32ArrayAttr(wgSizes));
    rocdlDialect->getFlatWorkGroupSizeAttrHelper().setAttr(
        funcOp,
        builder.getStringAttr(Twine(flatWgSize) + "," + Twine(flatWgSize)));
  }

  if (std::optional<IntegerAttr> attr =
          getConfigIntegerAttr(targetAttr, "waves_per_eu")) {
    rocdlDialect->getWavesPerEuAttrHelper().setAttr(funcOp, *attr);
  }

  // Kernel argument preloading is only supported on gfx940 and newer targets
  // from the CDNA family. This is enabled using the `inreg` function argument
  // attribute.
  FailureOr<amdgpu::Chipset> chipset = getChipsetVersion(targetAttr);
  if (failed(chipset))
    return;
  if (chipset->majorVersion != 9 || *chipset < amdgpu::Chipset(9, 4, 0))
    return;

  auto inRegAttrName =
      builder.getStringAttr(LLVM::LLVMDialect::getInRegAttrName());
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
    funcOp.setArgAttr(i, inRegAttrName, unitAttr);
}

static void dumpModuleToPath(StringRef path, StringRef baseName,
                             StringRef suffix, StringRef extension,
                             llvm::Module &module) {
  llvm::SmallVector<char, 0> data;
  llvm::raw_svector_ostream ostream(data);
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

class ROCMTargetDevice final : public TargetDevice {
public:
  ROCMTargetDevice(const ROCmOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);

    SmallVector<NamedAttribute> deviceConfigAttrs;
    if (options.legacySync) {
      // Indicates that the runtime HAL driver operates only in the legacy
      // synchronous mode.
      deviceConfigAttrs.emplace_back(b.getStringAttr("legacy_sync"),
                                     b.getUnitAttr());
    }
    auto deviceConfigAttr = b.getDictionaryAttr(deviceConfigAttrs);

    SmallVector<NamedAttribute> executableConfigAttrs;
    auto executableConfigAttr = b.getDictionaryAttr(executableConfigAttrs);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("rocm")->getDefaultExecutableTargets(
        context, "rocm", executableConfigAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("hip"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
  }

private:
  const ROCmOptions &options;
};

class ROCMTargetBackend final : public TargetBackend {
public:
  ROCMTargetBackend(const ROCmOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "hip"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    if (auto target = getExecutableTarget(context))
      executableTargetAttrs.push_back(target);
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(b.getStringAttr(name), value);
    };

    if (failed(options.verify(b)))
      return nullptr;

    if (auto target = GPU::getHIPTargetDetails(options.target,
                                               options.targetFeatures, context))
      addConfig("iree.gpu.target", target);

    addConfig("ukernels", b.getStringAttr(options.enableROCMUkernels));
    if (options.wavesPerEu > 0)
      addConfig("waves_per_eu", b.getI64IntegerAttr(options.wavesPerEu));

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("rocm"), b.getStringAttr("rocm-hsaco-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerROCDLDialectTranslation(registry);
    registry.insert<IREE::Codegen::IREECodegenDialect>();
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<IREE::GPU::IREEGPUDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<ROCDL::ROCDLDialect>();
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

  // Performs optimizations on |module| (including LTO-style whole-program
  // ones). Inspired by code section in
  // https://github.com/iree-org/iree/blob/main/compiler/plugins/target/CUDA/CUDATarget.cpp
  static void optimizeModule(llvm::Module &module,
                             llvm::TargetMachine &targetMachine) {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    fam.registerPass([&] { return targetMachine.getTargetIRAnalysis(); });

    llvm::PipelineTuningOptions pto;
    pto.SLPVectorization = false;

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

    llvm::OptimizationLevel ol = llvm::OptimizationLevel::O2;

    mpm.addPass(llvm::VerifierPass());
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
    mpm.addPass(llvm::VerifierPass());

    mpm.run(module, mam);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
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
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOpMap;
    std::optional<uint32_t> subgroupSize;
    for (IREE::HAL::ExecutableExportOp exportOp : exportOps) {
      exportOpMap[exportOp.getSymName()] = exportOp;

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
      } else if (variantOp.getObjects()->getValue().size() != 1) {
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
      // Perform the translation in a separate context to avoid any
      // multi-threading issues.
      llvm::LLVMContext context;

      // Set up attributes so upstream's conversions work right.
      for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
        // Un-exported functions are library functions or otherwise
        // not kernels, so don't need these annotations.
        if (!exportOpMap.contains(func.getName()))
          continue;
        annotateKernelForTranslation(func, exportOpMap[func.getName()],
                                     targetAttr, executableBuilder);
      }

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
              return variantOp->emitError("llvm_func_attrs attribute must be "
                                          "adictionary of strings. Attribute " +
                                          llvm::Twine(funcAttr.getName()) +
                                          " is not a StringAttr.");
            }
            llvmFunc->addFnAttr(funcAttr.getName(), value.getValue());
          }
        }
      }

      std::unique_ptr<llvm::TargetMachine> targetMachine;
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
        std::string features;
        if (targetArch.starts_with("gfx10") ||
            targetArch.starts_with("gfx11")) {
          switch (subgroupSize.value_or(64)) {
          case 32:
            features = "+wavefrontsize32";
            break;
          default:
          case 64:
            features = "+wavefrontsize64";
            break;
          }
        }
        if (!targetFeatures.empty()) {
          features += (features.empty() ? "" : ",") + targetFeatures.str();
        }

        targetMachine.reset(target->createTargetMachine(
            triple.str(), targetArch, features, opt, llvm::Reloc::Model::PIC_,
            std::nullopt, llvm::CodeGenOptLevel::Aggressive));

        if (!targetMachine) {
          return variantOp.emitError() << "cannot initialize target machine";
        }
      }

      llvmModule->setDataLayout(targetMachine->createDataLayout());

      for (llvm::Function &f : llvmModule->functions())
        f.addFnAttr(llvm::Attribute::AlwaysInline);

      // Link user-provided modules.
      llvm::Linker linker(*llvmModule);
      if (failed(linkCmdlineBitcodeFiles(
              variantOp.getLoc(), linker, llvm::Linker::OverrideFromSrc,
              *targetMachine, llvmModule->getContext()))) {
        return failure();
      }

      // Link module to any enabled ukernels.
      StringRef bitcodeDirectory = options.bitcodeDirectory;
      StringRef enabledUkernels;
      if (auto attr = getConfigStringAttr(targetAttr, "ukernels"))
        enabledUkernels = attr->getValue();
      if (!enabledUkernels.empty() && enabledUkernels != "none") {
        if (failed(linkUkernelBitcodeFiles(
                variantOp.getLoc(), llvmModule.get(), enabledUkernels,
                targetArch, bitcodeDirectory, llvm::Linker::OverrideFromSrc,
                *targetMachine))) {
          return failure();
        }
      }

      // Link module to HIP device library.
      if (bitcodeDirectory.empty()) {
        return variantOp.emitError()
               << "cannot find ROCM bitcode files. Check your installation "
                  "consistency and in the worst case, set "
                  "--iree-hip-bc-dir= to a path on your system.";
      }
      if (failed(linkHIPBitcodeIfNeeded(variantOp.getLoc(), llvmModule.get(),
                                        targetArch, bitcodeDirectory))) {
        return failure();
      }

      // Sets HIP platform globals based on the target architecture.
      if (failed(setHIPGlobals(variantOp.getLoc(), llvmModule.get(),
                               targetArch))) {
        return failure();
      }

      if (!serOptions.dumpIntermediatesPath.empty()) {
        dumpModuleToPath(serOptions.dumpIntermediatesPath,
                         serOptions.dumpBaseName, variantOp.getName(),
                         ".linked.ll", *llvmModule);
      }

      // Run LLVM optimization passes.
      optimizeModule(*llvmModule, *targetMachine);
      if (!serOptions.dumpIntermediatesPath.empty()) {
        dumpModuleToPath(serOptions.dumpIntermediatesPath,
                         serOptions.dumpBaseName, variantOp.getName(),
                         ".optimized.ll", *llvmModule);
      }

      // Dump the assembly output.
      if (!serOptions.dumpIntermediatesPath.empty()) {
        auto moduleCopy = llvm::CloneModule(*llvmModule);
        if (!moduleCopy) {
          llvm::errs() << "Error: cloning LLVM IR failed\n";
          return failure();
        }
        std::string targetISA =
            translateModuleToISA(*moduleCopy.get(), *targetMachine);
        dumpDataToPath(serOptions.dumpIntermediatesPath,
                       serOptions.dumpBaseName, variantOp.getName(), ".rocmasm",
                       targetISA);
      }

      // Serialize hsaco kernel into the binary that we will embed in the
      // final FlatBuffer.
      std::string targetObj = translateModuleToObj(*llvmModule, *targetMachine);
      targetHSACO = createHsaco(variantOp.getLoc(), targetObj, libraryName);
      if (targetHSACO.empty())
        return failure();
    }

    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serOptions.dumpBinariesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".hsaco", targetHSACO);
    }

    iree_compiler::FlatbufferBuilder builder;
    iree_hal_hip_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    // Only a single module today.
    SmallVector<iree_hal_hip_ModuleDef_ref_t> moduleRefs;
    {
      auto hsacoImageRef = flatbuffers_string_create(
          builder, targetHSACO.c_str(), targetHSACO.size());
      moduleRefs.push_back(
          iree_hal_hip_ModuleDef_create(builder, hsacoImageRef));
    }
    auto modulesRef = builder.createOffsetVecDestructive(moduleRefs);

    // Generate optional per-export debug information.
    // May be empty if no debug information was requested.
    auto exportDebugInfos =
        createExportDefs(serOptions.debugLevel, exportOps, builder);

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

    // Add the binary data to the target executable.
    executableBuilder.create<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));

    return success();
  }

private:
  const ROCmOptions &options;
};

namespace {
struct ROCMSession final
    : PluginSession<ROCMSession, ROCmOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"hip", ...
    targets.add("hip",
                [&]() { return std::make_shared<ROCMTargetDevice>(options); });
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

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_rocm(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::ROCMSession>(
      "hal_target_rocm");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::HAL::ROCmOptions);
