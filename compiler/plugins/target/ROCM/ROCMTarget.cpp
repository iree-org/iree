// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./ROCMTargetFeatures.h"
#include "./ROCMTargetUtils.h"

#include <cstdint>
#include <mutex>

#include "compiler/plugins/target/ROCM/ROCMTargetFeatures.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/rocm_executable_def_builder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

class ROCMOptions {
public:
  std::string targetChip = "gfx908";
  bool linkBitcode = true;
  int wavesPerEu = 0;
  std::string enableROCMUkernels = "none";

  const std::string &getBitcodeDirectory() const {
    if (bitcodeDirectory.empty()) {
      return getDefaultBitcodeDirectory();
    }
    return bitcodeDirectory;
  }

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("ROCM HAL Target");
    binder.opt<std::string>("iree-rocm-target-chip", targetChip,
                            llvm::cl::cat(category),
                            llvm::cl::desc("ROCm target Chip"));
    binder.opt<bool>("iree-rocm-link-bc", linkBitcode, llvm::cl::cat(category),
                     llvm::cl::desc("Whether to try Linking to AMD Bitcodes"));
    binder.opt<std::string>("iree-rocm-bc-dir", bitcodeDirectory,
                            llvm::cl::cat(category),
                            llvm::cl::desc("Directory of ROCM Bitcode"));
    binder.opt<int>("iree-rocm-waves-per-eu", wavesPerEu,
                    llvm::cl::cat(category),
                    llvm::cl::desc("Optimization hint specifying minimum "
                                   "number of waves per execution unit"));
    binder.opt<std::string>(
        "iree-rocm-enable-ukernels", enableROCMUkernels,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Enables microkernels in the llvmcpu backend. May be "
            "`default`, `none`, `all`, or a comma-separated list of "
            "specific unprefixed microkernels to enable, e.g. `mmt4d`."));
  }

private:
  static const std::string &getDefaultBitcodeDirectory() {
    static std::string defaultValue =
        mlir::iree_compiler::findPlatformLibDirectory("rocm");
    return defaultValue;
  }
  std::string bitcodeDirectory;
};
} // namespace

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

// Modified from lib/Target/AMDGPU/AMDGPUAttributor.cpp.
// Adds argument hints to preload kernel arguments to SGPRs.
// TODO: Query max number of user SGPRs from target machine.
static void addPreloadKernArgHint(llvm::Function *F) {
  static constexpr size_t maxSGPRs = 16;
  for (size_t i = 0, e = std::min(F->arg_size(), maxSGPRs); i != e; ++i) {
    llvm::Argument *Arg = F->getArg(i);
    // Check for incompatible attributes.
    if (Arg->hasByRefAttr() || Arg->hasNestAttr())
      break;
    Arg->addAttr(llvm::Attribute::InReg);
  }
}

class ROCMTargetDevice final : public TargetDevice {
public:
  ROCMTargetDevice(const ROCMOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(b.getStringAttr(name), value);
    };

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    addConfig("legacy_sync", b.getUnitAttr());

    auto configAttr = b.getDictionaryAttr(configItems);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("rocm")->getDefaultExecutableTargets(
        context, "rocm", configAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("rocm"),
                                            configAttr, executableTargetAttrs);
  }

private:
  const ROCMOptions &options;
};

class ROCMTargetBackend final : public TargetBackend {
public:
  ROCMTargetBackend(const ROCMOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "rocm"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(b.getStringAttr(name), value);
    };

    addConfig("target_arch", b.getStringAttr(options.targetChip));
    addConfig("ukernels", b.getStringAttr(options.enableROCMUkernels));

    ArrayAttr mmaAttrs = getROCMSupportedMmaAttrs(context, options.targetChip);
    if (mmaAttrs) {
      addConfig("mma_intrinsics", mmaAttrs);
    }

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
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    // For now we disable configuration if the variant has external object
    // files.
    if (variantOp.isExternal())
      return;

    buildLLVMGPUCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're bitcode
    // ala libdevice.bc, etc).
    if (variantOp.isExternal())
      return;

    buildLLVMGPUCodegenPassPipeline(passManager, true);
  }

  // Performs optimizations on |module| (including LTO-style whole-program
  // ones). Inspired by code section in
  // https://github.com/openxla/iree/blob/main/compiler/plugins/target/CUDA/CUDATarget.cpp
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
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;
    const std::string &bitcodeDirectory = options.getBitcodeDirectory();

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    ModuleOp innerModuleOp = variantOp.getInnerModule();

    auto llvmModule =
        mlir::translateModuleToLLVMIR(innerModuleOp, context, libraryName);
    if (!llvmModule) {
      return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                      "dialect to the native llvm::Module";
    }

    // Collect all the entry point names.
    SmallVector<IREE::HAL::ExecutableExportOp> exportOps;
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOpMap;
    for (auto op : variantOp.getExportOps()) {
      exportOps.push_back(op);
      exportOpMap[op.getSymName()] = op;
    }
    std::vector<std::array<int32_t, 3>> workgroupSizes;
    SmallVector<uint32_t> workgroupLocalMemories;
    int32_t subgroupSize = 64;
    StringRef subTarget = options.targetChip;
    StringRef GFX9("gfx9");
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      int32_t flatWgSize = 1;
      auto *llvmFunc = llvmModule->getFunction(func.getName());
      if (llvmFunc->isDeclaration())
        continue;
      std::array<int32_t, 3> workgroupSize;
      auto exportOp = exportOpMap[func.getName()];
      if (std::optional<ArrayAttr> workgroupSizeAttr =
              exportOp.getWorkgroupSize()) {
        for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
          workgroupSize[it.index()] = it.value().cast<IntegerAttr>().getInt();
          flatWgSize *= it.value().cast<IntegerAttr>().getInt();
        }
      } else {
        workgroupSize = {1, 1, 1};
      }

      if (auto setSubgroupSize = getSubgroupSize(exportOp)) {
        if (subgroupSize != 32 && subgroupSize != 64) {
          return variantOp.emitError()
                 << "invalid subgroup size " << subgroupSize;
        }
        subgroupSize = *setSubgroupSize;
      }

      int64_t wavesPerEu = options.wavesPerEu;
      IREE::Codegen::TranslationInfoAttr translationInfo =
          getTranslationInfo(exportOp);
      if (auto translationConfig = translationInfo.getConfiguration()) {
        if (auto attr = dyn_cast_or_null<IntegerAttr>(
                translationConfig.get("amdgpu-waves-per-eu"))) {
          wavesPerEu = attr.getValue().getSExtValue();
        }
      }

      workgroupSizes.push_back(workgroupSize);
      uint32_t workgroupLocalMemory = 0;
      if (auto workgroupLocalMemoryAttr = exportOp.getWorkgroupLocalMemory()) {
        workgroupLocalMemory = workgroupLocalMemoryAttr->getSExtValue();
      }
      workgroupLocalMemories.push_back(workgroupLocalMemory);
      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-workgroup-size(1, 256) attribute.
      // 3. Insert amdgpu-implicitarg-num-bytes=56 (which must be set on OpenCL
      // and HIP kernels per Clang)
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      std::string wgSizeRange = std::string("1, ") + std::to_string(flatWgSize);
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", wgSizeRange);
      if (wavesPerEu > 0)
        llvmFunc->addFnAttr("amdgpu-waves-per-eu", std::to_string(wavesPerEu));
      if (subTarget.starts_with(GFX9))
        addPreloadKernArgHint(llvmFunc);
    }

    std::unique_ptr<llvm::TargetMachine> targetMachine;
    {
      llvm::Triple triple("amdgcn-amd-amdhsa");
      std::string error;
      const llvm::Target *target =
          llvm::TargetRegistry::lookupTarget("", triple, error);
      if (target == nullptr) {
        return variantOp.emitError() << "cannot initialize target triple";
      }
      llvm::TargetOptions opt;
      opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
      opt.UnsafeFPMath = false;
      opt.NoInfsFPMath = false;
      opt.NoNaNsFPMath = true;
      std::string features;
      if (subTarget.starts_with(GFX9)) {
        features = "+sramecc,-xnack";
      } else {
        // GFX 10 or 11.
        if (subgroupSize == 32)
          features = "+wavefrontsize32";
        if (subgroupSize == 64)
          features = "+wavefrontsize64";
      }

      targetMachine.reset(target->createTargetMachine(
          triple.str(), options.targetChip, features, opt,
          llvm::Reloc::Model::PIC_, std::nullopt,
          llvm::CodeGenOptLevel::Aggressive));

      if (targetMachine == nullptr) {
        return variantOp.emitError() << "cannot initialize target machine";
      }
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());

    for (llvm::Function &f : llvmModule->functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);

    // Link user modules and libdevice (if required).
    // Note that linking order matters:
    llvm::Linker linker(*llvmModule);
    if (failed(linkCmdlineBitcodeFiles(
            variantOp.getLoc(), linker, llvm::Linker::OverrideFromSrc,
            *targetMachine, llvmModule->getContext()))) {
      return failure();
    }

    if (!options.enableROCMUkernels.empty() &&
        options.enableROCMUkernels != "none") {
      auto enabledUkernelsStr = StringRef(options.enableROCMUkernels);
      if (failed(linkUkernelBCFiles(
              variantOp.getLoc(), llvmModule.get(), enabledUkernelsStr,
              options.targetChip, bitcodeDirectory,
              llvm::Linker::OverrideFromSrc, *targetMachine)))
        return failure();
    }
    // Link module to Device Library
    if (options.linkBitcode) {
      if (bitcodeDirectory.empty()) {
        return variantOp.emitError()
               << "cannot find ROCM bitcode files. Check your installation "
                  "consistency and in the worst case, set --iree-rocm-bc-dir= "
                  "to an explicit location on your system.";
      }
      if (failed(linkROCDLIfNecessary(variantOp.getLoc(), llvmModule.get(),
                                      options.targetChip, bitcodeDirectory)))
        return failure();
    }
    if (!serOptions.dumpIntermediatesPath.empty()) {
      dumpModuleToPath(serOptions.dumpIntermediatesPath,
                       serOptions.dumpBaseName, variantOp.getName(),
                       ".linked.ll", *llvmModule);
    }
    // Add Optimize module
    optimizeModule(*llvmModule, *targetMachine);
    // Store optimized ll.
    if (!serOptions.dumpIntermediatesPath.empty()) {
      dumpModuleToPath(serOptions.dumpIntermediatesPath,
                       serOptions.dumpBaseName, variantOp.getName(),
                       ".optimized.ll", *llvmModule);
    }
    // Serialize hsaco kernel into the binary that we will embed in the
    // final FlatBuffer.
    std::unique_ptr<llvm::Module> moduleCopy;
    if (!serOptions.dumpIntermediatesPath.empty()) {
      moduleCopy = llvm::CloneModule(*llvmModule);
      if (!moduleCopy)
        llvm::errs() << "Error: cloning LLVM IR failed\n";
    }
    std::string targetObj = translateModuleToObj(*llvmModule, *targetMachine);
    std::string targetHSACO =
        createHsaco(variantOp.getLoc(), targetObj, libraryName);
    if (targetHSACO.empty()) {
      return failure();
    }

    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serOptions.dumpBinariesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".hsaco", targetHSACO);
    }

    iree_compiler::FlatbufferBuilder builder;
    iree_hal_rocm_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    SmallVector<iree_hal_rocm_SourceFileDef_ref_t> sourceFileRefs;
    if (auto sourcesAttr = variantOp.getSourcesAttr()) {
      for (auto sourceAttr : llvm::reverse(sourcesAttr.getValue())) {
        if (auto resourceAttr = dyn_cast_if_present<DenseResourceElementsAttr>(
                sourceAttr.getValue())) {
          auto filenameRef = builder.createString(sourceAttr.getName());
          auto contentRef = builder.streamUint8Vec([&](llvm::raw_ostream &os) {
            auto blobData = resourceAttr.getRawHandle().getBlob()->getData();
            os.write(blobData.data(), blobData.size());
            return true;
          });
          sourceFileRefs.push_back(iree_hal_rocm_SourceFileDef_create(
              builder, filenameRef, contentRef));
        }
      }
      std::reverse(sourceFileRefs.begin(), sourceFileRefs.end());
    }

    SmallVector<StringRef> entryPointNames;
    SmallVector<iree_hal_rocm_FileLineLocDef_ref_t> sourceLocationRefs;
    entryPointNames.resize(exportOps.size());
    for (auto exportOp : exportOps) {
      auto ordinalAttr = exportOp.getOrdinalAttr();
      if (!ordinalAttr) {
        return mlir::emitError(exportOp.getLoc())
               << "could not compile rocm binary: export op is missing ordinal";
      }
      int64_t ordinal = ordinalAttr.getInt();
      entryPointNames[ordinal] = exportOp.getName();

      // Optional source location information for debugging/profiling.
      if (serOptions.debugLevel >= 1) {
        // We only ever resize to the maximum -- so all previous data will
        // be kept as-is.
        sourceLocationRefs.resize(exportOps.size());
        if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
          auto filenameRef = builder.createString(loc->getFilename());
          sourceLocationRefs[ordinal] = iree_hal_rocm_FileLineLocDef_create(
              builder, filenameRef, loc->getLine());
        }
      }
    }

    // Optional compilation stage source files.
    SmallVector<iree_hal_rocm_StageLocationsDef_ref_t> stageLocationsRefs;
    if (serOptions.debugLevel >= 3) {
      for (auto exportOp : exportOps) {
        SmallVector<iree_hal_rocm_StageLocationDef_ref_t> stageLocationRefs;
        if (auto locsAttr = exportOp.getSourceLocsAttr()) {
          for (auto locAttr : locsAttr.getValue()) {
            if (auto loc =
                    findFirstFileLoc(cast<LocationAttr>(locAttr.getValue()))) {
              auto stageNameRef = builder.createString(locAttr.getName());
              auto filenameRef = builder.createString(loc->getFilename());
              stageLocationRefs.push_back(iree_hal_rocm_StageLocationDef_create(
                  builder, stageNameRef,
                  iree_hal_rocm_FileLineLocDef_create(builder, filenameRef,
                                                      loc->getLine())));
            }
          }
        }
        if (!stageLocationRefs.empty()) {
          // We only ever resize to the maximum -- so all previous data will
          // be kept as-is.
          stageLocationsRefs.resize(exportOps.size());
          int64_t ordinal = exportOp.getOrdinalAttr().getInt();
          stageLocationsRefs[ordinal] = iree_hal_rocm_StageLocationsDef_create(
              builder, builder.createOffsetVecDestructive(stageLocationRefs));
        }
      }
    }

    auto hsacoRef = flatbuffers_string_create(builder, targetHSACO.c_str(),
                                              targetHSACO.size());

    auto entryPointsRef = builder.createStringVec(entryPointNames);
    iree_hal_rocm_BlockSizeDef_vec_start(builder);
    auto blockSizes = workgroupSizes.begin();
    for (int i = 0, e = entryPointNames.size(); i < e; ++i) {
      iree_hal_rocm_BlockSizeDef_vec_push_create(
          builder, (*blockSizes)[0], (*blockSizes)[1], (*blockSizes)[2]);
      ++blockSizes;
    }
    auto workgroupLocalMemoriesRef =
        builder.createInt32Vec(workgroupLocalMemories);
    auto blockSizesRef = iree_hal_rocm_BlockSizeDef_vec_end(builder);
    iree_hal_rocm_ExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_hal_rocm_ExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_hal_rocm_ExecutableDef_shared_memory_sizes_add(
        builder, workgroupLocalMemoriesRef);
    iree_hal_rocm_ExecutableDef_hsaco_image_add(builder, hsacoRef);
    if (!sourceLocationRefs.empty()) {
      auto sourceLocationsRef =
          builder.createOffsetVecDestructive(sourceLocationRefs);
      iree_hal_rocm_ExecutableDef_source_locations_add(builder,
                                                       sourceLocationsRef);
    }
    if (!stageLocationsRefs.empty()) {
      auto stageLocationsRef =
          builder.createOffsetVecDestructive(stageLocationsRefs);
      iree_hal_rocm_ExecutableDef_stage_locations_add(builder,
                                                      stageLocationsRef);
    }
    if (!sourceFileRefs.empty()) {
      auto sourceFilesRef = builder.createOffsetVecDestructive(sourceFileRefs);
      iree_hal_rocm_ExecutableDef_source_files_add(builder, sourceFilesRef);
    }
    iree_hal_rocm_ExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));

    if (!serOptions.dumpIntermediatesPath.empty()) {
      std::string targetISA =
          translateModuleToISA(*moduleCopy.get(), *targetMachine);
      dumpDataToPath(serOptions.dumpIntermediatesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".rocmasm", targetISA);
    }

    return success();
  }

private:
  const ROCMOptions &options;
};

namespace {
struct ROCMSession
    : public PluginSession<ROCMSession, ROCMOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"rocm", ...
    targets.add("rocm",
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

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::HAL::ROCMOptions);
