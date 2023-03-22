// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "iree_cuda/libdevice_embedded.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

static llvm::cl::opt<bool> dumpPtx(
    "iree-hal-cuda-dump-ptx", llvm::cl::init(false),
    llvm::cl::desc("Dump ptx to the debug stream."));

static llvm::cl::opt<std::string> clTargetChip(
    "iree-hal-cuda-llvm-target-arch", llvm::cl::desc("LLVM target chip."),
    llvm::cl::init("sm_35"));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static void dumpBitcodeToPath(StringRef path, StringRef baseName,
                              StringRef suffix, StringRef extension,
                              llvm::Module &module) {
  llvm::SmallVector<char, 0> data;
  llvm::raw_svector_ostream ostream(data);
  llvm::WriteBitcodeToFile(module, ostream);
  dumpDataToPath(path, baseName, suffix, extension,
                 StringRef(data.data(), data.size()));
}

static std::string translateModuleToISA(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CGFT_AssemblyFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

/// Resolve __nv function by linking libdevice module.
/// |objectAttrs| may optionally specify additional bitcode files to link into
/// the generated code.
static LogicalResult linkObjects(Location loc, llvm::Module &module,
                                 llvm::TargetMachine &targetMachine,
                                 ArrayAttr objectAttrs) {
  // Ensure consistent target information.
  const llvm::Triple &targetTriple = targetMachine.getTargetTriple();
  module.setDataLayout(targetMachine.createDataLayout());
  module.setTargetTriple(targetTriple.str());

  auto specializationCallback = [&](llvm::Module &userModule) {
    // TODO(thomasraoux): inject __nvvm_reflect-style functions/globals for
    // bitcode specialization based on the targetMachine and configuration.
    // These could use any information we have on the IREE side as well as the
    // TargetMachine instead of just what __nvvm_reflect supports (arch/etc).
  };

  // Link user modules and libdevice (if required).
  // Note that linking order matters:
  llvm::Linker linker(module);
  unsigned linkerFlags =
      llvm::Linker::LinkOnlyNeeded | llvm::Linker::OverrideFromSrc;
  if (failed(linkBitcodeObjects(loc, linker, linkerFlags, targetMachine,
                                objectAttrs, module.getContext(),
                                specializationCallback))) {
    return mlir::emitError(loc)
           << "failed linking in user objects for target triple '"
           << targetTriple.str() << "'";
  }

  if (anyRequiredSymbols(module, "__nv_")) {
    llvm::MemoryBufferRef bitcodeBufferRef(
        llvm::StringRef(libdevice_embedded_create()->data,
                        libdevice_embedded_create()->size),
        "libdevice.xx.bc");
    if (failed(linkBitcodeModule(
            loc, linker, linkerFlags, targetMachine, "libdevice.xx.bc",
            llvm::parseBitcodeFile(bitcodeBufferRef, module.getContext())))) {
      return mlir::emitError(loc) << "failed linking in embedded libdevice "
                                     "bitcode for target triple '"
                                  << targetTriple.str() << "'";
    }
  }

  return success();
}

/// Performs optimizations on |module| (including LTO-style whole-program ones).
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
  si.registerCallbacks(pic, &fam);

  llvm::PassBuilder pb(&targetMachine, pto, std::nullopt, &pic);
  llvm::ModulePassManager mpm;
  StringRef nnvmReflectPassName = "nvvm-reflect";
  StringRef nnvmIntrRangePassName = "nvvm-intr-range";
  if (pb.parsePassPipeline(mpm, nnvmReflectPassName)) {
    llvm::errs() << "Could not parse -" << nnvmReflectPassName << "\n";
  }
  if (pb.parsePassPipeline(mpm, nnvmIntrRangePassName)) {
    llvm::errs() << "Could not parse -" << nnvmIntrRangePassName << "\n";
  }
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

class CUDATargetBackend final : public TargetBackend {
 public:
  CUDATargetBackend() = default;

  std::string name() const override { return "cuda"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect,
                    IREE::Codegen::IREECodegenDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're bitcode
    // ala libdevice.bc, etc).
    if (variantOp.isExternal()) return;

    buildLLVMGPUTransformPassPipeline(passManager, false);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // TODO(thomasraoux): property handle export ordinals; this code is assuming
    // that ordinals are dense starting at 0 but that is not required.

    // Collect all the entry point parameters.
    SmallVector<std::array<int32_t, 3>> workgroupSizes;
    SmallVector<uint32_t> workgroupLocalMemories;
    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      std::array<int32_t, 3> workgroupSize;
      if (Optional<ArrayAttr> workgroupSizeAttr = exportOp.getWorkgroupSize()) {
        for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
          workgroupSize[it.index()] = it.value().cast<IntegerAttr>().getInt();
        }
      } else {
        workgroupSize = {1, 1, 1};
      }
      workgroupSizes.push_back(workgroupSize);
      uint32_t workgroupLocalMemory = 0;
      if (auto workgroupLocalMemoryAttr = exportOp.getWorkgroupLocalMemory()) {
        workgroupLocalMemory = workgroupLocalMemoryAttr->getSExtValue();
      }
      workgroupLocalMemories.push_back(workgroupLocalMemory);
    }

    SmallVector<std::string> entryPointNames;
    std::string ptxImage;
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

      // Take exported names verbatim. The user must have already sanitized
      // these to match the names in their kernels. We don't support any kind of
      // mangling and if the user was silly enough to rely on nvcc C++ mangling
      // they'll have to figure that out.
      for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
        entryPointNames.emplace_back(exportOp.getSymName());
      }

      auto objectAttr = variantOp.getObjects()
                            ->getValue()
                            .front()
                            .cast<IREE::HAL::ExecutableObjectAttr>();
      if (auto data = objectAttr.loadData()) {
        ptxImage = data.value();
      } else {
        return variantOp.emitOpError()
               << "object file could not be loaded: " << objectAttr;
      }
    } else {
      ModuleOp innerModuleOp = variantOp.getInnerModule();

      // Remove all the functions that are not part of the CUDA kernel.
      // TODO(thomasraoux): remove this? this should not be required.
      auto illegalFuncOps =
          llvm::to_vector<4>(innerModuleOp.getOps<func::FuncOp>());
      for (auto funcOp : illegalFuncOps) {
        funcOp.erase();
      }

      auto llvmModule =
          mlir::translateModuleToLLVMIR(innerModuleOp, context, libraryName);
      if (!llvmModule) {
        return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                        "dialect to the native llvm::Module";
      }

      for (auto [exportOp, workgroupSize] :
           llvm::zip_equal(variantOp.getOps<IREE::HAL::ExecutableExportOp>(),
                           workgroupSizes)) {
        auto *llvmFunc = llvmModule->getFunction(exportOp.getName());
        if (llvmFunc->isDeclaration()) continue;

        // setName will make sure the function name is unique.
        llvmFunc->setName(sanitizeSymbolName(exportOp.getName()));
        entryPointNames.emplace_back(llvmFunc->getName());

        auto *annotations =
            llvmModule->getOrInsertNamedMetadata("nvvm.annotations");
        auto setMetadataValueI32 = [&](StringRef name, int value) {
          llvm::Metadata *llvmMetadata[] = {
              llvm::ValueAsMetadata::get(llvmFunc),
              llvm::MDString::get(llvmModule->getContext(), name),
              llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(llvmModule->getContext()), value))};
          annotations->addOperand(
              llvm::MDNode::get(llvmModule->getContext(), llvmMetadata));
        };
        // Mark the entry point as a kernel.
        setMetadataValueI32("kernel", 1);
        // Set the maximum number of threads in the thread block (CTA).
        setMetadataValueI32("maxntidx", workgroupSize[0]);
        setMetadataValueI32("maxntidy", workgroupSize[1]);
        setMetadataValueI32("maxntidz", workgroupSize[2]);
      }

      std::unique_ptr<llvm::TargetMachine> targetMachine;
      {
        llvm::Triple triple("nvptx64-nvidia-cuda");
        std::string targetChip = clTargetChip;
        std::string features = "+ptx60";
        std::string error;
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget("", triple, error);
        if (target == nullptr) {
          return variantOp.emitError() << "cannot initialize target triple";
        }
        targetMachine.reset(target->createTargetMachine(
            triple.str(), targetChip, features, {}, {}));
        if (targetMachine == nullptr) {
          return variantOp.emitError() << "cannot initialize target machine";
        }
      }

      // Dump just the codegen bitcode before linking and optimization.
      if (!options.dumpIntermediatesPath.empty()) {
        dumpBitcodeToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                          variantOp.getName(), ".codegen.bc", *llvmModule);
      }

      // Link user and device bitcode alongside the generated module.
      if (failed(linkObjects(variantOp.getLoc(), *llvmModule, *targetMachine,
                             variantOp.getObjectsAttr()))) {
        return failure();
      }

      // Dump all linked bitcode prior to optimization.
      if (!options.dumpIntermediatesPath.empty()) {
        dumpBitcodeToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                          variantOp.getName(), ".linked.bc", *llvmModule);
      }

      // Run LTO-style full optimization on the linked modules.
      optimizeModule(*llvmModule, *targetMachine);

      // Dump bitcode post-linking and optimization.
      if (!options.dumpIntermediatesPath.empty()) {
        dumpBitcodeToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                          variantOp.getName(), ".optimized.bc", *llvmModule);
      }

      // Serialize CUDA kernel into the binary that we will embed in the
      // final FlatBuffer.
      ptxImage = translateModuleToISA(*llvmModule, *targetMachine);
    }

    if (dumpPtx) {
      llvm::dbgs() << ptxImage;
    }
    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath(options.dumpBinariesPath, options.dumpBaseName,
                     variantOp.getName(), ".ptx", ptxImage);
    }

    FlatbufferBuilder builder;
    iree_CUDAExecutableDef_start_as_root(builder);

    auto ptxImageRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(ptxImage.c_str()),
        ptxImage.size());
    iree_CUDABlockSizeDef_vec_start(builder);
    for (const auto &workgroupSize : workgroupSizes) {
      iree_CUDABlockSizeDef_vec_push_create(builder, workgroupSize[0],
                                            workgroupSize[1], workgroupSize[2]);
    }
    auto blockSizesRef = iree_CUDABlockSizeDef_vec_end(builder);
    auto workgroupLocalMemoriesRef =
        builder.createInt32Vec(workgroupLocalMemories);
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_CUDAExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_CUDAExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_CUDAExecutableDef_shared_memory_size_add(builder,
                                                  workgroupLocalMemoriesRef);
    iree_CUDAExecutableDef_ptx_image_add(builder, ptxImageRef);
    iree_CUDAExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(getExecutableTarget(context));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    // Add some configurations to the `hal.executable.target` attribute.
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(StringAttr::get(context, name), value);
    };
    // Set target arch
    addConfig("target_arch", StringAttr::get(context, clTargetChip));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("cuda"), b.getStringAttr("cuda-nvptx-fb"),
        configAttr);
  }
};

void registerCUDATargetBackends() {
  // #hal.device.target<"cuda", ...
  // #hal.executable.target<"cuda", ...
  static TargetBackendRegistration registration("cuda", [=]() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXAsmPrinter();
    return std::make_shared<CUDATargetBackend>();
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
