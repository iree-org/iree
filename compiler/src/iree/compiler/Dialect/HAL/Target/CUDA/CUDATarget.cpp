// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/LLVMPasses.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/cuda_libdevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
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
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
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

// TODO: remove this workaround altogether once we decide not to support
// CUDA 11.3
static llvm::cl::opt<bool> clDisableLoopNounrollWa(
    "iree-hal-cuda-disable-loop-nounroll-wa",
    llvm::cl::desc("Disable the workaround for bug in ptxas for CUDA version "
                   "before 11.4."),
    llvm::cl::init(true));

static llvm::cl::opt<std::string> clTargetChip(
    "iree-hal-cuda-llvm-target-arch", llvm::cl::desc("LLVM target chip."),
    llvm::cl::init("sm_35"));

namespace llvm {
class FunctionPass;
FunctionPass *createNVVMIntrRangePass(unsigned int SmVersion);
FunctionPass *createNVVMReflectPass(unsigned int SmVersion);
}  // namespace llvm
namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static std::string translateModuleToISA(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    // Workaround for CUDA driver bug
    // (https://bugs.llvm.org/show_bug.cgi?id=48771), we mark all the loops with
    // the no unroll metadata. This bug is fixed in cuda 11.4 but since we still
    // run on older driver we need to keep it.
    // TODO(thomasraoux): Remove it once we stop supporting older drivers.
    if (!clDisableLoopNounrollWa) {
      codegenPasses.add(llvm::createSetNoUnrollPass());
    }
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CGFT_AssemblyFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

/// Return true if the moule contain any __nv function that require linking with
/// libdevice module.
static bool requiresDeviceLib(const llvm::Module &module) {
  for (const llvm::Function &function : module.functions()) {
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().startswith("__nv_"))) {
      return true;
    }
  }
  return false;
}

/// Link libdevice with |module|.
static void linkModule(llvm::Module &module) {
  llvm::Linker linker(module);

  llvm::MemoryBufferRef bitcodeBufferRef(
      llvm::StringRef(cuda_libdevice_create()->data,
                      cuda_libdevice_create()->size),
      "libdevice bitcode");
  auto bitcodeModuleValue =
      llvm::parseBitcodeFile(bitcodeBufferRef, module.getContext());
  if (!bitcodeModuleValue) {
    llvm::errs() << "failed to parse CUDA libdevice bitcode: "
                 << bitcodeModuleValue.takeError();
    return;
  }
  std::unique_ptr<llvm::Module> bitcodeModule =
      std::move(bitcodeModuleValue.get());
  // Ignore the data layout of the module we're importing. This avoids a
  // warning from the linker.
  bitcodeModule->setDataLayout(module.getDataLayout());
  linker.linkInModule(
      std::move(bitcodeModule), llvm::Linker::Flags::LinkOnlyNeeded,
      [](llvm::Module &M, const llvm::StringSet<> &GVS) {
        llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
          return !GV.hasName() || (GVS.count(GV.getName()) == 0);
        });
      });
}

/// Resolve __nv function by linking libdevice module and run llvm optimization
/// passes that will inline linked functions and optimize the module.
static void linkAndOptimize(llvm::Module &module,
                            llvm::TargetMachine &targetMachine) {
  if (requiresDeviceLib(module)) linkModule(module);

  // Workaround run those passed ahead as they are temporarily disabled in NVPTX
  // target.
  llvm::legacy::PassManager legacyPM;
  legacyPM.add(llvm::createNVVMIntrRangePass(35));
  legacyPM.add(llvm::createNVVMReflectPass(35));
  legacyPM.run(module);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  fam.registerPass([&] { return targetMachine.getTargetIRAnalysis(); });

  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = false;

  llvm::PassInstrumentationCallbacks pic;

  llvm::StandardInstrumentations si(false);
  si.registerCallbacks(pic, &fam);

  llvm::PassBuilder pb(&targetMachine, pto, llvm::None, &pic);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel ol = llvm::OptimizationLevel::O2;

  llvm::ModulePassManager mpm;
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

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildLLVMGPUTransformPassPipeline(passManager, false);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;
    context.setOpaquePointers(false);

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    ModuleOp innerModuleOp = variantOp.getInnerModule();

    // Remove all the functions that are not part of the CUDA kernel.
    // TODO: Find a better solution to handle this.
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

    // Collect all the entry point names.
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps;
    for (auto op : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      exportOps[op.getSymName()] = op;
    }
    std::vector<std::array<int32_t, 3>> workgroupSizes;
    std::vector<std::string> entryPointNames;
    std::vector<uint32_t> workgroupLocalMemories;

    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      auto *llvmFunc = llvmModule->getFunction(func.getName());
      if (llvmFunc->isDeclaration()) continue;
      // setName will make sure the function name is unique.
      llvmFunc->setName(sanitizeSymbolName(func.getName()));
      entryPointNames.emplace_back(llvmFunc->getName());
      std::array<int32_t, 3> workgroupSize;
      uint32_t workgroupLocalMemory = 0;
      auto exportOp = exportOps[func.getName()];
      if (auto workgroupLocalMemoryAttr = exportOp.getWorkgroupLocalMemory()) {
        workgroupLocalMemory = workgroupLocalMemoryAttr->getSExtValue();
      }
      if (Optional<ArrayAttr> workgroupSizeAttr = exportOp.getWorkgroupSize()) {
        for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
          workgroupSize[it.index()] = it.value().cast<IntegerAttr>().getInt();
        }
      } else {
        workgroupSize = {1, 1, 1};
      }
      workgroupLocalMemories.push_back(workgroupLocalMemory);
      workgroupSizes.push_back(workgroupSize);
      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmModule->getContext(), "kernel"),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmModule->getContext()), 1))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmModule->getContext(), llvmMetadata);
      llvmModule->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvmMetadataNode);
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
      targetMachine.reset(target->createTargetMachine(triple.str(), targetChip,
                                                      features, {}, {}));
      if (targetMachine == nullptr) {
        return variantOp.emitError() << "cannot initialize target machine";
      }
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());

    linkAndOptimize(*llvmModule, *targetMachine);

    FlatbufferBuilder builder;
    iree_CUDAExecutableDef_start_as_root(builder);

    // Serialize cuda kernel into the binary that we will embed in the
    // final FlatBuffer.
    std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
    if (dumpPtx) {
      llvm::dbgs() << targetISA;
    }
    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath(options.dumpBinariesPath, options.dumpBaseName,
                     variantOp.getName(), ".ptx", targetISA);
    }
    auto ptxCudeRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetISA.c_str()),
        targetISA.size());

    auto entryPointsRef = builder.createStringVec(entryPointNames);
    auto workgroupLocalMemoriesRef =
        builder.createInt32Vec(workgroupLocalMemories);

    iree_CUDABlockSizeDef_vec_start(builder);
    auto blockSizes = workgroupSizes.begin();
    for (auto shader : entryPointNames) {
      iree_CUDABlockSizeDef_vec_push_create(builder, (*blockSizes)[0],
                                            (*blockSizes)[1], (*blockSizes)[2]);
      ++blockSizes;
    }
    auto blockSizesRef = iree_CUDABlockSizeDef_vec_end(builder);

    iree_CUDAExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_CUDAExecutableDef_shared_memory_size_add(builder,
                                                  workgroupLocalMemoriesRef);
    iree_CUDAExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_CUDAExecutableDef_ptx_image_add(builder, ptxCudeRef);
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
