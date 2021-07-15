// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/CUDA/libdevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

static llvm::cl::opt<bool> dumpPtx("iree-cuda-dump-ptx", llvm::cl::init(false),
                                   llvm::cl::desc("Dump ptx"));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

CUDATargetOptions getCUDATargetOptionsFromFlags() {
  CUDATargetOptions targetOptions;
  // TODO: flags
  return targetOptions;
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

  llvm::legacy::FunctionPassManager FPM(&module);
  llvm::legacy::PassManager MPM;
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 2;
  builder.SizeLevel = 0;
  builder.Inliner = llvm::createFunctionInliningPass();
  builder.LoopVectorize = false;

  targetMachine.adjustPassManager(builder);

  builder.populateFunctionPassManager(FPM);
  builder.populateModulePassManager(MPM);

  FPM.doInitialization();
  for (llvm::Function &func : module) {
    FPM.run(func);
  }
  FPM.doFinalization();
  MPM.run(module);
}

/// Sanitize the function name as CUDA driver doesn't allow function names with
/// '.' character.
static std::string sanitizeNameForCuda(llvm::StringRef name) {
  std::string sanitizedName(name);
  std::replace(sanitizedName.begin(), sanitizedName.end(), '.', '_');
  return sanitizedName;
}

class CUDATargetBackend final : public TargetBackend {
 public:
  CUDATargetBackend(CUDATargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "cuda"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildLLVMGPUTransformPassPipeline(passManager, false);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableVariantOp variantOp,
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

    ModuleOp innerModuleOp = variantOp.getInnerModule();

    // Remove all the functions that are not part of the CUDA kernel.
    // TODO: Find a better solution to handle this.
    auto illegalFuncOps = llvm::to_vector<4>(innerModuleOp.getOps<FuncOp>());
    for (auto funcOp : illegalFuncOps) {
      funcOp.erase();
    }
    auto halInterfaceOps =
        llvm::to_vector<1>(innerModuleOp.getOps<IREE::HAL::InterfaceOp>());
    for (auto halOp : halInterfaceOps) {
      halOp.erase();
    }

    auto llvmModule =
        mlir::translateModuleToLLVMIR(innerModuleOp, context, libraryName);
    if (!llvmModule) {
      return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                      "dialect to the native llvm::Module";
    }
    std::vector<std::array<int32_t, 3>> workgroupSizes;
    std::vector<std::string> entryPointNames;
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      auto *llvmFunc = llvmModule->getFunction(func.getName());
      if (llvmFunc->isDeclaration()) continue;
      // setName will make sure the function name is unique.
      llvmFunc->setName(sanitizeNameForCuda(func.getName()));
      entryPointNames.emplace_back(llvmFunc->getName());
      std::array<int32_t, 3> workgroup_size;
      for (auto it : llvm::enumerate(func->getAttr("llvmgpu_workgroup_size")
                                         .cast<DenseIntElementsAttr>()
                                         .getIntValues())) {
        workgroup_size[it.index()] = it.value().getZExtValue();
      }
      workgroupSizes.push_back(workgroup_size);
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
      std::string targetChip = "sm_35";
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
    // final flatbuffer.
    std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
    if (dumpPtx) {
      llvm::dbgs() << targetISA;
    }
    auto ptxCudeRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetISA.c_str()),
        targetISA.size());

    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_CUDABlockSizeDef_vec_start(builder);
    auto blockSizes = workgroupSizes.begin();
    for (auto shader : entryPointNames) {
      iree_CUDABlockSizeDef_vec_push_create(builder, (*blockSizes)[0],
                                            (*blockSizes)[1], (*blockSizes)[2]);
      ++blockSizes;
    }
    auto blockSizesRef = iree_CUDABlockSizeDef_vec_end(builder);

    iree_CUDAExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_CUDAExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_CUDAExecutableDef_ptx_image_add(builder, ptxCudeRef);
    iree_CUDAExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.sym_name(),
        executableBuilder.getStringAttr("PTXE"),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.mime_typeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

 private:
  CUDATargetOptions options_;
};

void registerCUDATargetBackends(
    std::function<CUDATargetOptions()> queryOptions) {
  getCUDATargetOptionsFromFlags();
  static TargetBackendRegistration registration("cuda", [=]() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXAsmPrinter();
    return std::make_unique<CUDATargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
