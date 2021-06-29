// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/ROCM/ROCMTarget.h"

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/rocm_executable_def_builder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

ROCMTargetOptions getROCMTargetOptionsFromFlags() {
  ROCMTargetOptions targetOptions;
  static llvm::cl::opt<std::string> clROCMTargetChip(
      "iree-rocm-target-chip", llvm::cl::desc("ROCm target Chip"),
      llvm::cl::init("gfx908"));

  static llvm::cl::opt<bool> clROCMLinkBC(
      "iree-rocm-link-bc",
      llvm::cl::desc("Whether to try Linking to AMD Bitcodes"),
      llvm::cl::init(false));

  targetOptions.ROCMTargetChip = clROCMTargetChip;
  targetOptions.ROCMLinkBC = clROCMLinkBC;

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
                                      llvm::CGFT_ObjectFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

class ROCMTargetBackend final : public TargetBackend {
 public:
  ROCMTargetBackend(ROCMTargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "rocm"; }
  std::string filter_pattern() const override { return "rocm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerROCDLDialectTranslation(registry);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildLLVMGPUTransformPassPipeline(passManager, true);
  }

  LogicalResult serializeExecutable(
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<iree_compiler::IREE::HAL::ExecutableOp>()
            .getName()
            .str();

    ModuleOp innerModuleOp = variantOp.getInnerModule();

    // Remove all the functions that are not part of the ROCM kernel.
    // TODO: Find a better solution to handle this.
    auto illegalFuncOps = llvm::to_vector<4>(innerModuleOp.getOps<FuncOp>());
    for (auto funcOp : illegalFuncOps) {
      funcOp.erase();
    }
    auto halInterfaceOps = llvm::to_vector<1>(
        innerModuleOp.getOps<iree_compiler::IREE::HAL::InterfaceOp>());
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
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      auto *llvmFunc = llvmModule->getFunction(func.getName());
      if (llvmFunc->isDeclaration()) continue;
      std::array<int32_t, 3> workgroup_size;
      for (auto it : llvm::enumerate(func->getAttr("llvmgpu_workgroup_size")
                                         .cast<DenseIntElementsAttr>()
                                         .getIntValues())) {
        workgroup_size[it.index()] = it.value().getZExtValue();
      }
      workgroupSizes.push_back(workgroup_size);
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
    }

    std::unique_ptr<llvm::TargetMachine> targetMachine;
    {
      llvm::Triple triple("amdgcn--amdhsa-amdgiz");
      std::string targetChip = options_.ROCMTargetChip;
      std::string error;
      const llvm::Target *target =
          llvm::TargetRegistry::lookupTarget("", triple, error);
      if (target == nullptr) {
        return variantOp.emitError() << "cannot initialize target triple";
      }
      targetMachine.reset(
          target->createTargetMachine(triple.str(), targetChip, {}, {}, {}));
      if (targetMachine == nullptr) {
        return variantOp.emitError() << "cannot initialize target machine";
      }
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());

    iree_compiler::FlatbufferBuilder builder;
    iree_ROCMExecutableDef_start_as_root(builder);

    // Link module to Device Library
    if (options_.ROCMLinkBC)
      LinkROCDLIfNecessary(llvmModule.get(), options_.ROCMTargetChip);

    // Serialize hsaco kernel into the binary that we will embed in the
    // final flatbuffer.
    std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
    std::string targetHSACO = createHsaco(targetISA, libraryName);
    auto hsacoRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetHSACO.c_str()),
        targetHSACO.size());

    auto entryPointNames = llvm::to_vector<8>(llvm::map_range(
        variantOp.getBlock()
            .getOps<iree_compiler::IREE::HAL::ExecutableEntryPointOp>(),
        [&](auto op) { return op.getName(); }));
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_ROCMBlockSizeDef_vec_start(builder);
    auto blockSizes = workgroupSizes.begin();
    for (auto shader : entryPointNames) {
      iree_ROCMBlockSizeDef_vec_push_create(builder, (*blockSizes)[0],
                                            (*blockSizes)[1], (*blockSizes)[2]);
      ++blockSizes;
    }
    auto blockSizesRef = iree_ROCMBlockSizeDef_vec_end(builder);

    iree_ROCMExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_ROCMExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_ROCMExecutableDef_hsaco_image_add(builder, hsacoRef);
    iree_ROCMExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.sym_name(),
        executableBuilder.getStringAttr("HSACO"),
        builder.getBufferAttr(executableBuilder.getContext()));

    return success();
  }

 private:
  ROCMTargetOptions options_;
};

void registerROCMTargetBackends(
    std::function<ROCMTargetOptions()> queryOptions) {
  getROCMTargetOptionsFromFlags();
  static iree_compiler::IREE::HAL::TargetBackendRegistration registration(
      "rocm", [=]() {
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUAsmPrinter();
        return std::make_unique<ROCMTargetBackend>(queryOptions());
      });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
