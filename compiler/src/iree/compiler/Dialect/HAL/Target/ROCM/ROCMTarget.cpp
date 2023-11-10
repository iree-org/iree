// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/ROCM/ROCMTarget.h"

#include <mutex>

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/rocm_executable_def_builder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

static llvm::cl::opt<std::string>
    clROCMTargetChip("iree-rocm-target-chip",
                     llvm::cl::desc("ROCm target Chip"),
                     llvm::cl::init("gfx908"));

static llvm::cl::opt<bool>
    clROCMLinkBC("iree-rocm-link-bc",
                 llvm::cl::desc("Whether to try Linking to AMD Bitcodes"),
                 llvm::cl::init(false));

static llvm::cl::opt<std::string> clROCMBitcodeDir(
    "iree-rocm-bc-dir", llvm::cl::desc("Directory of ROCM Bitcode"),
    llvm::cl::init(mlir::iree_compiler::findPlatformLibDirectory("rocm")));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

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
class ROCMTargetBackend final : public TargetBackend {
public:
  std::string name() const override { return "rocm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerROCDLDialectTranslation(registry);
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
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
    if (variantOp.isExternal())
      return;

    buildLLVMGPUCodegenPassPipeline(passManager, true);
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

    ModuleOp innerModuleOp = variantOp.getInnerModule();

    // Remove all the functions that are not part of the ROCM kernel.
    // TODO: Find a better solution to handle this.
    auto illegalFuncOps = llvm::to_vector(innerModuleOp.getOps<func::FuncOp>());
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
    for (auto op : variantOp.getExportOps()) {
      exportOps[op.getSymName()] = op;
    }
    std::vector<std::array<int32_t, 3>> workgroupSizes;
    SmallVector<uint32_t> workgroupLocalMemories;
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      int32_t flatWgSize = 1;
      auto *llvmFunc = llvmModule->getFunction(func.getName());
      if (llvmFunc->isDeclaration())
        continue;
      std::array<int32_t, 3> workgroupSize;
      auto exportOp = exportOps[func.getName()];
      if (std::optional<ArrayAttr> workgroupSizeAttr =
              exportOp.getWorkgroupSize()) {
        for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
          workgroupSize[it.index()] = it.value().cast<IntegerAttr>().getInt();
          flatWgSize *= it.value().cast<IntegerAttr>().getInt();
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
      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-workgroup-size(1, 256) attribute.
      // 3. Insert amdgpu-implicitarg-num-bytes=56 (which must be set on OpenCL
      // and HIP kernels per Clang)
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      std::string wgSizeRange = std::string("1, ") + std::to_string(flatWgSize);
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", wgSizeRange);
    }

    std::unique_ptr<llvm::TargetMachine> targetMachine;
    {
      llvm::Triple triple("amdgcn--amdhsa-amdgiz");
      std::string targetChip = clROCMTargetChip;
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
    iree_hal_rocm_ExecutableDef_start_as_root(builder);

    // Link module to Device Library
    std::string rocmBitcodeDir = clROCMBitcodeDir;
    if (clROCMLinkBC) {
      if (clROCMBitcodeDir.empty()) {
        return variantOp.emitError()
               << "cannot find ROCM bitcode files. Check your installation "
                  "consistency and in the worst case, set --iree-rocm-bc-dir= "
                  "to an explicit location on your system.";
      }
      linkROCDLIfNecessary(llvmModule.get(), clROCMTargetChip, rocmBitcodeDir);
    }

    // Serialize hsaco kernel into the binary that we will embed in the
    // final FlatBuffer.
    std::string targetObj = translateModuleToObj(*llvmModule, *targetMachine);
    std::string targetHSACO =
        createHsaco(variantOp.getLoc(), targetObj, libraryName);
    if (targetHSACO.empty()) {
      return failure();
    }

    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath(options.dumpBinariesPath, options.dumpBaseName,
                     variantOp.getName(), ".hsaco", targetHSACO);
    }

    auto hsacoRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetHSACO.c_str()),
        targetHSACO.size());

    auto entryPointNames = llvm::map_to_vector<8>(
        variantOp.getBlock()
            .getOps<iree_compiler::IREE::HAL::ExecutableExportOp>(),
        [&](auto op) { return op.getName(); });
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
    iree_hal_rocm_ExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));

    if (!options.dumpIntermediatesPath.empty()) {
      std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
      dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                     variantOp.getName(), ".rocmasm", targetISA);
    }

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

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    // Add some configurations to the `hal.executable.target` attribute.
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(StringAttr::get(context, name), value);
    };
    // Set target arch
    addConfig("target_arch", StringAttr::get(context, clROCMTargetChip));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("rocm"), b.getStringAttr("rocm-hsaco-fb"),
        configAttr);
  }
};

void registerROCMTargetBackends() {
  // #hal.device.target<"rocm", ...
  // #hal.executable.target<"rocm", ...
  static iree_compiler::IREE::HAL::TargetBackendRegistration registration(
      "rocm", [=]() {
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUAsmPrinter();
        return std::make_shared<ROCMTargetBackend>();
      });
}

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
