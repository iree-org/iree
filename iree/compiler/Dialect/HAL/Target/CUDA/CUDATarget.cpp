// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"

#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Target/LLVMIR/Export.h"

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

class CUDATargetBackend final : public TargetBackend {
 public:
  CUDATargetBackend(CUDATargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "cuda"; }
  std::string filter_pattern() const override { return "cuda"; }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildNVVMTransformPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;
    mlir::registerLLVMDialectTranslation(*targetOp.getContext());

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        targetOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    ModuleOp innerModuleOp = targetOp.getInnerModule();

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
      return targetOp.emitError() << "failed to translate the MLIR LLVM "
                                     "dialect to the native llvm::Module";
    }
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      auto *llvmFunc = llvmModule->getFunction(func.getName());

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
        return targetOp.emitError() << "cannot initialize target triple";
      }
      targetMachine.reset(target->createTargetMachine(triple.str(), targetChip,
                                                      features, {}, {}));
      if (targetMachine == nullptr) {
        return targetOp.emitError() << "cannot initialize target machine";
      }
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());

    std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
    // Serialize cuda kernel into the binary that we will embed in the
    // final flatbuffer.
    FlatbufferBuilder builder;
    auto ptxCudeRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetISA.c_str()),
        targetISA.size());

    auto entryPointNames = llvm::to_vector<8>(
        llvm::map_range(targetOp.getBlock().getOps<ExecutableEntryPointOp>(),
                        [&](auto op) { return op.getName(); }));
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_CUDABlockSizeDef_vec_start(builder);
    for (auto shader : entryPointNames) {
      // Hard-coded workgroup size.
      iree_CUDABlockSizeDef_vec_push_create(builder, 1, 1, 1);
    }
    auto blockSizesRef = iree_CUDABlockSizeDef_vec_end(builder);

    iree_CUDAExecutableDef_start_as_root(builder);
    iree_CUDAExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_CUDAExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_CUDAExecutableDef_ptx_image_add(builder, ptxCudeRef);
    iree_CUDAExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::CUDA),
        builder.getBufferAttr(executableBuilder.getContext()));

    return success();
  }

  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) override {
    // For now we are not tiling and just dispatch everything as 1,1,1.
    auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
    return {constantOne, constantOne, constantOne};
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
