// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/LLVM/AOT/LLVMAOTTarget.h"

#include <cstdlib>

#include "iree/compiler/Dialect/HAL/Target/LLVM/AOT/LLVMAOTTargetLinker.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/schemas/dylib_executable_def_generated.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMAOTTargetBackend final : public LLVMBaseTargetBackend {
 public:
  explicit LLVMAOTTargetBackend(LLVMTargetOptions options)
      : LLVMBaseTargetBackend(options) {}

  // NOTE: we could vary these based on the options, such as by arch/etc.
  std::string name() const override { return "llvm_aot"; }
  std::string filter_pattern() const override { return "dylib*"; }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // Remove all private functions, e.g tile size calcuations.
    SmallVector<FuncOp, 4> nonPublicFn;
    for (auto func : targetOp.getInnerModule().getOps<FuncOp>()) {
      if (SymbolTable::getSymbolVisibility(func) !=
          SymbolTable::Visibility::Public) {
        nonPublicFn.push_back(func);
      }
    }
    for (auto func : nonPublicFn) {
      func.erase();
    }

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule =
        mlir::translateModuleToLLVMIR(targetOp.getInnerModule(), context);
    if (!llvmModule) {
      return failure();
    }

    iree::DyLibExecutableDefT dyLibExecutableDef;
    // Create invocation function an populate entry_points.
    auto entryPointOps = targetOp.getBlock().getOps<ExecutableEntryPointOp>();

    for (auto entryPointOp : entryPointOps) {
      dyLibExecutableDef.entry_points.push_back(
          std::string(entryPointOp.sym_name()));
    }

    // LLVMIR opt passes.
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      targetOp.emitError("Can't create target machine for target triple: " +
                         options_.targetTriple);
      return failure();
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());

    if (failed(
            runLLVMIRPasses(options_, targetMachine.get(), llvmModule.get()))) {
      return targetOp.emitError(
          "Can't build LLVMIR opt passes for ExecutableOp module");
    }

    std::string objData;
    if (failed(runEmitObjFilePasses(targetMachine.get(), llvmModule.get(),
                                    &objData))) {
      return targetOp.emitError("Can't compile LLVMIR module to an obj");
    }

    std::string sharedLibData;
    const char *linkerToolPath = std::getenv("IREE_LLVMAOT_LINKER_PATH");
    if (linkerToolPath != nullptr) {
      auto sharedLibDataStatus = linkLLVMAOTObjects(linkerToolPath, objData);
      if (!sharedLibDataStatus.ok()) {
        return targetOp.emitError(
            "Can't link executable and generate target dylib, using linker "
            "toolchain: '" +
            std::string(linkerToolPath) + "'");
      }
      sharedLibData = sharedLibDataStatus.value();
    } else {
      auto sharedLibDataStatus = linkLLVMAOTObjectsWithLLDElf(objData);
      if (!sharedLibDataStatus.ok()) {
        return targetOp.emitError(
            "Can't link executable and generate target dylib using "
            "lld::elf::link");
      }
      sharedLibData = sharedLibDataStatus.value();
    }
    dyLibExecutableDef.library_embedded = {sharedLibData.begin(),
                                           sharedLibData.end()};

    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::DyLibExecutableDef::Pack(fbb, &dyLibExecutableDef);
    iree::FinishDyLibExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::DyLib),
        std::move(bytes));

    return success();
  }
};

void registerLLVMAOTTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("dylib-llvm-aot", [=]() {
#define INIT_LLVM_TARGET(TargetName)        \
  LLVMInitialize##TargetName##Target();     \
  LLVMInitialize##TargetName##TargetMC();   \
  LLVMInitialize##TargetName##TargetInfo(); \
  LLVMInitialize##TargetName##AsmPrinter(); \
  LLVMInitialize##TargetName##AsmParser();
    INIT_LLVM_TARGET(X86)
    INIT_LLVM_TARGET(ARM)
    INIT_LLVM_TARGET(AArch64)
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
