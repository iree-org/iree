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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTarget.h"

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/schemas/dylib_executable_def_generated.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMAOTTargetBackend final : public TargetBackend {
 public:
  LLVMAOTTargetBackend(LLVMTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options, such as by arch/etc.
  std::string name() const override { return "dylib*"; }

  void buildTranslationPassPipeline(ExecutableTargetOp targetOp,
                                    OpPassManager& passManager) override {
    buildLLVMTransformPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder& executableBuilder) override {
    // LLVM is not thread safe and currently translation shares an LLVMContext.
    // Since we serialize executables from multiple threads we have to take a
    // global lock here.
    static llvm::sys::SmartMutex<true> mutex;
    llvm::sys::SmartScopedLock<true> lock(mutex);

    iree::DyLibExecutableDefT dyLibExecutableDef;

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule = mlir::translateModuleToLLVMIR(targetOp.getInnerModule());

    // Create invocation function an populate entry_points.
    auto executableOp = cast<ExecutableOp>(targetOp.getParentOp());
    auto entryPointOps =
        executableOp.getBlock().getOps<ExecutableEntryPointOp>();
    const bool addCInterface = true;
    for (auto entryPointOp : entryPointOps) {
      std::string funcName =
          addCInterface ? "_mlir_ciface_" + std::string(entryPointOp.sym_name())
                        : std::string(entryPointOp.sym_name());
      dyLibExecutableDef.entry_points.push_back(funcName);
      createLLVMInvocationFunc(funcName, llvmModule.get());
    }

    if (!llvmModule) {
      return failure();
    }

    // TODO(scotttodd): LLVM AOT compilation to
    //   dyLibExecutableDef.library_embedded.assign(...)

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

 private:
  LLVMTargetOptions options_;
};

void registerLLVMAOTTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("dylib-llvm-aot", [=]() {
    // Initalize registered targets.
    llvm::InitializeNativeTarget();
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
