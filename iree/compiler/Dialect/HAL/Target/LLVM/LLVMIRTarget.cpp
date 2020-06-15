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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRTarget.h"

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMIRTargetBackend final : public TargetBackend {
 public:
  LLVMIRTargetBackend(LLVMTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options, such as by arch/etc.
  std::string name() const override { return "llvm-ir*"; }

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

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule = mlir::translateModuleToLLVMIR(targetOp.getInnerModule());

    // Create invocation function an populate entry_points.
    iree::LLVMIRExecutableDefT llvmIrExecutableDef;
    auto executableOp = cast<IREE::HAL::ExecutableOp>(targetOp.getParentOp());
    auto entryPointOps =
        executableOp.getBlock().getOps<IREE::HAL::ExecutableEntryPointOp>();
    const bool addCInterface = true;
    for (auto entryPointOp : entryPointOps) {
      std::string funcName =
          addCInterface ? "_mlir_ciface_" + std::string(entryPointOp.sym_name())
                        : std::string(entryPointOp.sym_name());
      llvmIrExecutableDef.entry_points.push_back(funcName);
      createLLVMInvocationFunc(funcName, llvmModule.get());
    }

    // LLVMIR opt passes.
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      targetOp.emitError("Can't create target machine for target triple: " +
                         options_.targetTriple);
      return failure();
    }
    if (failed(
            runLLVMIRPasses(options_, targetMachine.get(), llvmModule.get()))) {
      return targetOp.emitError(
          "Can't build LLVMIR opt passes for ExecutableOp module");
    }

    // Serialize LLVM module.
    std::string bufferString;
    llvm::raw_string_ostream ostream(bufferString);
    llvmModule->print(ostream, nullptr);
    ostream.flush();

    // Creates executable bytes.
    llvmIrExecutableDef.llvmir_module = {bufferString.begin(),
                                         bufferString.end()};

    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::LLVMIRExecutableDef::Pack(fbb, &llvmIrExecutableDef);
    iree::FinishLLVMIRExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::LLVM),
        std::move(bytes));

    return success();
  }

  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
      OpBuilder& builder) override {
    // For now we are not tiling and just dispatch everything as 1,1,1.
    auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
    return {constantOne, constantOne, constantOne};
  }

 private:
  LLVMTargetOptions options_;
};

void registerLLVMIRTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("llvm-ir", [=]() {
    // Initalize registered targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return std::make_unique<LLVMIRTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
