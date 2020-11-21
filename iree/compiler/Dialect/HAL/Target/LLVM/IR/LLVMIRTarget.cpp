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

#include "iree/compiler/Dialect/HAL/Target/LLVM/IR/LLVMIRTarget.h"

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/llvmir_executable_def_builder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMIRTargetBackend final : public LLVMBaseTargetBackend {
 public:
  explicit LLVMIRTargetBackend(LLVMTargetOptions options)
      : LLVMBaseTargetBackend(options) {}

  // NOTE: we could vary these based on the options, such as by arch/etc.
  std::string name() const override { return "llvm_ir"; }
  std::string filter_pattern() const override { return "llvm-ir*"; }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation to LLVM in a separate context to avoid
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
      return targetOp.emitError("Failed to translate executable to LLVM IR");
    }

    // LLVMIR opt passes.
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      targetOp.emitError("Can't create target machine for target triple: " +
                         options_.targetTriple);
      return failure();
    }
    LogicalResult translationResult =
        runLLVMIRPasses(options_, targetMachine.get(), llvmModule.get());
    if (failed(translationResult)) {
      return targetOp.emitError(
          "Can't build LLVMIR opt passes for ExecutableOp module");
    }

    // Serialize LLVM module directly into flatbuffer.
    FlatbufferBuilder builder;
    auto bitcodeModuleRef = builder.streamUint8Vec([&](raw_ostream &stream) {
      llvmModule->print(stream, nullptr);
      return true;
    });

    auto entryPointsRef = builder.createStringVec(llvm::map_range(
        targetOp.getBlock().getOps<ExecutableEntryPointOp>(),
        [&](ExecutableEntryPointOp op) { return op.sym_name(); }));

    iree_LLVMIRExecutableDef_start_as_root(builder);
    iree_LLVMIRExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_LLVMIRExecutableDef_bitcode_module_add(builder, bitcodeModuleRef);
    iree_LLVMIRExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::LLVM),
        builder.getBufferAttr(executableBuilder.getContext()));
    return success();
  }
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
