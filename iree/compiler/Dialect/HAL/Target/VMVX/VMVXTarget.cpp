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

#include "iree/compiler/Dialect/HAL/Target/VMVX/VMVXTarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

VMVXTargetOptions getVMVXTargetOptionsFromFlags() {
  VMVXTargetOptions targetOptions;
  // TODO(benvanik): flags.
  return targetOptions;
}

class VMVXTargetBackend final : public TargetBackend {
 public:
  VMVXTargetBackend(VMVXTargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "vmvx"; }
  std::string filter_pattern() const override { return "vmvx"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VM::VMDialect, VMVX::VMVXDialect>();
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXTransformPassPipeline(passManager);

    OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
    // TODO(#614): remove this when the std->vm conversion isn't looking for
    // iree.module.export.
    nestedModulePM.addPass(IREE::VM::createMarkPublicSymbolsExportedPass());

    // TODO(benvanik): derive these from a vm target triple.
    auto vmOptions = IREE::VM::getTargetOptionsFromFlags();
    vmOptions.f32Extension = true;
    vmOptions.optimizeForStackSize = false;
    IREE::VM::buildVMTransformPassPipeline(nestedModulePM, vmOptions);
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return success();

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName = llvm::formatv("{0}_linked", name());
    auto linkedExecutableOp = builder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());

    // Add our VMVX hal.executable.target with an empty module.
    builder.setInsertionPointToStart(linkedExecutableOp.getBody());
    auto linkedTargetOp = builder.create<IREE::HAL::ExecutableTargetOp>(
        moduleOp.getLoc(), name(), filter_pattern());
    builder.setInsertionPoint(&linkedTargetOp.getBlock().back());
    auto linkedModuleOp = builder.create<ModuleOp>(moduleOp.getLoc());

    // Add an empty vm.module to that module (as our vm.funcs must live in it).
    builder.setInsertionPointToStart(linkedModuleOp.getBody());
    builder.create<IREE::VM::ModuleOp>(moduleOp.getLoc(), "linked_module");

    // Try linking together all executables in moduleOp.
    return linkExecutablesInto(
        moduleOp, sourceExecutableOps, linkedExecutableOp, linkedTargetOp,
        [](mlir::ModuleOp moduleOp) {
          return *moduleOp.getOps<IREE::VM::ModuleOp>().begin();
        },
        builder);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Serialize the VM module to bytes and embed it directly.
    SmallVector<char> moduleData;
    {
      IREE::VM::BytecodeTargetOptions bytecodeOptions;
      llvm::raw_svector_ostream stream(moduleData);
      if (failed(translateModuleToBytecode(targetOp.getInnerModule(),
                                           bytecodeOptions, stream))) {
        return targetOp.emitOpError()
               << "failed to serialize VM bytecode module";
      }
    }
    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(moduleData.size())},
                        IntegerType::get(executableBuilder.getContext(), 8)),
        std::move(moduleData));

    // Add the binary data to the target executable.
    // NOTE: this snapshots the flatbuffer builder data at the time it is called
    // and future changes to the target op will not be observed.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(),
        executableBuilder.getStringAttr("VMVX"), bufferAttr);
    return success();
  }

 private:
  VMVXTargetOptions options_;
};

void registerVMVXTargetBackends(
    std::function<VMVXTargetOptions()> queryOptions) {
  getVMVXTargetOptionsFromFlags();
  static TargetBackendRegistration registration("vmvx", [=]() {
    return std::make_unique<VMVXTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
