// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/VMLA/VMLATarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/vmla_executable_def_builder.h"
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

VMLATargetOptions getVMLATargetOptionsFromFlags() {
  VMLATargetOptions targetOptions;
  // TODO(benvanik): flags.
  return targetOptions;
}

class VMLATargetBackend final : public TargetBackend {
 public:
  VMLATargetBackend(VMLATargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "vmla"; }
  std::string filter_pattern() const override { return "vmla"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VM::VMDialect, VMLA::VMLADialect>();
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
    IREE::VMLA::buildVMLATransformPassPipeline(nestedModulePM);

    // TODO(#614): remove this when the std->vm conversion isn't looking for
    // iree.module.export.
    nestedModulePM.addPass(IREE::VM::createMarkPublicSymbolsExportedPass());

    IREE::VM::buildVMTransformPassPipeline(
        nestedModulePM, IREE::VM::getTargetOptionsFromFlags());
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return success();

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName = llvm::formatv("vmla_linked_{1}", name());
    auto linkedExecutableOp = builder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());

    // Add our VMLA hal.executable.target with an empty module.
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
    FlatbufferBuilder builder;
    iree_VMLAExecutableDef_start_as_root(builder);

    // Serialize the VM module to bytes directly into a flatbuffer.
    IREE::VM::BytecodeTargetOptions bytecodeOptions;
    auto dataRef = builder.streamUint8Vec([&](raw_ostream &stream) {
      return succeeded(translateModuleToBytecode(targetOp.getInnerModule(),
                                                 bytecodeOptions, stream));
    });
    if (!dataRef) {
      return targetOp.emitError() << "failed to serialize converted VM module";
    }

    // Pack the executable definition and get the bytes with the proper header.
    // The header is used to verify the contents at runtime.
    iree_VMLAExecutableDef_bytecode_module_add(builder, dataRef);
    iree_VMLAExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    // NOTE: this snapshots the flatbuffer builder data at the time it is called
    // and future changes will not be observed.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(),
        executableBuilder.getStringAttr("VMLA"),
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
  VMLATargetOptions options_;
};

void registerVMLATargetBackends(
    std::function<VMLATargetOptions()> queryOptions) {
  getVMLATargetOptionsFromFlags();
  static TargetBackendRegistration registration("vmla", [=]() {
    return std::make_unique<VMLATargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
