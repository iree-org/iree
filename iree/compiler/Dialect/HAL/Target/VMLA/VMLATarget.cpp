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

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "iree/schemas/vmla_executable_def_generated.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
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

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpPassManager &passManager) override {
    IREE::VMLA::buildVMLATransformPassPipeline(passManager);

    // TODO(#614): remove this when the std->vm conversion isn't looking for
    // iree.module.export.
    passManager.addPass(IREE::VM::createMarkPublicSymbolsExportedPass());

    IREE::VM::buildVMTransformPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Serialize the VM module to bytes.
    std::string byteStreamValue;
    llvm::raw_string_ostream byte_stream(byteStreamValue);
    IREE::VM::BytecodeTargetOptions bytecodeOptions;
    if (failed(translateModuleToBytecode(targetOp.getInnerModule(),
                                         bytecodeOptions, byte_stream))) {
      return targetOp.emitError() << "failed to serialize converted VM module";
    }

    // Pack the executable definition and get the bytes with the proper header.
    // The header is used to verify the contents at runtime.
    ::flatbuffers::FlatBufferBuilder fbb;
    iree::VMLAExecutableDefT vmlaExecutableDef;
    vmlaExecutableDef.bytecode_module.resize(byteStreamValue.size());
    std::memcpy(vmlaExecutableDef.bytecode_module.data(),
                byteStreamValue.data(), byteStreamValue.size());
    auto executableOffset =
        iree::VMLAExecutableDef::Pack(fbb, &vmlaExecutableDef);
    iree::FinishVMLAExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::VMLA),
        std::move(bytes));

    return success();
  }

  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
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
