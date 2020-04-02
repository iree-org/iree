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

// TODO(benvanik): add flags.
// static llvm::cl::OptionCategory halVMLAOptionsCategory(
//     "IREE VMLA backend options");

VMLATargetOptions getVMLATargetOptionsFromFlags() {
  VMLATargetOptions targetOptions;
  // TODO(benvanik): flags.
  return targetOptions;
}

LogicalResult translateToVMLAExecutable(
    IREE::HAL::ExecutableOp executableOp,
    ExecutableTargetOptions executableOptions,
    VMLATargetOptions targetOptions) {
  // Clone the module containing the things we want to translate. We do this so
  // that multiple targets can pull from the same source without conflicting.
  auto sourceOp = executableOp.getSourceOp().clone();
  auto sourceOpErase = llvm::make_scope_exit([&]() { sourceOp.erase(); });
  auto flowExecutableOp =
      *sourceOp.getInnerModule().getOps<IREE::Flow::ExecutableOp>().begin();
  auto innerModuleOp = flowExecutableOp.getInnerModule();

  // Markup all entry points as module exports.
  // TODO(benvanik): this won't be required when replaced with sym_visibility.
  for (auto funcOp : innerModuleOp.getOps<FuncOp>()) {
    if (SymbolTable::getSymbolVisibility(funcOp) ==
        SymbolTable::Visibility::Public) {
      funcOp.setAttr("iree.module.export",
                     UnitAttr::get(innerModuleOp.getContext()));
    }
  }

  // Convert to VMLA.
  PassManager conversionPassManager(innerModuleOp.getContext());
  applyPassManagerCLOptions(conversionPassManager);
  IREE::VMLA::buildVMLATransformPassPipeline(conversionPassManager);
  IREE::VM::buildVMTransformPassPipeline(conversionPassManager);
  if (failed(conversionPassManager.run(innerModuleOp))) {
    return innerModuleOp.emitError() << "failed to run conversion passes";
  }

  // Serialize the VM module to bytes.
  std::string byteStreamValue;
  llvm::raw_string_ostream byte_stream(byteStreamValue);
  IREE::VM::BytecodeTargetOptions bytecodeOptions;
  if (failed(translateModuleToBytecode(innerModuleOp, bytecodeOptions,
                                       byte_stream))) {
    return innerModuleOp.emitError()
           << "failed to serialize converted VM module";
  }

  // Pack the executable definition and get the bytes with the proper header.
  // The header is used to verify the contents at runtime.
  ::flatbuffers::FlatBufferBuilder fbb;
  iree::VMLAExecutableDefT vmlaExecutableDef;
  vmlaExecutableDef.bytecode_module.resize(byteStreamValue.size());
  std::memcpy(vmlaExecutableDef.bytecode_module.data(), byteStreamValue.data(),
              byteStreamValue.size());
  auto executableOffset =
      iree::VMLAExecutableDef::Pack(fbb, &vmlaExecutableDef);
  iree::FinishVMLAExecutableDefBuffer(fbb, executableOffset);
  std::vector<uint8_t> bytes;
  bytes.resize(fbb.GetSize());
  std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

  // Add the binary data to the target executable.
  OpBuilder targetBuilder = OpBuilder::atBlockEnd(&executableOp.getBlock());
  targetBuilder.setInsertionPoint(&executableOp.getBlock().back());
  auto binaryOp = targetBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      executableOp.getLoc(),
      static_cast<uint32_t>(IREE::HAL::ExecutableFormat::VMLA),
      std::move(bytes));
  OpBuilder binaryBuilder(&binaryOp.getBlock().back());
  auto vmModuleOp = *innerModuleOp.getOps<IREE::VM::ModuleOp>().begin();
  binaryBuilder.clone(*vmModuleOp);
  return success();
}

static ExecutableTargetRegistration targetRegistration(
    "vmla", +[](IREE::HAL::ExecutableOp executableOp,
                ExecutableTargetOptions executableOptions) {
      return translateToVMLAExecutable(executableOp, executableOptions,
                                       getVMLATargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
