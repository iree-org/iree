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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/Module.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

LLVMTargetOptions getLLVMTargetOptionsFromFlags() {
  LLVMTargetOptions targetOptions;
  return targetOptions;
}

// TODO(ataei) Move this to utils ?
// Returns a list of entry point names matching the expected export ordinals.
static std::vector<std::string> populateEntryPointNames(
    IREE::Flow::ExecutableOp executableOp) {
  std::vector<std::string> entryPointNames;
  for (auto& op : executableOp.getBlock().getOperations()) {
    if (auto entryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
      entryPointNames.push_back(std::string(entryOp.function_ref()));
    }
  }
  return entryPointNames;
}

// Adds a sequence of passess to a given pass manager that progressively lower
// from HLO to LLVM throught linalg dialect.
void buildLLVMTransformPassPipeline(OpPassManager& pm) {
  // HLO -> Linalg on buffers.
  addXLAToLinalgOnBuffersPasses(pm);

  // Linalg -> Loops
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Loops -> STD
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // STD -> LLVM
  pm.addPass(
      createLowerToLLVMPass(/*useAlloca*/ false, /*emitCWrappers*/ true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

LogicalResult translateToLLVMExecutable(
    IREE::HAL::ExecutableOp executableOp,
    ExecutableTargetOptions executableOptions,
    LLVMTargetOptions targetOptions) {
  // Clone the module containing the things we want to translate. We do this so
  // that multiple targets can pull from the same source without conflicting.
  auto sourceOp = executableOp.getSourceOp().clone();
  auto sourceOpErase =
      llvm::make_scope_exit([&sourceOp]() { sourceOp.erase(); });
  auto flowExecutableOp =
      *sourceOp.getInnerModule().getOps<IREE::Flow::ExecutableOp>().begin();
  auto moduleOp = flowExecutableOp.getInnerModule();
  if (failed(makeLegacyExecutableABI(sourceOp))) {
    return failure();
  }

  // Lower module to LLVM Dialect.
  PassManager conversionPassManager(moduleOp.getContext());
  buildLLVMTransformPassPipeline(conversionPassManager);
  if (failed(conversionPassManager.run(moduleOp)))
    return moduleOp.emitError()
           << "failed to run IREE -> LLVM conversion passes";

  // At this moment we are leaving MLIR LLVM dialect land translating module
  // into target independent LLVMIR.
  auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp);

  // Serialize LLVM module.
  std::string bufferString;
  llvm::raw_string_ostream ostream(bufferString);
  llvmModule->print(ostream, nullptr);
  ostream.flush();

  // Creates executable bytes.
  iree::LLVMIRExecutableDefT llvmIrExecutableDef;
  llvmIrExecutableDef.llvmir_module = {bufferString.begin(),
                                       bufferString.end()};
  llvmIrExecutableDef.entry_points = populateEntryPointNames(flowExecutableOp);
  ::flatbuffers::FlatBufferBuilder fbb;
  auto executableOffset =
      iree::LLVMIRExecutableDef::Pack(fbb, &llvmIrExecutableDef);
  iree::FinishLLVMIRExecutableDefBuffer(fbb, executableOffset);
  std::vector<uint8_t> bytes;
  bytes.resize(fbb.GetSize());
  std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

  // Add the binary data to the target executable.
  OpBuilder targetBuilder(&executableOp.getBlock());
  targetBuilder.setInsertionPoint(&executableOp.getBlock().back());
  auto binaryOp = targetBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      executableOp.getLoc(),
      static_cast<uint32_t>(IREE::HAL::ExecutableFormat::LLVM),
      std::move(bytes));
  OpBuilder binaryBuilder(&binaryOp.getBlock().back());
  binaryBuilder.clone(*moduleOp.getOperation());
  return success();
}

static ExecutableTargetRegistration targetRegistration(
    "llvm-ir", +[](IREE::HAL::ExecutableOp executableOp,
                   ExecutableTargetOptions executableOptions) {
      return translateToLLVMExecutable(executableOp, executableOptions,
                                       getLLVMTargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
