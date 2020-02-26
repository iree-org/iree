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

#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"
#include "iree/compiler/Translation/XLAToLinalg/Passes.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
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

// Adds a sequence of passess to a given pass manager that progressively lower
// from HLO to LLVM throught linalg dialect.
void buildLLVMTransformPassPipeline(OpPassManager& pm) {
  // HLO -> Linalg.
  pm.addPass(createXLAToLinalgPass());

  // Convert Tensors to memrefs.
  pm.addPass(createLinalgTensorToBufferConversionPass());

  // Linalg -> Loops
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Loops -> STD
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // STD -> LLVM
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

LogicalResult translateToLLVMExecutable(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp,
    ExecutableTargetOptions executableOptions,
    LLVMTargetOptions targetOptions) {
  auto moduleOp = sourceOp.getInnerModule().clone();
  makeLegacyExecutableABI(sourceOp, moduleOp, targetOp);

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
  ::flatbuffers::FlatBufferBuilder fbb;
  auto executableOffset =
      iree::LLVMIRExecutableDef::Pack(fbb, &llvmIrExecutableDef);
  iree::FinishLLVMIRExecutableDefBuffer(fbb, executableOffset);
  std::vector<uint8_t> bytes;
  bytes.resize(fbb.GetSize());
  std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

  // Add the binary data to the target executable.
  OpBuilder targetBuilder(&targetOp.getBlock());
  targetBuilder.setInsertionPoint(&targetOp.getBlock().back());
  auto binaryOp = targetBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      targetOp.getLoc(),
      static_cast<uint32_t>(IREE::HAL::ExecutableFormat::LLVM),
      std::move(bytes));
  binaryOp.getBlock().getOperations().insert(
      Block::iterator(binaryOp.getBlock().back()), moduleOp);
  return success();
}

static ExecutableTargetRegistration targetRegistration(
    "llvm-ir",
    +[](IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp,
        ExecutableTargetOptions executableOptions) {
      return translateToLLVMExecutable(sourceOp, targetOp, executableOptions,
                                       getLLVMTargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
