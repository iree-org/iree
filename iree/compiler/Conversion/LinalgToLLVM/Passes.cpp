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

#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

void addLinalgToLLVMPasses(OpPassManager &passManager) {
  // Linalg -> SCF
  passManager.addPass(createConvertLinalgToLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // SCF -> STD
  passManager.addPass(createLowerToCFGPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createResolveShapeOpsPass());
  // Convert ExecuableOp entry function to use memref arguments.
  passManager.addPass(createHALInterfaceToMemrefArgumentsPass());

  // (Linalg, STD) -> LLVM
  // OpPassManager& llvmPassManager = passManager.nest<ModuleOp>();
  passManager.addPass(createConvertLinalgToLLVMPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildLLVMTransformPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createInlinerPass());

  // HLO -> Linalg on buffers.
  passManager.addPass(createDecomposeHLOClampPass());
  addHLOToLinalgOnBuffersPasses(passManager);

  // Linalg -> LLVM passes.
  addLinalgToLLVMPasses(passManager);
}

static PassPipelineRegistration<> linalgLLVMVPipeline(
    "iree-codegen-linalg-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager);
    });

static PassPipelineRegistration<> hloToLinalgLLVMVPipeline(
    "iree-codegen-hlo-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager);
    });

}  // namespace iree_compiler
}  // namespace mlir
