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
#include "iree/compiler/Translation/LLVM/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/XLAToLinalg/LinalgTensorToBuffer.h"
#include "iree/compiler/Translation/XLAToLinalg/XLAToLinalg.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static void addLLVMTransformPassPipeline(OpPassManager& pm) {
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

static PassPipelineRegistration<> xlaHLOOToLLVMPassPipeline(
    "iree-llvm-transformation-pipeline",
    "Runs the progressive lowering pipeline from XLA-HLO to LLVMIR",
    [](OpPassManager& passManager) {
      addLLVMTransformPassPipeline(passManager);
    });

void buildLLVMTransformPassPipeline(PassManager& conversionPassManager) {
  addLLVMTransformPassPipeline(conversionPassManager);
}

}  // namespace iree_compiler
}  // namespace mlir
