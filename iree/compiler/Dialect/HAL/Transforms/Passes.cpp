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

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  Option<bool> serializeExecutables{
      *this, "serialize-executables",
      llvm::cl::desc("Whether to serialize hal.executable.target ops to "
                     "hal.executable.binary ops."),
      llvm::cl::init(true)};
};

}  // namespace

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   TargetOptions targetOptions,
                                   const TransformOptions &transformOptions) {
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createMaterializeInterfacesPass(targetOptions));

  // TODO(#1036): when dynamic pass registration is supported we can just
  // directly call TargetBackend::buildTranslationPassPipeline function. For now
  // we need to run each backend translation in isolation and we do that within
  // this pass.
  passManager.addPass(createTranslateExecutablesPass(targetOptions));

  // After all executables are translated we allow the backends to link them
  // together. For example, the LLVM AOT backend may combine all executable
  // targets for the same architecture into a single executable and link it as
  // a shared library.
  passManager.addPass(createLinkExecutablesPass(targetOptions));

  passManager.addPass(createConvertFlowToHALPass());

  // Phase ordering note: Before this pass, functions signatures will be based
  // on explicit shape types (such as ranked_shape). After this pass, these
  // composite types will be expanded to primitives (i.e. one 'index' for each
  // dynamic dim in the case of ranked_shape).
  passManager.addPass(Shape::createExpandFunctionRankedShapeDimsPass());

  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // For each exported function, processes the reflection metadata and
  // generates public ABI wrappers for various calling conventions.
  // Phase ordering note: This operates on functions whose signatures have
  // been expanded to primitives.
  passManager.addPass(createPublicABIGenerationPass());

  // Gather cachable resources such as executables and descriptor sets and
  // cache them at initialization-time.
  passManager.addPass(createMaterializeResourceCachesPass(targetOptions));

  // Inline hal.device.switch ops and memoize their queries such that we can
  // better CSE/fold dispatch logic.
  passManager.addPass(createInlineDeviceSwitchesPass());
  passManager.addPass(createMemoizeDeviceQueriesPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // TODO(#1036): run this once per hal.executable.target in a nested pass
  // manager so that we have as many passes as hal.executable.target ops.
  if (transformOptions.serializeExecutables) {
    passManager.addPass(createSerializeExecutablesPass(targetOptions));
    // NOTE: symbol DCE will destroy executable target contents, so only run it
    // if we serialized things.
    passManager.addPass(createSymbolDCEPass());
  }
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   TargetOptions targetOptions) {
  TransformOptions transformOptions;
  buildHALTransformPassPipeline(passManager, targetOptions, transformOptions);
}

void registerHALTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions>(
      "iree-hal-transformation-pipeline",
      "Runs the full IREE HAL dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildHALTransformPassPipeline(passManager, getTargetOptionsFromFlags(),
                                      transformOptions);
      });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
