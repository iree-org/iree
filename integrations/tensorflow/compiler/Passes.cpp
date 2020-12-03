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

#include "integrations/tensorflow/compiler/Passes.h"

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/dialect.h"
#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "iree/compiler/Dialect/Shape/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace TF {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<tf_tensorlist::TFTensorListDialect,
                  tf_strings::TFStringsDialect>();
}

// All IREE-specific passes that lower TF representations before reaching the
// IREE core should go here.
void buildTFImportPassPipeline(OpPassManager &pm) {
  //----------------------------------------------------------------------------
  // Lowering shape-related constructs
  //----------------------------------------------------------------------------
  pm.addPass(Shape::createConvertHLOToShapePass());
  // TODO(GH-2277): Lower HLO shape constraints instead of eliding them here.
  pm.addPass(createRemoveShapeConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(Shape::createConvertShapeToShapexPass());
  pm.addPass(createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Lowering TensorList-related parts of tf dialect to tf_tensorlist dialect
  //----------------------------------------------------------------------------
  pm.addPass(tf_tensorlist::createConvertTFToTFTensorListPass());
  pm.addPass(tf_strings::createConvertTFToTFStringsPass());

  //----------------------------------------------------------------------------
  // Lowering tf_saved_model dialect to IREE dialects
  //----------------------------------------------------------------------------
  // First, eliminate tf_saved_model.global_tensor's and corresponding
  // tf_saved_model.bound_input's.
  pm.addPass(createLowerGlobalTensorsPass());

  // Lower exported functions.
  //
  // This pass must run second because:
  // - It assumes that tf_saved_model.bound_inputs have been eliminated
  // - It removes tf_saved_model.semantics from the module, which we can only
  //   do at the very end.
  pm.addPass(createLowerExportedFunctionsPass());

  //----------------------------------------------------------------------------
  // Ensure that all Tensorflow has been legalized away
  //----------------------------------------------------------------------------
  pm.addPass(createStripModuleMetadataPass());
  pm.nest<ModuleOp>().addPass(createStripFunctionMetadataPass());
  pm.addPass(createVerifyFullyConvertedPass());
}

void registerTFImportPassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "iree-tf-import-pipeline",
      "Run IREE-specific passes for importing TF code into IREE",
      [](OpPassManager &passManager) {
        buildTFImportPassPipeline(passManager);
      });
}

}  // namespace TF
}  // namespace iree_compiler
}  // namespace mlir
