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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_PASSES_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_PASSES_H_

#include "iree_tf_compiler/dialect/tf_strings/conversion/convert_tf_strings_to_strings.h"
#include "iree_tf_compiler/dialect/tf_strings/conversion/convert_tf_to_tf_strings.h"
#include "iree_tf_compiler/dialect/tf_tensorlist/conversion/convert_tf_tensorlist_to_tensorlist.h"
#include "iree_tf_compiler/dialect/tf_tensorlist/conversion/convert_tf_to_tf_tensorlist.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Create a single pipeline that will run all the needed IREE-specific TF import
// passes in the right order.
void buildTFImportPassPipeline(OpPassManager &pm);

void registerTFImportPassPipeline();

//===----------------------------------------------------------------------===//
// IREE-specific Passes For TensorFlow Import
//===----------------------------------------------------------------------===//

// Converts the TF dialect to the XLA MHLO dialect.
std::unique_ptr<FunctionPass> createConvertToMHLOPass();

// In a module tagged with `tf_saved_model.semantics`, lowers
// `tf_saved_model.global_variable`'s to `flow.variable`'s.
//
// This pass should be run before adopting the exports, which transitions to
// a module that does not have `tf_saved_model.semantics`.
std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass();

// In a module tagged with `tf_saved_model.semantics`, lowers any tf_saved_model
// exported functions to IREE exported functions with appropriate reflection
// metadata.
std::unique_ptr<OperationPass<ModuleOp>> createLowerExportedFunctionsPass();

// Push resource casts forward to better propagate resource related shapes.
std::unique_ptr<OperationPass<ModuleOp>> createPropagateResourceCastsPass();

// Strips all TF-related attributes; none are needed by IREE.
std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass();
std::unique_ptr<OperationPass<FuncOp>> createStripFunctionMetadataPass();

// Validates whether any Tensorflow operations remain.
std::unique_ptr<OperationPass<FuncOp>> createVerifyFullyConvertedPass();

// Creates an IREE-specific variant of the upstream XLA LegalizeTF pass.
std::unique_ptr<OperationPass<FuncOp>> createIREEXLALegalizeTF();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerAllDialects(mlir::DialectRegistry &registry);

inline void registerAllPasses() {
  registerTFImportPassPipeline();

  createConvertToMHLOPass();
  createLowerGlobalTensorsPass();
  createLowerExportedFunctionsPass();
  createPropagateResourceCastsPass();
  createStripModuleMetadataPass();
  createStripFunctionMetadataPass();
  createVerifyFullyConvertedPass();

  tf_strings::createConvertTFToTFStringsPass();
  tf_strings::createConvertTFStringsToStringsPass();
  tf_tensorlist::createConvertTFTensorListToTensorListPass();
  tf_tensorlist::createConvertTFToTFTensorListPass();
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_PASSES_H_
