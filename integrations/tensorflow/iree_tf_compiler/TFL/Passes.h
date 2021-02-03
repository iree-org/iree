// Copyright 2021 Google LLC
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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASSES_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Create a single pipeline that will run all the needed IREE-specific TFL
// import passes in the right order.
void buildTFLImportPassPipeline(OpPassManager &pm);

//===----------------------------------------------------------------------===//
// IREE-specific passes for TFLite import
//===----------------------------------------------------------------------===//

// Converts TFLite attributes that are useful to corresponding IREE attributes.
std::unique_ptr<OperationPass<ModuleOp>> createConvertModuleMetadataPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertFunctionMetadataPass();

// Strips all leftover TFLite-related attributes; none are needed by IREE.
std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass();
std::unique_ptr<OperationPass<FuncOp>> createStripFunctionMetadataPass();

// Validates whether any TFLite operations remain.
std::unique_ptr<OperationPass<FuncOp>> createVerifyFullyConvertedPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerTFLImportPassPipeline();

inline void registerAllPasses() {
  registerTFLImportPassPipeline();

  createConvertModuleMetadataPass();
  createConvertFunctionMetadataPass();
  createStripModuleMetadataPass();
  createStripFunctionMetadataPass();
  createVerifyFullyConvertedPass();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASSES_H_
