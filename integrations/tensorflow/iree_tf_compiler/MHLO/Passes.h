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

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_MHLO_PASSES_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_MHLO_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace MHLO {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void buildMHLOImportPassPipeline(OpPassManager &pm);
void registerMHLOImportPassPipeline();

//===----------------------------------------------------------------------===//
// IREE-specific Passes For MHLO Import
//===----------------------------------------------------------------------===//

// Annotates an appropriate iree.abi attribute on public functions that
// operate exclusively on tensor types. This corresponds to the expectations
// of MHLO and is suitable for such programs.
std::unique_ptr<OperationPass<FuncOp>> createEmitDefaultIREEABIPass();

// Flattens tuple values in function signatures and blocks.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenTuplesInCFGPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerAllDialects(mlir::DialectRegistry &registry);

inline void registerAllPasses() {
  registerMHLOImportPassPipeline();

  createEmitDefaultIREEABIPass();
  createFlattenTuplesInCFGPass();
}

}  // namespace MHLO
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_MHLO_PASSES_H_
