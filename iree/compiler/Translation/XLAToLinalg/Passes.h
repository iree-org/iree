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

//===- Passes.h - IREE specific passes used in Linalg To SPIRV conversion--===//
//
// IREE specific passes used in the XLA -> Linalg -> SPIRV Conversion.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_LINALGTOSPIRV_PASSES_H
#define IREE_COMPILER_TRANSLATION_LINALGTOSPIRV_PASSES_H
#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

/// Fuses linalg operations on tensors in dispatch function. For now does only
/// producer consumer fusion.
std::unique_ptr<OpPassBase<FuncOp>> createLinalgFusionPass();

/// Pass to convert a linalg.generic operation on tensors to a
/// linalg.generic operation on memrefs. To be used only if the entire dispatch
/// region becomes a single linalg.generic op after conversion from xla to
/// linalg and fusion.
std::unique_ptr<OpPassBase<FuncOp>> createLinalgTensorToBufferConversionPass();

// Creates XLAHLO to linalg transformation pass.
std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgPass();

// Creates XLAHLO to linalg named ops transformation pass.
std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgOpsOnBufferPass();

/// Populates patterns to convert a linalg.generic operation on tensors to a
/// linalg.generic operation on memrefs. To be used only if the entire dispatch
/// region becomes a single linalg.generic op after conversion from xla to
/// linalg and fusion.
void populateLinalgTensorToBufferConversionPattern(
    MLIRContext* context, OwningRewritePatternList* patterns);

// Populates patterns to convert HLO to Linalg.
void populateXlaToLinalgConversionPattern(MLIRContext* context,
                                          OwningRewritePatternList* patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_PASSES_H
