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

//===- Passes.h - Codegen pass to convert from XLA to Linalg on buffers----===//
//
// IREE specific passes used in the XLA to Linalg conversion
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_CODEGENPASSES_PASSES_H
#define IREE_COMPILER_TRANSLATION_CODEGENPASSES_PASSES_H
#include <memory>

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Populates passes to convert from XLA-HLO to Linalg on buffers as well as
/// handling some IREE specific conversions (like IREE::LoadInputOp and
/// IREE::StoreOutputOp). At the end of the pass, the dispatch function will
/// only contain linalg ops or standard ops if the pipeline succeeds.
void addXLAToLinalgOnBuffersPasses(OpPassManager &pm);

/// Fuses linalg operations on tensors in dispatch function. For now does only
/// producer consumer fusion.
std::unique_ptr<OpPassBase<FuncOp>> createLinalgFusionPass();

/// Creates XLA-HLO to Linalg on buffers transformation pass.
std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgOnBuffersPass();

/// Creates XLA-HLO to Linalg on tensors transformation pass.
std::unique_ptr<OpPassBase<FuncOp>> createXLAToLinalgOnTensorsPass();

/// Populates the patterns that convert from XLA to Linalg on tensors. Imports
/// patterns from XLA, as well as some IREE specific modifications.
void populateXLAToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

/// Populates the patterns that convert from XLA to Linalg on buffers. Currently
/// only implements conversions when the XLA op is the only op XLA op in the
/// dispatch region.
void populateXLAToLinalgOnBuffersConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_TRANSLATION_CODEGENPASSES_PASSES_H
