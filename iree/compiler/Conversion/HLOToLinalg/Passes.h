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
#ifndef IREE_COMPILER_CONVERSION_HLOTOLINALG_PASSES_H_
#define IREE_COMPILER_CONVERSION_HLOTOLINALG_PASSES_H_
#include <memory>

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Creates XLA-HLO to Linalg on buffers transformation pass.
std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnBuffersPass();

/// Creates XLA-HLO to Linalg on tensors transformation pass.
std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass();

/// Resolves shape related ops (std.dim, shapex.tie_shape, etc.) by tracing
/// them back to the original HAL interface bindings.
std::unique_ptr<OperationPass<FuncOp>> createResolveShapeOpsPass();

/// Populates the patterns that convert from XLA to Linalg on tensors. Imports
/// patterns from XLA, as well as some IREE specific modifications.
void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

/// Populates the patterns that convert from XLA to Linalg on buffers. Currently
/// only implements conversions when the XLA op is the only op XLA op in the
/// dispatch region.
using TensorToBufferMap = DenseMap<Value, Value>;
void populateHLOToLinalgOnBuffersConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    TensorToBufferMap const &resultTensorToBufferMap);

/// Populates passes to convert from XLA-HLO to Linalg on buffers as well as
/// handling some IREE specific conversions (like iree.interface.* and
/// iree.placeholder op). At the end of the pass, the dispatch function will
/// only contain linalg ops or standard ops if the pipeline succeeds. The pass
/// manager `pm` passed in here is expected to operate on the module within the
/// IREE::HAL::ExecutableTargetOp.
void addHLOToLinalgOnBuffersPasses(OpPassManager &pm);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_HLOTOLINALG_PASSES_H_
