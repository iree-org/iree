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

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that perform input dialect
// legalization required by the Flow dialect.
//
// NOTE: this will eventually be moved out to an associated import tool - it
// currently relies on linking in all of the input dialects (mhlo, etc) and
// instead those should be taken care of prior to coming into the compiler.
void buildInputTransformPassPipeline(OpPassManager &passManager);

void registerInputTransformPassPipeline();

// Adds a set of passes to the given pass manager that run the required flow
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc to flow>
//   buildInputTransformPassPipeline
//   buildFlowTransformPassPipeline
//   <run conversion from flow to sequencer/hal/vm/etc>
void buildFlowTransformPassPipeline(OpPassManager &passManager);

void registerFlowTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Convert operations to equivalent flow.tensor.* ops. This is run after
// dispatch region creation to catch operations that were left outside of
// dispatch regions and could be represented as flow.tensor.* ops.
std::unique_ptr<OperationPass<FuncOp>> createConvertToFlowTensorOpsPass();

// Legalizes the input types to those supported by the flow dialect.
// This will fail if types that cannot be supported at all are present, however
// conditionally supported types (based on availability, etc) may still be
// allowed to pass through successfully.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeInputTypesPass();

/// Creates XLA-HLO preprocessing transformation pass. In this pass we should
/// have all mhlo -> mhlo transformations that are shared between all
/// backends.
std::unique_ptr<OperationPass<FuncOp>> createHLOToHLOPreprocessingPass();

// Runs pre-partitioning conversion passes to convert to the flow dialect.
// This converts some input ops directly to flow ops when doing so has a
// benefit. Other ops are left unmodified and will be outlined later on.
std::unique_ptr<OperationPass<FuncOp>> createPrePartitioningConversionPass();

// Expands dynamic !shapex.ranked_shape dimensions in variables.
std::unique_ptr<OperationPass<ModuleOp>> createExpandVariableDynamicDimsPass();

//===----------------------------------------------------------------------===//
// Dispatches (flow.dispatch.workgroups)
//===----------------------------------------------------------------------===//

/// Pass to perform dispatch of Linalg on tensor ops by tiling and distribution.
/// A dispatch region is created for each tiled loop nest.
std::unique_ptr<OperationPass<FuncOp>> createDispatchLinalgOnTensorsPass();

// Outlines dispatch regions into executables.
std::unique_ptr<OperationPass<ModuleOp>> createOutlineDispatchRegionsPass();

// Injects tracing markers for dispatch operation tensor inputs and outputs.
std::unique_ptr<OperationPass<FuncOp>> createInjectDispatchTracingPass();

// Exports all functions and dispatch executables as `() -> ()` benchmark funcs.
std::unique_ptr<OperationPass<ModuleOp>> createExportBenchmarkFuncsPass();

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

// Outlines large tensor constants into flow.variables at the module level.
//
// TODO(#5493): implement the support for inlining constants into the command
// buffer and raise this value to one that is measured to be good.
static constexpr size_t kMinLargeConstantSize = 1;
std::unique_ptr<OperationPass<ModuleOp>> createOutlineLargeConstantsPass(
    size_t minLargeConstantSize = kMinLargeConstantSize);

// Deduplicates equivalent executables.
std::unique_ptr<OperationPass<ModuleOp>> createDeduplicateExecutablesPass();

//===----------------------------------------------------------------------===//
// Stream Formation and Folding
//===----------------------------------------------------------------------===//

// Identifies dispatches that can be grouped into streams within functions.
std::unique_ptr<OperationPass<FuncOp>> createFormStreamsPass();

// Reorders blocks to hoist ops that cannot be put into streams.
std::unique_ptr<OperationPass<FuncOp>> createHoistUnstreamableOpsPass();

// TODO(benvanik): cross-function stream flows.

//===----------------------------------------------------------------------===//
// Module Analysis and Finalization
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Simplification and Development Tools
//===----------------------------------------------------------------------===//

// Strips constant flow.variables and replaces them with splats.
// This destructively removes data (often model weights and other parameters)
// and is intended for use as a development tool.
// TODO(scotttodd): pass pipeline with this and other development passes to
//                  generate test cases / models suitable for check-in
std::unique_ptr<OperationPass<ModuleOp>>
createStripAndSplatConstantVariablesPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerFlowPasses();

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
