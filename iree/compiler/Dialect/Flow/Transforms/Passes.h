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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required flow
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc to flow>
//   buildFlowTransformPassPipeline & run
//   <run conversion from flow to sequencer/hal/vm/etc>
void buildFlowTransformPassPipeline(OpPassManager &passManager);

void registerFlowTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Flattens tuple values in function signatures and blocks.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenTuplesInCFGPass();

// Legalizes the input types to those supported by the flow dialect.
// This will fail if types that cannot be supported at all are present, however
// conditionally supported types (based on availability, etc) may still be
// allowed to pass through successfully.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeInputTypesPass();

/// Creates XLA-HLO preprocessing transformation pass. In this pass we should
/// have all mhlo -> mhlo transformations that are shared between all
/// backends.
std::unique_ptr<OperationPass<FuncOp>> createHLOPreprocessingPass();

// Runs pre-partitioning conversion passes to convert to the flow dialect.
// This converts some input ops directly to flow ops when doing so has a
// benefit. Other ops are left unmodified and will be outlined later on.
std::unique_ptr<OperationPass<FuncOp>> createPrePartitioningConversionPass();

// Runs post-partitioning conversion passes to legalize the flow dialect.
// This converts any leftover ops that did not already get converted or outlined
// to dispatch regions.
std::unique_ptr<OperationPass<FuncOp>> createPostPartitioningConversionPass();

// Materializes reflection metadata on exported function arguments and results.
// This runs as close to the input processing as possible as it needs to
// annotate the ABI that the consumer is expecting to interop with.
// Note that this does not combine the argument and result metadata into top
// level function metadata. That happens late in transformation, as additional
// synthetic arguments and results may still need to be added.
std::unique_ptr<OperationPass<FuncOp>> createMaterializeExportedReflection();

// Merges f_partial argument and result reflection metadata into a function
// level signature. This should be run late once all synthetic arguments have
// been added and no further exported function signature changes are
// expected.
std::unique_ptr<OperationPass<FuncOp>> createMergeExportedReflection();

//===----------------------------------------------------------------------===//
// Dispatches (flow.dispatch.region)
//===----------------------------------------------------------------------===//

// Analyzes a module to identify which functions are dispatchable.
// This information is cached on the module and is used by other FuncOp-scoped
// passes to quickly access the module-level dispatchability information.
std::unique_ptr<OperationPass<ModuleOp>> createDispatchabilityAnalysisPass();

// Identifies dispatchable regions of functions and wraps them in
// flow.dispatch_regions.
std::unique_ptr<OperationPass<FuncOp>> createIdentifyDispatchRegionsPass();

// Identifies dispatchable regions of functions and wraps them in
// flow.dispatch_regions (version 2).
std::unique_ptr<OperationPass<FuncOp>> createIdentifyDispatchRegions2Pass();

// Folds multiple dispatch regions together that have compatible workloads.
std::unique_ptr<OperationPass<FuncOp>>
createFoldCompatibleDispatchRegionsPass();

// Rematerializes small previously-CSE'd constants into dispatch regions.
std::unique_ptr<OperationPass<FuncOp>>
createRematerializeDispatchConstantsPass();

// Outlines dispatch regions into executables.
std::unique_ptr<OperationPass<ModuleOp>> createOutlineDispatchRegionsPass();

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

// TODO(benvanik): pass to dedupe similar executables (by making dynamically
// shaped, adjusting types, etc).

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
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerFlowPasses() {
  registerFlowTransformPassPipeline();
  createFlattenTuplesInCFGPass();
  createLegalizeInputTypesPass();
  createHLOPreprocessingPass();
  createPrePartitioningConversionPass();
  createPostPartitioningConversionPass();
  createMaterializeExportedReflection();
  createMergeExportedReflection();
  createDispatchabilityAnalysisPass();
  createIdentifyDispatchRegionsPass();
  createIdentifyDispatchRegions2Pass();
  createFoldCompatibleDispatchRegionsPass();
  createRematerializeDispatchConstantsPass();
  createOutlineDispatchRegionsPass();
  createFormStreamsPass();
  createHoistUnstreamableOpsPass();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
