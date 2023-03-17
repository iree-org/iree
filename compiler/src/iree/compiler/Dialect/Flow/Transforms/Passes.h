// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_

#include <functional>

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

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  // Enables the iree-util-hoist-into-globals pass. This should eventually
  // become the default.
  bool constExprHoisting = false;

  // Enables passes to perform numeric precision reduction.
  bool numericPrecisionReduction = false;

  // Hook to populate a constant evaluation pass pipeline. If nullptr, then
  // no passes are added for constant evaluation. This must be injected in
  // because constant-evaluators can depend on the whole compiler, of which
  // this is a part, and we maintain strict optionality for this component.
  std::function<void(OpPassManager &passManager)> buildConstEvalPassPipeline;
};

// Adds a set of passes to the given pass manager that run the required flow
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   Input legalization by one of:
//     - Directly passing supported flow plus core ops
//   buildFlowTransformPassPipeline
//   <run conversion from flow to sequencer/hal/vm/etc>
void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions);

void registerFlowTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Apply patterns to erase unused linalg operands and remove dead code
// associated.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands();

// Expands tensor shape dimensions into SSA values across the program.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createExpandTensorShapesPass();

// Cleans up any remaining shape metadata ops after lowering.
std::unique_ptr<Pass> createCleanupTensorShapesPass();

// Cleans up any numeric narrowing ops inserted by
// iree-flow-infer-numeric-narrowing.
std::unique_ptr<Pass> createCleanupNumericNarrowingPass();

// Creates a pass to convert linalg convolution ops with 1x1 kernels into
// linalg.matmul
std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass();

// Creates a pass to convert dispatch.region ops to dispatch.workgroups ops.
std::unique_ptr<Pass> createConvertRegionToWorkgroupsPass();

// Pass to convert a tensor.pad operation into a linalg.fill +
// tensor.insert_slice.
std::unique_ptr<Pass> createTensorPadToTensorInsertSlicePass(
    bool skipSingleLinalgOpUses = false);

// Create a pass to detach elementwise ops from named Linalg ops.
std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();

// Creates a pass to fuse Linalg operations on tensors.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFusionOfTensorOpsPass(bool fuseMultiUse = false,
                            unsigned multiUseFusionIteration = 2);

// Infers and inserts util.numeric.optional_narrow ops at points that may be
// beneficial.
std::unique_ptr<Pass> createInferNumericNarrowingPass();

// Create a pass to initialize all empty tensors after dispatch formation to
// zero or uninitialized allocations.
std::unique_ptr<Pass> createInitializeEmptyTensorsPass(bool zeroFill = false);

// Create a pass to interchange generic ops to force the reduction loop to be
// the most inner loops.
std::unique_ptr<Pass> createInterchangeGenericOpsPass();

// Create a pass to interchange generic ops to make the input indexing map
// identity.
std::unique_ptr<Pass> createInterchangeTransposeGenericOpsPass();

// Create a pass to convert operations to `flow` ops. This pass is currently
// only used for testing, since the conversion to Flow ops happens within
// dispatch region formation.
std::unique_ptr<Pass> createConvertToFlowPass();

// Optimizes numerics given annotations added via
// iree-flow-infer-numeric-narrowing.
std::unique_ptr<Pass> createOptimizeNumericsPass();

// Sets encoding for tensors to allow tiled execution of operations.
std::unique_ptr<Pass> createSetEncodingPass();

// Strips the signed/unsigned portion off of tensors.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createStripSignednessPass();

// Verifies that the input to the Flow transformation pipeline is legal.
// This includes checking for operations from dialects that are expected
// to be legalized before this pass.
std::unique_ptr<Pass> createVerifyInputLegalityPass();

//===----------------------------------------------------------------------===//
// Dispatches (flow.dispatch.region)
//===----------------------------------------------------------------------===//

// Pass to form dispatch.region ops from Linalg on tensor ops. A dispatch region
// is created for each tiled loop nest. This pass only moves the root compute op
// into the dispatch region, allowing producers to be outside.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormDispatchRegionsPass(bool fuseMultiUse = false,
                              bool generateWorkloadRegion = true);

// Pass to collapse dimensions of Linalg Ops on tensor ops.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCollapseDimensionsPass();

// Pass to clone into dispatch regions producers of values used in the dispatch
// regions but defined in the above. This prepares the dispatch regions for
// converting to dispatch workgroups with explicit captures.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCloneProducersIntoDispatchRegionsPass();

//===----------------------------------------------------------------------===//
// Dispatches (flow.dispatch.workgroups)
//===----------------------------------------------------------------------===//

// Pass to perform dispatch of dispatch.region ops that contain Linalg on tensor
// ops by tiling and distribution. A dispatch region is created for each tiled
// loop nest.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormDispatchWorkgroupsPass(bool generateWorkloadRegion = true);

// Pass to perform dispatch of Linalg on tensor ops by using the transform
// dialect. Dispatch regions are created as specified by the transform module
// that is parsed from `transformFileName`.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDispatchWithTransformDialect(
    llvm::StringRef transformFileName = llvm::StringRef(),
    llvm::StringRef debugPayloadRootTag = llvm::StringRef(),
    llvm::StringRef debugTransformRootTag = llvm::StringRef());

// Captures dynamic shape dimensions required by dispatch operands.
std::unique_ptr<Pass> createCaptureDispatchDynamicDimsPass();

// Outlines dispatch regions into executables.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchRegionsPass();

// Injects tracing markers for dispatch operation tensor inputs and outputs.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createInjectDispatchTracingPass();

// Exports all functions and dispatch executables as `() -> ()` benchmark funcs.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createExportBenchmarkFuncsPass();

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

// Outlines large tensor constants into util.globals at the module level.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineLargeConstantsPass();

// Deduplicates equivalent executables.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDeduplicateExecutablesPass();

// Create a pass to raise sequence of ops to higher level linalg.ext
// representation.
std::unique_ptr<Pass> createRaiseSpecialOps();

// Create a pass to split reduction dimension.
std::unique_ptr<Pass> createSplitReductionPass();

// Create a pass to collapse reduction dimensions
std::unique_ptr<Pass> createCollapseDimsPass();

//===----------------------------------------------------------------------===//
// Module Analysis and Finalization
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Simplification and Development Tools
//===----------------------------------------------------------------------===//

// Strips constant util.globals and replaces them with splats.
// This destructively removes data (often model weights and other parameters)
// and is intended for use as a development tool.
// TODO(scotttodd): pass pipeline with this and other development passes to
//                  generate test cases / models suitable for check-in
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createStripAndSplatConstantVariablesPass();

/// Creates a pass to dump a graph for dispatches
std::unique_ptr<Pass> createDumpDispatchGraphPass(
    raw_ostream &os = llvm::errs());

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerFlowPasses();

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
