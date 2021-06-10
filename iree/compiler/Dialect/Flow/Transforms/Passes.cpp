// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/LinalgToLinalg/Passes.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

// TODO(benvanik): change to a pipeline option.
static llvm::cl::opt<bool> clExportBenchmarkFuncs(
    "iree-flow-export-benchmark-funcs",
    llvm::cl::desc(
        "Exports one function per original module entry point and "
        "unique flow.executable that dispatches with dummy arguments."),
    llvm::cl::init(false));

// TODO(benvanik): change to a pipeline option.
static llvm::cl::opt<bool> clTraceDispatchTensors(
    "iree-flow-trace-dispatch-tensors2",
    llvm::cl::desc(
        "Trace runtime input/output tensors for each dispatch function."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clDemoteF32ToF16(
    "iree-flow-demote-f32-to-f16",
    llvm::cl::desc("Convert all f32 ops and values into f16 counterparts "
                   "unconditionally before main flow conversions"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnable1x1ConvToMatmul(
    "iree-flow-enable-1x1-conv-to-matmul",
    llvm::cl::desc("Enable converting 1x1 linalg convolution ops to linalg "
                   "matmul ops pass."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableConvToImg2Col(
    "iree-flow-enable-conv-img2col-transform",
    llvm::cl::desc("Enable converting convolution ops to img2col form."),
    llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Prepare HLO for use as an input to the Flow dialect.
void buildMHLOInputTransformPassPipeline(OpPassManager &passManager) {
  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  // TODO: Currently recurses into SCF in Linalg generic - with hilarity.
  passManager.addNestedPass<FuncOp>(mlir::createLowerToCFGPass());

  // Various shape functions may have been materialized in the `shape.shape_of`
  // style of treating shapes as tensors. We prefer to legalize these to
  // scalar ops as early as possible to avoid having them persist as tensor
  // computations.
  passManager.addNestedPass<FuncOp>(createShapeToShapeLowering());
  passManager.addPass(createConvertShapeToStandardPass());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  // Now that control flow has been lowered, promote and extract_element
  // to tensor loads. This will be done again later once everything that can
  // be is lowered to device.
  passManager.addNestedPass<FuncOp>(createPromoteTensorLoadsPass());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  passManager.addNestedPass<FuncOp>(
      IREE::Flow::createHLOToHLOPreprocessingPass());

  // Perform initial cleanup.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Legalize input types. We do this after flattening tuples so that we don't
  // have to deal with them.
  // TODO(nicolasvasilache): createLegalizeInputTypesPass is old and does not
  // handle region conversion properly (parent cloned before children). Revisit
  // when using ops with regions such as scf.for and linalg.generic.
  passManager.addPass(
      mlir::iree_compiler::IREE::Flow::createLegalizeInputTypesPass());

  // Convert to Linalg. After this point, HLO will be eliminated.
  passManager.addNestedPass<FuncOp>(
      mlir::iree_compiler::createHLOToLinalgOnTensorsPass());

  // Note that some MHLO ops are left by the above and must resolve via
  // canonicalization. See comments in the above pass and find a better way.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
}

// Prepare TOSA for use as an input to the Flow dialect.
void buildTOSAInputTransformPassPipeline(OpPassManager &passManager) {
  passManager.addNestedPass<FuncOp>(tosa::createTosaToSCF());

  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  // TODO: Currently recurses into SCF in Linalg generic - with hilarity.
  passManager.addNestedPass<FuncOp>(mlir::createLowerToCFGPass());

  // Now that control flow has been lowered, promote and extract_element
  // to tensor loads. This will be done again later once everything that can
  // be is lowered to device.
  passManager.addNestedPass<FuncOp>(createPromoteTensorLoadsPass());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  passManager.addNestedPass<FuncOp>(tosa::createTosaToStandard());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(Flow::createPromoteI1ToI8Pass());
  passManager.addNestedPass<FuncOp>(tosa::createTosaToLinalgOnTensors());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
}

void registerInputTransformPassPipeline() {
  PassPipelineRegistration<> tosa(
      "iree-tosa-input-transformation-pipeline",
      "Runs the TOSA IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildTOSAInputTransformPassPipeline(passManager);
      });
  PassPipelineRegistration<> mhlo(
      "iree-mhlo-input-transformation-pipeline",
      "Runs the MHLO IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildMHLOInputTransformPassPipeline(passManager);
      });
}

void buildFlowTransformPassPipeline(OpPassManager &passManager) {
  //----------------------------------------------------------------------------
  // Entry dialect cleanup
  //----------------------------------------------------------------------------
  passManager.addPass(createVerifyCompilerInputLegality());

  // Perform initial cleanup.
  // NOTE: There is no principled reason to be doing this here. But also ensures
  // some consistency at the tool boundary.
  passManager.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Replaces variables with !shapex.ranked_shape types with individual
  // variables for each dimension. This allows for constant dimensions to be
  // DCE'd in following passes.
  passManager.addPass(IREE::Flow::createExpandVariableDynamicDimsPass());

  // Materialize dynamic shapes in the IR, also expanding function signatures
  // such that:
  //   - Dynamic ranked tensors: (tensor<?x?xf32>) expands to
  //     (tensor<?x?xf32>, ranked_shape<[?,?]>), and ultimately expands to
  //     (tensor<?x?xf32>, i32, i32)
  //   - Unranked tensors: **unsupported**
  // The generated ABI wrappers assume such an expansion and will generate code
  // to produce it from the original reflection metadata captured in the
  // previous pass.
  passManager.addNestedPass<FuncOp>(
      Shape::createExpandFunctionDynamicDimsPass());

  // Special case peephole optimizations.
  if (clEnable1x1ConvToMatmul) {
    passManager.addNestedPass<FuncOp>(
        mlir::iree_compiler::createConvert1x1ConvToMatmulPass());
  }
  if (clEnableConvToImg2Col) {
    passManager.addNestedPass<FuncOp>(
        mlir::iree_compiler::createConvertConv2DToImg2ColPass());
  }
  passManager.addPass(
      mlir::iree_compiler::createPadTensorToSubTensorInsertPass());

  // Elementwise, fusion, tiling and distribution.
  passManager.addNestedPass<FuncOp>(
      mlir::createConvertElementwiseToLinalgPass());
  passManager.addNestedPass<FuncOp>(mlir::createLinalgFoldUnitExtentDimsPass());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(
      mlir::iree_compiler::createFusionOfTensorOpsPass());
  passManager.addNestedPass<FuncOp>(
      IREE::Flow::createConvertToFlowTensorOpsPass());
  passManager.addNestedPass<FuncOp>(mlir::createCSEPass());
  passManager.addNestedPass<FuncOp>(
      IREE::Flow::createDispatchLinalgOnTensorsPass());
  // NOTE: required because the current dispatch-linalg-on-tensors pass
  // creates a lot of dead IR that needs to be cleaned up.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  // Outline the dispatch regions into their own functions wrapped in
  // executables.
  passManager.addPass(IREE::Flow::createOutlineDispatchRegionsPass());

  // Cleanup identity ops that clutter up the IR and canonicalize.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  // Deduplicate executables created from dispatch regions.
  // Note: this only deduplicates equivalent executables. We could in addition
  // generalize executables to prune further (e.g. by promoting a dimension to
  // an argument if two executables differ only in that one dimension).
  passManager.addPass(IREE::Flow::createDeduplicateExecutablesPass());

  // TODO: Prune and rename this pass. This runs after sending everything
  // possible to the device and then legalizes any remaining h<->d loads,
  // typically coming from top level flow control.
  passManager.addNestedPass<FuncOp>(IREE::Flow::createPromoteTensorLoadsPass());

  // Create one function per remaining flow.executable that can be used with
  // iree-benchmark-module to benchmark each dispatch individually, as well as
  // exporting all original model entry points.
  if (clExportBenchmarkFuncs) {
    passManager.addPass(IREE::Flow::createExportBenchmarkFuncsPass());
  }

  // Inject tracing that logs both input and output tensors from all dispatches.
  // We do this after deduping so that the executable names match later stages.
  if (clTraceDispatchTensors) {
    passManager.addNestedPass<FuncOp>(
        IREE::Flow::createInjectDispatchTracingPass());
  }

  //----------------------------------------------------------------------------
  // Stream formation.
  // Pre-conditions:
  //   - Full formation of dispatch regions
  //----------------------------------------------------------------------------

  // Form streams.
  // Cleanup the IR before we try to form streams.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Reorder blocks to increase the grouping of streamable ops.
  passManager.addNestedPass<FuncOp>(
      IREE::Flow::createHoistUnstreamableOpsPass());
  // The hoisting pass does some reordering. Canonicalize to avoid unnecessary
  // arbitrary ordering.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  passManager.addNestedPass<FuncOp>(IREE::Flow::createFormStreamsPass());

  // Prior to leaving the pipeline we need to clean things up for following
  // layers. These transforms may be undone by subsequent CSE/folding passes.
  passManager.addPass(IREE::Flow::createOutlineLargeConstantsPass());

  // Forming streams involves a fair amount of subgraph stitching, which can
  // cause duplication. Run CSE to collapse.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());
}

void registerFlowTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-flow-transformation-pipeline",
      "Runs the full IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildFlowTransformPassPipeline(passManager);
      });
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerFlowPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerInputTransformPassPipeline();
  registerFlowTransformPassPipeline();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
