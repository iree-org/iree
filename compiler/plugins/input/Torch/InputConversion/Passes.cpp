// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler::TorchInput {

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc" // IWYU pragma: export
} // namespace

void createTorchToIREEPipeline(
    OpPassManager &pm, const TorchToIREELoweringPipelineOptions &options) {
  // This pipeline adapted from
  // createTorchBackendToLinalgOnTensorsBackendPipeline. Keep in sync with
  // additions there. Lower to linalg + guards which is the input to codegen
  // backends. We do this first as it tends to involve pattern-matching against
  // constants, (e.g. dimensions which must be constant in a ranked programming
  // model) and those constants get somewhat obscured by TorchToArith.
  // Dynamic shape bindings add a lot of structure to the IR which we prefer to
  // leverage and eliminate prior to any other activity, so do this first.
  pm.addNestedPass<func::FuncOp>(createBindSymbolicShapesPass());

  if (options.strictSymbolicShapes) {
    pm.addNestedPass<func::FuncOp>(createSetStrictSymbolicShapesPass());
    // Run canonicalization in case any previously non-strict dynamic code can
    // now be simplified.
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
  pm.addNestedPass<func::FuncOp>(createBitCastQuantTensorPass());
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createReduceOpVariantsPass(llvm::StringRef()));
  pm.addNestedPass<func::FuncOp>(
      mlir::torch::TorchConversion::createConvertCustomQuantOpPass());
  if (options.decompose)
    pm.addNestedPass<func::FuncOp>(
        torch::Torch::createDecomposeComplexOpsPass(BackendLegalOps::get()));
  pm.addNestedPass<func::FuncOp>(torch::Torch::createFuseQuantizedOpsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(torch::Torch::createScalarizeShapesPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTMTensorPass());
  pm.addNestedPass<func::FuncOp>(
      TorchInput::createConvertTMTensorToLinalgExtPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTensorPass());
  pm.addNestedPass<func::FuncOp>(
      TorchInput::createConvertTorchUnstructuredToLinalgExtPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToArithPass());
  pm.addPass(torch::createConvertTorchConversionToMLProgramPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Resolve `dim` ops on tensors (which currently live in the `memref`
  // dialect for some reason -- we don't have memrefs at this level).
  pm.addNestedPass<func::FuncOp>(
      memref::createResolveShapedTypeResultDimsPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Regular function calls in torch have to be inlined presently. In the
  // future, we would like to support async invocation, which will operate
  // differently and would not be subject to inlining.
  pm.addPass(mlir::createInlinerPass());

  pm.addPass(createFuncConversionPass());
  pm.addNestedPass<IREE::Util::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());

  // Finish the type conversion from `torch` types to the types of the
  // linalg-on-tensors backend contract.
  pm.addNestedPass<IREE::Util::FuncOp>(
      torch::TorchConversion::createFinalizingBackendTypeConversionPass());
}

void registerTMTensorConversionPasses() {
  // Generated.
  registerPasses();

  mlir::PassPipelineRegistration<TorchToIREELoweringPipelineOptions>(
      "torch-to-iree",
      "Pipeline to lower from the Torch backend contract to legal IREE input.",
      createTorchToIREEPipeline);
}

} // namespace mlir::iree_compiler::TorchInput
