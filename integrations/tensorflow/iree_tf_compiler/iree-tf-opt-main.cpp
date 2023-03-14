// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for iree-tf-opt and derived binaries.
//
// This is a bare-bones, minimal *-opt just for testing the handful of local
// passes here. If you need something, add it, but add only what you need as
// each addition will likely end up on the build critical path.

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree_tf_compiler/MHLO/Passes.h"
#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_passes.h"
#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

int main(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::iree_compiler::IREE::Input::IREEInputDialect>();
  registry.insert<mlir::chlo::ChloDialect, mlir::mhlo::MhloDialect>();

  // TensorFlow integration passes.
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::iree_integrations::TF::registerAllPasses();
  mlir::iree_integrations::MHLO::registerAllPasses();

  // Select MLIR passes.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::registerRemoveShapeConstraintsPass();
  mlir::registerSymbolDCEPass();

  // Select TF passes.
  mlir::registerExecutorGraphPruningPassPass();
  mlir::registerTensorFlowShapeInferencePassPass();
  mlir::registerTensorFlowOptimizePassPass();
  mlir::TFDevice::registerDecomposeResourceOpsPassPass();

  // Old style static registration based TF passes.
  mlir::TF::CreateDeviceIndexSelectorPass();
  mlir::TF::CreateGuaranteeAllFuncsOneUsePass();
  mlir::TF::CreateTFFunctionalControlFlowToCFG();

  // Tosa related passes.
  mlir::tosa::registerLegalizeTosaPasses();
  mlir::tosa::registerTFtoTOSALegalizationPipeline();
  mlir::tosa::registerTFLtoTOSALegalizationPipeline();

  if (failed(MlirOptMain(argc, argv, "IREE-TF modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
