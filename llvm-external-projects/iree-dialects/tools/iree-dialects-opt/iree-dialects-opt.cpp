// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();

  registerTransformsPasses();
  registerSCFPasses();

  // Local dialects.
  mlir::iree_compiler::IREE::PYDM::registerPasses();
  mlir::iree_compiler::IREE::LinalgExt::registerPasses();

  DialectRegistry registry;
  registry.insert<
      // Local dialects
      mlir::iree_compiler::IREE::Input::IREEInputDialect,
      mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
      mlir::iree_compiler::IREE::PYDM::IREEPyDMDialect,
      // Upstream dialects
      mlir::arith::ArithmeticDialect, mlir::cf::ControlFlowDialect,
      mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
      mlir::func::FuncDialect, mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect>();

  IREE::LinalgExt::registerTiledOpInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
