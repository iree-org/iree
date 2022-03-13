// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;

namespace mlir {
namespace test_ext {
/// Test passes, do not deserve an include.
void registerTestLinalgTransformWrapScope();
void registerTestListenerPasses();
}  // namespace test_ext
}  // namespace mlir

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();

  DialectRegistry registry;
  registry.insert<
      // clang-format off
      // Local dialects
      mlir::iree_compiler::IREE::Input::IREEInputDialect,
      mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
      mlir::iree_compiler::IREE::PYDM::IREEPyDMDialect,
      mlir::linalg::transform::LinalgTransformDialect,
      // Upstream dialects
      mlir::arith::ArithmeticDialect, 
      mlir::AffineDialect, 
      mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect, 
      mlir::linalg::LinalgDialect, 
      mlir::memref::MemRefDialect,
      mlir::pdl::PDLDialect, 
      mlir::pdl_interp::PDLInterpDialect, 
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect
      // clang-format on
      >();

  // Core dialect passes.
  registerTransformsPasses();
  registerSCFPasses();
  // Local dialect passes.
  mlir::iree_compiler::IREE::PYDM::registerPasses();
  mlir::iree_compiler::IREE::LinalgExt::registerPasses();
  mlir::linalg::transform::registerLinalgTransformInterpreterPass();
  mlir::linalg::transform::registerLinalgTransformExpertExpansionPass();
  mlir::linalg::transform::registerDropScheduleFromModulePass();
  // Local test passes.
  mlir::test_ext::registerTestLinalgTransformWrapScope();
  mlir::test_ext::registerTestListenerPasses();

  // External models.
  IREE::LinalgExt::registerTiledOpInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        // Note: without preloading, 3 tests fail atm.
                        /*preloadDialectsInContext=*/true));
}
