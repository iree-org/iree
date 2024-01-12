// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h>

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;

namespace mlir {
namespace test_ext {
/// Test passes, do not deserve an include.
void registerVectorExtTestPasses();
} // namespace test_ext
} // namespace mlir

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();

  DialectRegistry registry;
  registry.insert<
      // clang-format off
      // Local dialects
      mlir::iree_compiler::IREE::Input::IREEInputDialect,
      mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
      mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect,
      // Upstream dialects
      mlir::async::AsyncDialect,
      mlir::arith::ArithDialect,
      mlir::affine::AffineDialect,
      mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::memref::MemRefDialect,
      mlir::pdl::PDLDialect,
      mlir::pdl_interp::PDLInterpDialect,
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect,
      mlir::transform::TransformDialect,
      mlir::vector::VectorDialect
      // clang-format on
      >();

  // Core dialect passes.
  memref::registerMemRefPasses();
  registerTransformsPasses();
  registerSCFPasses();
  // Local dialect passes.
  mlir::iree_compiler::IREE::LinalgExt::registerPasses();
  mlir::linalg::transform::registerTransformDialectInterpreterPass();
  mlir::linalg::transform::registerDropSchedulePass();
  // Local test passes.
  mlir::test_ext::registerVectorExtTestPasses();

  // External models.
  mlir::func::registerInlinerExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);

  registry.addExtensions<IREE::LinalgExt::LinalgExtTransformOpsExtension,
                         transform_ext::StructuredTransformOpsExtension>();
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::tensor::registerFindPayloadReplacementOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);

  // Dialect extensions.
  transform::registerPDLExtension(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
