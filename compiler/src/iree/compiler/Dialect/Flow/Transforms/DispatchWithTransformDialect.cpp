// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for dispatch region
/// formation. This needs to be its own pass because the registration mechanism
/// and ops available are different than for other interpreters.
class DispatchWithTransformDialect
    : public DispatchWithTransformDialectBase<DispatchWithTransformDialect> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    IREE::Flow::FlowDialect,
                    affine::AffineDialect,
                    arith::ArithDialect,
                    linalg::LinalgDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect
    >();
    // clang-format on
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp transformModule =
        transform::detail::getPreloadedTransformModule(context);
    if (failed(transform::applyTransformNamedSequence(
            getOperation(), transformModule,
            options.enableExpensiveChecks(true), entryPoint)))
      return signalPassFailure();
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};

/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass> createDispatchWithTransformDialect() {
  return std::make_unique<DispatchWithTransformDialect>();
}
} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
