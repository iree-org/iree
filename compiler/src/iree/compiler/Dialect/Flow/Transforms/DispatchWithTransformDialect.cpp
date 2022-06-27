// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for dispatch region
/// formation. This needs to be its own pass because the registration mechanism
/// and ops available are different than for other interpreters.
struct DispatchWithTransformDialect
    : public DispatchWithTransformDialectBase<DispatchWithTransformDialect> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    IREE::Flow::FlowDialect,
                    AffineDialect,
                    arith::ArithmeticDialect,
                    linalg::LinalgDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect
    >();
    // clang-format on
  }

  DispatchWithTransformDialect(StringRef transformFileName) {
    this->transformFileName = transformFileName.str();
  }
  DispatchWithTransformDialect(const DispatchWithTransformDialect &pass) {
    this->transformFileName = pass.transformFileName;
    // TODO: if we really don't like shared_ptr, we could also clone the
    // transformModule here.
    sharedTransformModule = pass.sharedTransformModule;
  }
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;

 private:
  Statistic numDispatches{this, "number of dispatches",
                          "Number of Flow dispatches created"};

  // The parsed transform module to be used for creating dispatches.
  // TODO: Figure a better way to build a transform module and transport it in
  // the proper places in the IR as it is transformed by IREE so that it is
  // available with better ownership semantics.
  // Note: we wrap the OwningOpRef to get the desired destruction mechanism.
  // Note: shared_ptr is not great but we know the sharedTransformModule is
  // readonly.
  // Alternatives comprise:
  //   1. no shared_ptr but copying the module with every pass clone that the
  //      OpPassManager decides to perform.
  //   2. lifting ownership of the parsed transform module higher up in the
  //      IREE stack. This may be only shift the problem as we have passes
  //      building pass managers in IREE.
  //   3. build better support to embed the transformation module in the
  //      input IR and transport it to the place of use in IREE. This is deemed
  //      too intrusive atm.
  //   4. (future) config/resources mechanism that is being proposed in core?
  std::shared_ptr<OwningOpRef<ModuleOp>> sharedTransformModule;
};

LogicalResult DispatchWithTransformDialect::initialize(MLIRContext *context) {
  OwningOpRef<ModuleOp> module;
  if (failed(transform::parseTransformModuleFromFile(context, transformFileName,
                                                     module)))
    return failure();
  sharedTransformModule =
      std::make_shared<OwningOpRef<ModuleOp>>(std::move(module));
  return success();
}

void DispatchWithTransformDialect::runOnOperation() {
  Operation *target = getOperation();
  bool parsedTransform = (sharedTransformModule && *sharedTransformModule);
  assert(parsedTransform || (target->getNumRegions() == 1 &&
                             target->getRegion(0).getBlocks().size() == 1) &&
                                "Cannot extract transform from op");
  Region &transformRegion = parsedTransform
                                ? (*sharedTransformModule)->getRegion()
                                : target->getRegion(0);
  if (failed(transform::applyTransformsInRegion(transformRegion, target)))
    return signalPassFailure();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDispatchWithTransformDialect(StringRef transformFileName) {
  return std::make_unique<DispatchWithTransformDialect>(transformFileName);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
