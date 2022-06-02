// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SourceMgr.h"
#include <mlir/Pass/PassRegistry.h>

#define DEBUG_TYPE "transform-interpreter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

static llvm::cl::opt<std::string> clTransformFileName(
    "linalg-transform-file-name",
    llvm::cl::desc("mlir file containing a top-level module that specifies "
                   "the transformations to apply."),
    llvm::cl::init(""));

namespace {
/// Simple pass that applies transform dialect ops directly contained in a
/// module.
class LinalgTransformInterp : public PassWrapper<LinalgTransformInterp, Pass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTransformInterp)

  StringRef getArgument() const override { return "linalg-transform-interp"; }

  StringRef getDescription() const override {
    return "apply transform dialect operations one by one";
  }

  bool canScheduleOn(RegisteredOperationName name) const override {
    return true;
  }

  void runOnOperation() override {
    Operation *topLevel = getOperation();
    if (topLevel->getNumRegions() != 1 ||
        !llvm::hasSingleElement(topLevel->getRegion(0))) {
      topLevel->emitError() << "can only run '" << getArgument()
                            << "' on single-region single-block operations";
      return signalPassFailure();
    }

    if (clTransformFileName.empty()) {
      transform::TransformState state(topLevel->getRegion(0), topLevel);
      Block &body = topLevel->getRegion(0).front();
      for (auto op : body.getOps<transform::TransformOpInterface>()) {
        if (failed(state.applyTransform(op)))
          return signalPassFailure();
      }
      return;
    }

    // If a transform file is specified, parse its content into a ModuleOp.
    std::string errorMessage;
    auto memoryBuffer = openInputFile(clTransformFileName, &errorMessage);
    if (!memoryBuffer) {
      llvm::errs() << errorMessage << "\n";
      return signalPassFailure();
    }
    // Tell sourceMgr about this buffer, the parser will pick it up.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
    OwningOpRef<ModuleOp> transformModule(
        parseSourceFile<ModuleOp>(sourceMgr, &getContext()));
    transform::TransformState state(
        transformModule->getOperation()->getRegion(0), topLevel);
    for (auto op : transformModule->getBody()
                       ->getOps<transform::TransformOpInterface>()) {
      if (failed(state.applyTransform(op)))
        return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: this is only necessary to make registry subset happy when running
    // the lowering to LLVM. The lowering should be changed to stop using the
    // nested pass manager and this will go away.

    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    arith::ArithmeticDialect,
                    AffineDialect,
                    bufferization::BufferizationDialect,
                    func::FuncDialect,
                    linalg::LinalgDialect,
                    linalg::transform::LinalgTransformDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect
        // clang-format on
        >();

    // TODO: these should be registered by the extension instead, but there is
    // no support for it in core currently.
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }
};

struct DropSchedulePass : public PassWrapper<DropSchedulePass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropSchedulePass)

  StringRef getArgument() const final { return "linalg-drop-schedule"; }

  StringRef getDescription() const final {
    return "Drop the schedule from the operation";
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<::mlir::transform::TransformOpInterface>(nestedOp)) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

/// Create a Linalg Transform interpreter pass.
std::unique_ptr<Pass> mlir::createLinalgTransformInterpreterPass() {
  return std::make_unique<LinalgTransformInterp>();
}

/// Create a Linalg pass to drop the schedule from the module.
std::unique_ptr<Pass> mlir::createDropSchedulePass() {
  return std::make_unique<DropSchedulePass>();
}

/// Registration hook for the Linalg drop schedule from module pass.
void mlir::linalg::transform::registerDropSchedulePass() {
  PassRegistration<DropSchedulePass>();
}

/// Registration hook for the Linalg Transform interpreter pass.
void mlir::linalg::transform::registerLinalgTransformInterpreterPass() {
  PassRegistration<LinalgTransformInterp>();
}
