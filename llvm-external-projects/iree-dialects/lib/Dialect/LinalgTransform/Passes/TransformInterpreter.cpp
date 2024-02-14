// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

template <typename T>
class PassWrapperStub : public PassWrapper<T, Pass> {};

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for codegen.
/// This needs to be its own pass because the registration mechanism and ops
/// available are different than for other interpreters.
class TransformDialectInterpreter
    : public mlir::transform::TransformInterpreterPassBase<
          TransformDialectInterpreter, PassWrapperStub> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformDialectInterpreter)

  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: this is only necessary to make registry subset happy when running
    // the lowering to LLVM. The lowering should be changed to stop using the
    // nested pass manager and this will go away.

    // clang-format off
    registry.insert<arith::ArithDialect,
                    affine::AffineDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
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

  StringRef getArgument() const override {
    return "transform-dialect-interpreter";
  }

  StringRef getDescription() const override {
    return "apply transform dialect operations one by one";
  }

  bool canScheduleOn(RegisteredOperationName name) const override {
    return true;
  }

  TransformDialectInterpreter(StringRef transformFileName = StringRef()) {
    this->transformFileName = transformFileName.str();
  }
  TransformDialectInterpreter(const TransformDialectInterpreter &pass)
      : TransformInterpreterPassBase(pass) {
    transformFileName = pass.transformFileName;
    debugPayloadRootTag = pass.debugPayloadRootTag;
    debugTransformRootTag = pass.debugTransformRootTag;
  }

  Pass::Option<std::string> transformFileName{
      *this, "transform-file-name",
      ::llvm::cl::desc(
          "Optional filename containing a transform dialect specification to "
          "apply. If left empty, the IR is assumed to contain one top-level "
          "transform dialect operation somewhere in the module."),
      ::llvm::cl::init("")};
  Pass::Option<std::string> debugPayloadRootTag{
      *this, "debug-payload-root-tag",
      ::llvm::cl::desc("Select the operation with 'transform.target_tag' "
                       "attribute having the given value as payload IR root."),
      ::llvm::cl::init("")};
  Pass::Option<std::string> debugTransformRootTag{
      *this, "debug-transform-root-tag",
      ::llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as container IR for top-level transform ops."),
      ::llvm::cl::init("")};
  Pass::ListOption<std::string> transformLibraryPaths{
      *this, "transform-library-paths", llvm::cl::ZeroOrMore,
      llvm::cl::desc(
          "Optional name of the file containing transform dialect symbol "
          "definitions to be injected into the transform module.")};
};

struct DropSchedulePass : public PassWrapper<DropSchedulePass, Pass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropSchedulePass)

  StringRef getArgument() const final {
    return "transform-dialect-drop-schedule";
  }

  StringRef getDescription() const final {
    return "Drop the schedule from the operation";
  }

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return true;
  }

  void runOnOperation() override {
    SmallVector<Operation *> toDelete;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<::mlir::transform::TransformOpInterface>(nestedOp)) {
        toDelete.push_back(nestedOp);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    for (auto op : toDelete) {
      op->erase();
    }
    SmallVector<ModuleOp> modulesToDelete;
    // Remove potential empty module after cleanup.
    getOperation()->walk([&](ModuleOp module) {
      if (module.getBodyRegion().hasOneBlock() && module.getBody()->empty()) {
        modulesToDelete.push_back(module);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    for (auto module : modulesToDelete) {
      module->erase();
    }
  }
};
} // namespace

/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass>
mlir::createTransformDialectInterpreterPass(llvm::StringRef transformFileName) {
  return std::make_unique<TransformDialectInterpreter>(transformFileName);
}

/// Create a Linalg pass to drop the schedule from the module.
std::unique_ptr<Pass> mlir::createDropSchedulePass() {
  return std::make_unique<DropSchedulePass>();
}

/// Registration hook for the Linalg drop schedule from module pass.
void mlir::linalg::transform::registerDropSchedulePass() {
  PassRegistration<DropSchedulePass>();
}

/// Registration hook for the Transform dialect interpreter pass.
void mlir::linalg::transform::registerTransformDialectInterpreterPass() {
  PassRegistration<TransformDialectInterpreter>();
}
