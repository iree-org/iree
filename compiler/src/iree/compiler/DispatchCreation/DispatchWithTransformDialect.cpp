// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_DISPATCHWITHTRANSFORMDIALECTPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for dispatch region
/// formation. This needs to be its own pass because the registration mechanism
/// and ops available are different than for other interpreters.
namespace {
struct DispatchWithTransformDialectPass final
    : public impl::DispatchWithTransformDialectPassBase<
          DispatchWithTransformDialectPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Load the module from the spec path. The module will be unloaded once the
    // pass finishes.
    OwningOpRef<ModuleOp> transformModule;
    if (failed(transform::detail::assembleTransformLibraryFromPaths(
            context, transformSpecPath, transformModule)))
      return signalPassFailure();
    Operation *payloadRoot = getOperation();
    Operation *transformEntryPoint = transform::detail::findTransformEntryPoint(
        getOperation(), *transformModule, "__transform_main");
    if (!transformEntryPoint) {
      getOperation()->emitError() << "could not find transform entry point "
                                     "__transform_main in transform module";
      return signalPassFailure();
    }

    if (failed(transform::applyTransformNamedSequence(
            payloadRoot, transformEntryPoint, *transformModule,
            options.enableExpensiveChecks(!disableExpensiveChecks)))) {
      return signalPassFailure();
    }
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
