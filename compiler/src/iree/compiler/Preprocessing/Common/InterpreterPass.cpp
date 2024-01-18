// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace mlir;

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_INTERPRETERPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::iree_compiler::Preprocessing

namespace {
class InterpreterPass
    : public iree_compiler::Preprocessing::impl::InterpreterPassBase<
          InterpreterPass> {
public:
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
        getOperation(), *transformModule, "__preprocessing_main");
    if (!transformEntryPoint) {
      getOperation()->emitError() << "could not find transform entry point "
                                     "__preprocessing_main in transform module";
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
