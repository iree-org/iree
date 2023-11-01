// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for codegen.
/// This needs to be its own pass because the registration mechanism and ops
/// available are different than for other interpreters.
class TransformDialectInterpreterPass
    : public iree_compiler::TransformDialectInterpreterBase<
          TransformDialectInterpreterPass> {
public:
  TransformDialectInterpreterPass(StringRef libraryFileName = StringRef(),
                                  StringRef entryPoint = StringRef()) {
    this->libraryFileName = libraryFileName.str();
    this->entryPoint = entryPoint.str();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::iree_compiler::registerTransformDialectTranslationDependentDialects(
        registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    transform::TransformOptions options;
    if (entryPoint.empty()) {
      entryPoint =
          transform::TransformDialect::kTransformEntryPointSymbolName.str();
    }
    auto dialect = context->getOrLoadDialect<
        mlir::iree_compiler::IREE::Codegen::IREECodegenDialect>();
    FailureOr<ModuleOp> maybeTransformLibrary;
    if (!libraryFileName.empty()) {
      maybeTransformLibrary =
          dialect->getOrLoadTransformLibraryModule(libraryFileName);
    }

    Operation *payloadRoot = getOperation();
    ModuleOp transformModule =
        succeeded(maybeTransformLibrary) ? *maybeTransformLibrary : ModuleOp();
    Operation *transformEntryPoint = transform::detail::findTransformEntryPoint(
        getOperation(), transformModule, entryPoint);
    if (!transformEntryPoint) {
      Operation *transformModuleOrPayloadRoot =
          transformModule ? transformModule : payloadRoot;
      transformModuleOrPayloadRoot->emitError()
          << "failed to find transform entry point '" << entryPoint << "'";
      return signalPassFailure();
    }
    if (failed(transform::applyTransformNamedSequence(
            payloadRoot, transformEntryPoint, transformModule,
            options.enableExpensiveChecks(true))))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace iree_compiler {

extern llvm::cl::opt<std::string> clCodegenTransformDialectStrategyName;
extern llvm::cl::opt<std::string> clCodegenTransformDialectLibraryFileName;

/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass> createTransformDialectInterpreterPass() {
  return std::make_unique<TransformDialectInterpreterPass>(
      clCodegenTransformDialectLibraryFileName,
      clCodegenTransformDialectStrategyName);
}
} // namespace iree_compiler
} // namespace mlir
