// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
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
    : public mlir::transform::TransformInterpreterPassBase<
          TransformDialectInterpreterPass,
          iree_compiler::TransformDialectInterpreterBase> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::iree_compiler::registerTransformDialectTranslationDependentDialects(
        registry);
  }

  // We don't register libraries here because we expect them to be pre-loaded
  // much earlier on in the compiler pipeline.
  TransformDialectInterpreterPass(
      StringRef transformFileName = StringRef(),
      StringRef debugPayloadRootTag = StringRef(),
      StringRef debugTransformRootTag = StringRef()) {
    this->transformFileName = transformFileName.str();
    this->debugPayloadRootTag = debugPayloadRootTag.str();
    this->debugTransformRootTag = debugTransformRootTag.str();
  }
  TransformDialectInterpreterPass(const TransformDialectInterpreterPass &pass) =
      default;
};
} // namespace

namespace mlir {
namespace iree_compiler {

extern llvm::cl::opt<std::string> clCodegenTransformDialectTestName;
static llvm::cl::opt<std::string> clCodegenTransformDialectDebugPayloadTag(
    "iree-codegen-transform-dialect-debug-payload-tag",
    llvm::cl::desc("tag attribute value for the transform dialect interpreter "
                   "payload root operation"),
    llvm::cl::init(""));
static llvm::cl::opt<std::string> clCodegenTransformDialectDebugTransformTag(
    "iree-codegen-transform-dialect-debug-transform-tag",
    llvm::cl::desc(
        "tag attribute value for the transform dialect transform op container"),
    llvm::cl::init(""));

/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass>
createTransformDialectInterpreterPass(llvm::StringRef transformFileName,
                                      llvm::StringRef debugPayloadRootTag,
                                      llvm::StringRef debugTransformRootTag) {
  // If the strategy filename is prefixed with `@`, it refers to a library
  // call.
  std::string clFileName = !clCodegenTransformDialectTestName.empty() &&
                                   clCodegenTransformDialectTestName[0] != '@'
                               ? clCodegenTransformDialectTestName
                               : std::string();
  return std::make_unique<TransformDialectInterpreterPass>(
      transformFileName.empty() ? clFileName : transformFileName,
      debugPayloadRootTag.empty() ? clCodegenTransformDialectDebugPayloadTag
                                  : debugPayloadRootTag,
      debugTransformRootTag.empty() ? clCodegenTransformDialectDebugTransformTag
                                    : debugTransformRootTag);
}
} // namespace iree_compiler
} // namespace mlir
