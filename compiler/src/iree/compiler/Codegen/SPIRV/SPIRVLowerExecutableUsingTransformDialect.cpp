// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"

namespace mlir::iree_compiler {

namespace {
class SPIRVLowerExecutableUsingTransformDialectPass
    : public SPIRVLowerExecutableUsingTransformDialectBase<
          SPIRVLowerExecutableUsingTransformDialectPass> {
public:
  void runOnOperation() override;
};
} // namespace

void SPIRVLowerExecutableUsingTransformDialectPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto funcOps = moduleOp.getOps<FunctionOpInterface>();

  if (funcOps.empty() || !llvm::hasSingleElement(funcOps)) {
    // Can only handle dispatches with single functions on the transform dialect
    // path.
    return;
  }

  auto funcOp = *funcOps.begin();
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo || translationInfo.getDispatchLoweringPassPipeline() !=
                              IREE::Codegen::DispatchLoweringPassPipeline::
                                  TransformDialectCodegen) {
    return;
  }

  // Run the interpreter and drop schedule passes.
  SymbolRefAttr codegenSpec = translationInfo.getCodegenSpec();
  StringRef entryPoint =
      codegenSpec ? codegenSpec.getLeafReference() : StringRef("");
  OpPassManager modulePassManager(ModuleOp::getOperationName());
  modulePassManager.addPass(
      iree_compiler::createTransformDialectInterpreterPass(entryPoint));
  modulePassManager.addPass(createDropSchedulePass());

  OpPassManager &funcPassManager = modulePassManager.nest<func::FuncOp>();
  funcPassManager.addPass(createGenericVectorizationPass());
  addSPIRVVectorLoweringPasses(funcPassManager);

  if (failed(runPipeline(modulePassManager, moduleOp))) {
    moduleOp.emitOpError("failed to run transform dialect passes");
    return signalPassFailure();
  }

  // Make sure that the translation info is set to `None` to avoid using
  // other pass pipelines.
  auto translationInfoModified = getTranslationInfo(funcOp);
  if (!translationInfoModified ||
      translationInfoModified.getDispatchLoweringPassPipeline() !=
          IREE::Codegen::DispatchLoweringPassPipeline::None) {
    funcOp->emitOpError("expected transform dialect lowering to set the "
                        "translation_info to use None");
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createSPIRVLowerExecutableUsingTransformDialectPass() {
  return std::make_unique<SPIRVLowerExecutableUsingTransformDialectPass>();
}

} // namespace mlir::iree_compiler
