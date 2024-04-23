// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Utilities.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-apply-pdl-patterns"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_APPLYPDLPATTERNSPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::iree_compiler::Preprocessing

// Populate patterns from files.
static LogicalResult
populatePDLModuleFromFileName(MLIRContext *context, RewritePatternSet &patterns,
                              llvm::StringRef pdlModuleFileName) {
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(pdlModuleFileName, &errorMessage);
  if (!memoryBuffer) {
    return emitError(FileLineColLoc::get(
               StringAttr::get(context, pdlModuleFileName), 0, 0))
           << "failed to open pattern module file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  PDLPatternModule pdlModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  pdlModule.registerRewriteFunction("rewriteAsFlowDispatch",
                                    rewriteAsFlowDispatch);
  pdlModule.registerConstraintFunction("checkTensorElementType",
                                       checkTensorElementType);
  patterns.insert(std::move(pdlModule));
  return success();
}

namespace {

class ApplyPDLPatternsPass
    : public iree_compiler::Preprocessing::impl::ApplyPDLPatternsPassBase<
          ApplyPDLPatternsPass> {
public:
  using iree_compiler::Preprocessing::impl::ApplyPDLPatternsPassBase<
      ApplyPDLPatternsPass>::ApplyPDLPatternsPassBase;

  LogicalResult initialize(MLIRContext *context) override {
    if (patternsFile.empty()) {
      return success();
    }
    RewritePatternSet tmpPatterns(context);
    if (failed(populatePDLModuleFromFileName(context, tmpPatterns,
                                             patternsFile))) {
      return failure();
    }
    patterns = std::move(tmpPatterns);
    return success();
  }

  void runOnOperation() override {
    // If there is nothing to do then return.
    if (!patterns.getPDLByteCode()) {
      return;
    }

    // Apply the patterns.
    auto operation = getOperation();
    if (failed(applyPatternsAndFoldGreedily(operation, patterns))) {
      operation->emitOpError("failed to apply patterns specified in ")
          << patternsFile;
      return signalPassFailure();
    }
  }

private:
  /// Loaded PDL patterns
  FrozenRewritePatternSet patterns;
};
} // namespace
