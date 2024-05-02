// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "iree/compiler/Preprocessing/Common/Utilities.h"

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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/CPPGen.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-preprocessing-apply-pdll-patterns"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_APPLYPDLLPATTERNSPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

} // namespace mlir::iree_compiler::Preprocessing

// Populate patterns from PDLL files.
static LogicalResult
populatePDLLModuleFromFileName(MLIRContext *context,
                               RewritePatternSet &patterns,
                               llvm::StringRef pdllModuleFileName) {
// TODO(#17233)
// Its easier to give these functions definitions than try to disable
// tblgen, just return failure
#ifndef IREE_LLVM_BUNDLED
  return failure();
#else
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(pdllModuleFileName, &errorMessage);
  if (!memoryBuffer) {
    return emitError(FileLineColLoc::get(
               StringAttr::get(context, pdllModuleFileName), 0, 0))
           << "failed to open pattern module file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());

  // parse file and build pdll ast module
  pdll::ods::Context pdllOdsContext;
  pdll::ast::Context pdllAstContext(pdllOdsContext);
  FailureOr<pdll::ast::Module *> pdllModule =
      pdll::parsePDLLAST(pdllAstContext, sourceMgr);
  if (failed(pdllModule)) {
    return failure();
  }

  PDLPatternModule pdlModule =
      pdll::codegenPDLLToMLIR(context, pdllAstContext, sourceMgr, **pdllModule);
  pdlModule.registerRewriteFunction("rewriteAsFlowDispatch",
                                    rewriteAsFlowDispatch);
  pdlModule.registerConstraintFunction("checkTensorElementType",
                                       checkTensorElementType);
  patterns.insert(std::move(pdlModule));
  return success();
#endif
}

namespace {

class ApplyPDLLPatternsPass
    : public iree_compiler::Preprocessing::impl::ApplyPDLLPatternsPassBase<
          ApplyPDLLPatternsPass> {
public:
  using iree_compiler::Preprocessing::impl::ApplyPDLLPatternsPassBase<
      ApplyPDLLPatternsPass>::ApplyPDLLPatternsPassBase;

  LogicalResult initialize(MLIRContext *context) override {
// TODO(#17233)
#ifndef IREE_LLVM_BUNDLED
    return failure();
#else
    if (patternsFile.empty()) {
      return success();
    }
    RewritePatternSet tmpPatterns(context);
    if (failed(populatePDLLModuleFromFileName(context, tmpPatterns,
                                              patternsFile))) {
      return failure();
    }
    patterns = std::move(tmpPatterns);
    return success();
#endif
  }

  void runOnOperation() override {
// TODO(#17233)
#ifdef IREE_LLVM_BUNDLED
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
#endif
  }

private:
  /// Loaded PDL patterns
  FrozenRewritePatternSet patterns;
};

} // namespace
