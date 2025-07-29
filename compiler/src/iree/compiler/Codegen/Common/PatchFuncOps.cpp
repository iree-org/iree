// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-debug-patch-func-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PATCHFUNCOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

llvm::cl::opt<std::string> clCodegenPatchedFuncOpsFileName(
    "iree-codegen-debug-patched-func-ops-file-name",
    llvm::cl::desc("File path to a module containing func ops that will be "
                   "used for patching existing func ops."),
    llvm::cl::init(""));

static LogicalResult getMatchedFuncOp(StringRef fileName,
                                      FunctionOpInterface funcOp,
                                      FunctionOpInterface &replacement) {
  std::optional<ModuleOp> moduleOp;
  auto dialect = funcOp->getContext()
                     ->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
  FailureOr<ModuleOp> maybeModuleOp =
      dialect->getOrLoadPatchedFuncOpsForDebugging(fileName.str());
  if (failed(maybeModuleOp)) {
    return failure();
  }
  moduleOp = *maybeModuleOp;
  LDBG() << "--found patching library @" << fileName;

  for (auto candidate : moduleOp->getOps<FunctionOpInterface>()) {
    if (funcOp.getName() == candidate.getName()) {
      replacement = candidate;
      break;
    }
  }
  return success();
}

namespace {
struct PatchFuncOpsPass : public impl::PatchFuncOpsPassBase<PatchFuncOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void PatchFuncOpsPass::runOnOperation() {
  if (clCodegenPatchedFuncOpsFileName.empty()) {
    LDBG() << "skip, because no file is provided";
    return;
  }
  Operation *startOp = getOperation();
  while (!isa_and_present<ModuleOp>(startOp)) {
    startOp = startOp->getParentOp();
  }
  ModuleOp moduleOp = dyn_cast<ModuleOp>(startOp);
  if (!moduleOp) {
    getOperation()->emitError(
        "can not find ModuleOp in parent chain, wrong scope?");
    return signalPassFailure();
  }
  StringRef fileName = llvm::StringRef(clCodegenPatchedFuncOpsFileName);
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    FunctionOpInterface replacement;
    if (failed(getMatchedFuncOp(fileName, funcOp, replacement))) {
      funcOp.emitError() << "failed to parse file: "
                         << clCodegenPatchedFuncOpsFileName;
      return signalPassFailure();
    }
    if (!replacement) {
      LDBG() << "--did not find matching funcOp" << funcOp.getName();
      continue;
    }
    LDBG() << "--found matching funcOp" << funcOp.getName();
    // Do not use takeBody method because it drops the reference in the module
    // op that contains the patches.
    IRMapping mapper;
    funcOp.getBlocks().clear();
    replacement.getFunctionBody().cloneInto(&funcOp.getFunctionBody(), mapper);
  }
}

} // namespace mlir::iree_compiler
