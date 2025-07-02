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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-debug-patch-func-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PATCHFUNCOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

llvm::cl::opt<std::string> clCodegenPatchedFuncOpsFileName(
    "iree-codegen-debug-patched-func-ops-file-name", llvm::cl::desc("TBD"),
    llvm::cl::init(""));

static LogicalResult getMatchedFuncOp(StringRef fileName,
                                      FunctionOpInterface funcOp,
                                      FunctionOpInterface &replacement) {
  std::optional<ModuleOp> module;
  auto dialect = funcOp->getContext()
                     ->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
  auto maybeModule =
      dialect->getOrLoadPatchedFuncOpsForDebugging(std::string(fileName));
  if (failed(maybeModule)) {
    return failure();
  }
  module = *maybeModule;
  LDBG("--found patching library @" << fileName);

  for (auto candidate : module->getOps<FunctionOpInterface>()) {
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
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() final;
};
} // namespace

void PatchFuncOpsPass::runOnOperation() {
  if (clCodegenPatchedFuncOpsFileName.empty()) {
    LDBG("skip, because no file is provided");
    return;
  }
  auto moduleOp = getOperation();
  StringRef fileName = llvm::StringRef(clCodegenPatchedFuncOpsFileName);
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    FunctionOpInterface replacement;
    if (failed(getMatchedFuncOp(fileName, funcOp, replacement))) {
      funcOp.emitError() << "failed to parse file: "
                         << clCodegenPatchedFuncOpsFileName;
      return signalPassFailure();
    }

    if (!replacement) {
      LDBG("--did not find matching funcOp" << funcOp.getName());
      continue;
    }

    funcOp.getCallableRegion()->takeBody(*replacement.getCallableRegion());
  }
}

} // namespace mlir::iree_compiler
