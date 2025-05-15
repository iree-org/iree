// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-attr-based-pipeline"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_ATTRBASEDPIPELINEPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc" // IWYU pragma: export

namespace {

struct AttrBasedPipelinePass
    : public impl::AttrBasedPipelinePassBase<AttrBasedPipelinePass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Method to get the pass manager nested on a particular operation. There does
// not seem to be a way to do this without specializing on the op itself.
// When possible to do so, this method could be deleted.
static std::optional<OpPassManager>
getFunctionOpInterfacePassManager(FunctionOpInterface interfaceOp) {
  return TypeSwitch<Operation *, std::optional<OpPassManager>>(
             interfaceOp.getOperation())
      .Case<func::FuncOp, IREE::Util::FuncOp>(
          [&](auto funcOp) { return OpPassManager(funcOp.getOperationName()); })
      .Default([&](Operation *op) { return std::nullopt; });
}

void AttrBasedPipelinePass::runOnOperation() {
  auto op = getOperation();
  SmallVector<FunctionOpInterface> funcLikeOps;
  op->walk([&](FunctionOpInterface funcLikeOp) {
    funcLikeOps.push_back(funcLikeOp);
  });

  for (auto funcLikeOp : funcLikeOps) {
    auto attr = dyn_cast_or_null<DictionaryAttr>(
        IREE::Util::getPassPipelinesAttribute(funcLikeOp));
    if (!attr) {
      continue;
    }
    auto passPipelineAttr = attr.getAs<IREE::Util::PipelineAttr>(pipelineName);
    if (!passPipelineAttr) {
      funcLikeOp.emitRemark(
          "expected preprocessing_pipeline attribute to be a `StringAttr` that "
          "specifies the pass pipeline to apply");
      continue;
    }
    LLVM_DEBUG({ llvm::dbgs() << "Parsed Attribute : " << passPipelineAttr; });

    std::optional<OpPassManager> passManager =
        getFunctionOpInterfacePassManager(funcLikeOp);
    if (!passManager) {
      continue;
    }

    std::string pipelineStr =
        passPipelineAttr.getPipelineString().getValue().str();
    if (failed(parsePassPipeline(pipelineStr, *passManager))) {
      funcLikeOp->emitOpError("failed to populate pass manager specified by : ")
          << pipelineStr;
      return signalPassFailure();
    }

    if (failed(runPipeline(*passManager, funcLikeOp))) {
      funcLikeOp->emitOpError("failed to run pass specified as attribute : ")
          << pipelineStr;
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::Util
