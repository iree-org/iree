// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-pcf-convert-forall-to-loops"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_CONVERTFORALLTOLOOPSPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

// DO NOT SUBMIT
struct ConvertForallToLoopsPass final
    : impl::ConvertForallToLoopsPassBase<ConvertForallToLoopsPass> {
  void runOnOperation() override;
};

void ConvertForallToLoopsPass::runOnOperation() {
  // Collect all mapping-less forall ops to convert to sequential pcf.loop ops.
  SmallVector<scf::ForallOp> opsToConvert;
  getOperation()->walk([&](scf::ForallOp forallOp) {
    std::optional<ArrayAttr> mapping = forallOp.getMapping();
    if (!mapping || mapping->empty()) {
      opsToConvert.push_back(forallOp);
    }
  });

  IRRewriter rewriter(getOperation());
  PCF::ScopeAttr sequentialScope = PCF::SequentialAttr::get(&getContext());
  for (auto forallOp : opsToConvert) {
    rewriter.setInsertionPoint(forallOp);
    if (failed(convertForallToPCF(rewriter, forallOp, sequentialScope))) {
      forallOp->emitOpError("failed to convert forall");
      return signalPassFailure();
    }
  }
}

} // namespace

} // namespace mlir::iree_compiler::IREE::PCF
