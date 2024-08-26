// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-llvmgpu-configure-vector-layouts"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONFIGURETENSORLAYOUTSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

LogicalResult setContractionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                   RewriterBase &rewriter,
                                   linalg::LinalgOp contract) {
  // TODO: Add SIMT fallback.
  if (!schedule) {
    return contract->emitError("missing mma schedule for contraction");
  }

  // This function should have only be called on a contraction op.
  assert(linalg::isaContractionOpInterface(contract) &&
         "cannot set contraction anchor on non contraction op");

  auto layouts = schedule.getContractionLayout(contract);
  if (failed(layouts)) {
    return contract->emitError("cannot get concrete layout for contraction");
  }

  auto [aLayout, bLayout, cLayout] = *layouts;
  Location loc = contract.getLoc();

  Value lhs = contract->getOperand(0);
  Value rhs = contract->getOperand(1);
  Value acc = contract->getOperand(2);

  // Set layouts for lhs, rhs and acc.
  rewriter.setInsertionPoint(contract);
  auto layoutedLhs =
      rewriter.create<IREE::VectorExt::ToLayoutOp>(loc, lhs, aLayout);
  auto layoutedRhs =
      rewriter.create<IREE::VectorExt::ToLayoutOp>(loc, rhs, bLayout);
  auto layoutedAcc =
      rewriter.create<IREE::VectorExt::ToLayoutOp>(loc, acc, cLayout);

  // Promote matmul lhs and rhs.
  // TODO: We should read this from the lowering_config on the operation.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  layoutedLhs.setSharedMemoryConversion(true);
  layoutedRhs.setSharedMemoryConversion(true);

  contract->setOperand(0, layoutedLhs.getResult());
  contract->setOperand(1, layoutedRhs.getResult());
  contract->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(contract);
  auto toLayout = rewriter.create<IREE::VectorExt::ToLayoutOp>(
      loc, contract->getResult(0), cLayout);
  rewriter.replaceAllUsesExcept(contract->getResult(0), toLayout.getResult(),
                                toLayout);

  return success();
}

struct LLVMGPUConfigureTensorLayoutsPass final
    : impl::LLVMGPUConfigureTensorLayoutsPassBase<
          LLVMGPUConfigureTensorLayoutsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
    auto scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
        configDict.get(scheduleAttrName));

    // Vector layout option setter aimed at contractions. For now, layout
    // setting for other problems like reductions is TODO.
    SmallVector<linalg::LinalgOp> contracts;

    func->walk([&](linalg::LinalgOp linalgOp) {
      if (linalg::isaContractionOpInterface(linalgOp)) {
        contracts.push_back(linalgOp);
      }
    });

    IRRewriter rewriter(func);

    for (linalg::LinalgOp contract : contracts) {
      if (failed(setContractionAnchor(scheduleAttr, rewriter, contract))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
