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

  FailureOr<VectorContractOpInfo> opInfo =
      VectorContractOpInfo::inferFromIndexingMaps(
          contract.getIndexingMapsArray());
  assert(succeeded(opInfo) && "contraction should have been inferred");

  auto layouts = schedule.getContractionLayout(opInfo.value(), contract);
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

LogicalResult setConvolutionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                   RewriterBase &rewriter,
                                   linalg::LinalgOp conv) {
  // TODO: Add SIMT fallback.
  if (!schedule) {
    return conv->emitError("missing mma schedule for convolution");
  }

  // This function should have only be called on a convolution op.
  FailureOr<linalg::ConvolutionDimensions> convDims =
      linalg::inferConvolutionDims(conv);
  assert(succeeded(convDims) &&
         "cannot set convolution anchor on non convolution op");

  // Only convs with unit filter dims can be directly converted to matmul.
  SmallVector<int64_t> shape = conv.getStaticLoopRanges();
  if (!llvm::all_of(convDims->filterLoop,
                    [&shape](unsigned dim) { return shape[dim] == 1; })) {
    return failure();
  }

  llvm::SmallBitVector filterDims(conv.getNumLoops(), false);
  for (unsigned idx : convDims->filterLoop) {
    filterDims.set(idx);
  }

  SmallVector<AffineMap> maps = conv.getIndexingMapsArray();
  for (AffineMap &map : maps) {
    map = projectDims(map, filterDims, /*compressDimsFlag=*/false);
  }

  FailureOr<VectorContractOpInfo> opInfo =
      VectorContractOpInfo::inferFromIndexingMaps(maps);
  assert(succeeded(opInfo) &&
         "unit filter dim convolution should have been infered");

  auto layouts = schedule.getContractionLayout(opInfo.value(), conv);
  if (failed(layouts)) {
    return conv->emitError("cannot get concrete layout for convolution");
  }

  auto [aLayout, bLayout, cLayout] = *layouts;
  Location loc = conv.getLoc();

  Value lhs = conv->getOperand(0);
  Value rhs = conv->getOperand(1);
  Value acc = conv->getOperand(2);

  // Set layouts for lhs, rhs and acc.
  rewriter.setInsertionPoint(conv);
  auto layoutedLhs = rewriter.create<IREE::VectorExt::ToLayoutOp>(
      loc, lhs.getType(), lhs, aLayout);
  auto layoutedRhs = rewriter.create<IREE::VectorExt::ToLayoutOp>(
      loc, rhs.getType(), rhs, bLayout);
  auto layoutedAcc = rewriter.create<IREE::VectorExt::ToLayoutOp>(
      loc, acc.getType(), acc, cLayout);

  // Promote matmul lhs and rhs.
  // TODO: We should read this from the lowering_config on the operation.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  layoutedLhs.setSharedMemoryConversion(true);
  layoutedRhs.setSharedMemoryConversion(true);

  conv->setOperand(0, layoutedLhs.getResult());
  conv->setOperand(1, layoutedRhs.getResult());
  conv->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(conv);
  auto toLayout = rewriter.create<IREE::VectorExt::ToLayoutOp>(
      loc, conv->getResult(0).getType(), conv->getResult(0), cLayout);
  rewriter.replaceAllUsesExcept(conv->getResult(0), toLayout.getResult(),
                                toLayout);

  return success();
}

LogicalResult setAttentionMatmulAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                       RewriterBase &rewriter,
                                       linalg::LinalgOp contract) {
  // TODO: Add SIMT fallback.
  if (!schedule) {
    return contract->emitError("missing mma schedule for contraction");
  }

  if (contract->hasAttr("attention_qk_matmul")) {
    // subgroup_n count for attention matmul is always 1, because it is the
    // reduction dimension. The subgroup_n count is in reality, for the second
    // matmul.
    IREE::GPU::MMAScheduleAttr qkSchedule =
        rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
            schedule.getIntrinsic(),
            /*subgroup_m_count=*/schedule.getSubgroupMCount(),
            /*subgroup_n_count=*/1);
    return setContractionAnchor(qkSchedule, rewriter, contract);
  }

  if (contract->hasAttr("attention_pv_matmul")) {
    // subgroup_n count for attention matmul is always 1, because it is the
    // reduction dimension. The subgroup_n count is in reality, for the second
    // matmul.
    return setContractionAnchor(schedule, rewriter, contract);
  }

  return contract->emitError("attention matmul should have either "
                             "attention_qk_matmul or attention_pv_matmul set");
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

    // Vector layout option setter aimed at contractions and convolutions. For
    // now, layout setting for other problems like reductions is TODO.
    SmallVector<linalg::LinalgOp> contracts;
    SmallVector<linalg::LinalgOp> convs;
    SmallVector<linalg::LinalgOp> attentionMatmuls;

    func->walk([&](linalg::LinalgOp linalgOp) {
      if (linalg::isaContractionOpInterface(linalgOp)) {
        if (linalgOp->hasAttr("attention_qk_matmul") ||
            linalgOp->hasAttr("attention_pv_matmul")) {
          attentionMatmuls.push_back(linalgOp);
        } else {
          contracts.push_back(linalgOp);
        }
      } else if (succeeded(linalg::inferConvolutionDims(linalgOp))) {
        convs.push_back(linalgOp);
      }
    });

    IRRewriter rewriter(func);

    for (linalg::LinalgOp contract : contracts) {
      if (failed(setContractionAnchor(scheduleAttr, rewriter, contract))) {
        return signalPassFailure();
      }
    }

    for (linalg::LinalgOp conv : convs) {
      if (failed(setConvolutionAnchor(scheduleAttr, rewriter, conv))) {
        return signalPassFailure();
      }
    }

    for (linalg::LinalgOp attentionMatmul : attentionMatmuls) {
      if (failed(setAttentionMatmulAnchor(scheduleAttr, rewriter,
                                          attentionMatmul))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
