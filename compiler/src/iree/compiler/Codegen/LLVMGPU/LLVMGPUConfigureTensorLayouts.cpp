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

static LogicalResult setContractionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                          RewriterBase &rewriter,
                                          linalg::LinalgOp contract,
                                          bool promoteLhs = true,
                                          bool promoteRhs = true) {
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
  if (promoteLhs) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promoteRhs) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

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

static LogicalResult setConvolutionAnchor(IREE::GPU::MMAScheduleAttr schedule,
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

/// Let's assume we have an matmul intrinsic (@) doing a matmul
/// ((M, K) X (K, N)) which produces a particular layout:
///
/// C = A @ B
///
/// If we transpose and swap the operands, we can keep the same matmul
/// intrinsic, but transpose the layout of the output intrinsic:
///
/// A.T = transpose(A)
/// B.T = transpose(B)
/// C.T = B.T @ A.T
/// C = transpose(C.T)
///
/// This is useful when the "@" instruction that the hardware lowers to
/// has a specific thread layout but the further uses of C expects a transposed
/// layout to the produced layout.
///
/// For example, for "@" lowering to AMDGPU MFMA instructions, the operands
/// have layout L and L.T and the result has the layout L.T .
/// So if you have a chain of matmuls:
///
/// C (L.T) = A (L) @ B (L.T)
/// E (L.T) = C (L.T)  @ D (L.T)
///            ^^^^^^^
///            Expected layout by instruction is L
///
/// To fix this, we can apply this transformation on the first matrix:
///
/// C.T (L.T) = B.T (L) @ A (L.T)
/// C   (L)   = transpose C.T (L.T)
/// E   (L.T) = C (L)  @ D (L.T)
///            ^^^^^
///            Layout matches the instruction!
///
/// Note that the mathematical formula
///   C = A @ B --> C.T = B.T @ A.T
/// is only defined on standard "@" function, it may be a different
/// transformation for other indexing maps.
///
/// For linalg operands, since the indexing maps are part of the op defination,
/// we can achieve the same transformation by simply swapping the operands.
static void swapOperandsToTransposeIntrinsic(RewriterBase &rewriter,
                                             linalg::GenericOp contractOp) {
  Value lhs = contractOp->getOperand(0);
  Value rhs = contractOp->getOperand(1);

  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  std::swap(indexingMaps[0], indexingMaps[1]);

  contractOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(indexingMaps));
  contractOp->setOperand(0, rhs);
  contractOp->setOperand(1, lhs);
}

static IREE::GPU::MMAScheduleAttr
transposeSchedule(RewriterBase &rewriter, IREE::GPU::MMAScheduleAttr schedule) {
  return rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
      schedule.getIntrinsic(), schedule.getSubgroupNCount(),
      schedule.getSubgroupMCount());
}

static LogicalResult
setAttentionMatmulAnchor(IREE::GPU::MMAScheduleAttr schedule,
                         RewriterBase &rewriter, linalg::LinalgOp qkMatmul,
                         linalg::LinalgOp pvMatmul) {
  // TODO: Add SIMT fallback.
  if (!schedule) {
    return pvMatmul->emitError("missing mma schedule for contraction");
  }

  // Check if the intrinsic output for qkMatmul can be reused for pvMatmul.
  // We know that pvMatmul takes result of qkMatmul as it's lhs.
  // If the intrinsic output of pvMatmul can be used as rhs of pvMatmul,
  // we swap operands of both contracts to get output as transposed intrinsic.
  bool reuseIntrinsicOutput = false;
  bool transposeIntrinsic = false;

  auto intrinsic = cast<IREE::GPU::MMAAttr>(schedule.getIntrinsic());
  IREE::GPU::MMASingleSubgroupLayout lhsLayout =
      intrinsic.getASingleSubgroupLayout();
  IREE::GPU::MMASingleSubgroupLayout rhsLayout =
      intrinsic.getBSingleSubgroupLayout();
  IREE::GPU::MMASingleSubgroupLayout outLayout =
      intrinsic.getCSingleSubgroupLayout();

  auto matchLayout = [](IREE::GPU::MMASingleSubgroupLayout layoutA,
                        IREE::GPU::MMASingleSubgroupLayout layoutB) -> bool {
    return (layoutA.element == layoutB.element) &&
           (layoutA.thread == layoutB.thread) &&
           (layoutA.tstrides == layoutB.tstrides);
  };

  // TODO: Move this check to KernelConfig and set appropriate attributes
  // in lowering_config for the operation. This allows us to check shared
  // memory usage and decide what kind of pipelining we can do.
  if (matchLayout(outLayout, lhsLayout)) {
    reuseIntrinsicOutput = true;
  } else if (matchLayout(outLayout, rhsLayout)) {
    reuseIntrinsicOutput = true;
    transposeIntrinsic = true;
  }

  // subgroup_n count for attention matmul is always 1, because it is the
  // reduction dimension. The subgroup_n count is in reality, for the pvMatmul.
  IREE::GPU::MMAScheduleAttr qkSchedule =
      rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
          schedule.getIntrinsic(),
          /*subgroup_m_count=*/schedule.getSubgroupMCount(),
          /*subgroup_n_count=*/1);
  IREE::GPU::MMAScheduleAttr pvSchedule = schedule;

  // Transpose the intrinsic if requested. See docs for
  // swapOperandsToTransposeIntrinsic for more information on why this is done.
  if (transposeIntrinsic) {
    auto qkGeneric = dyn_cast<linalg::GenericOp>(qkMatmul.getOperation());
    auto pvGeneric = dyn_cast<linalg::GenericOp>(pvMatmul.getOperation());
    if (!qkGeneric || !pvGeneric) {
      pvMatmul->emitOpError("Non generic qkMatmul/pvMatmul transpose intrinsic "
                            "not yet implemented");
      return failure();
    }
    swapOperandsToTransposeIntrinsic(rewriter, qkGeneric);
    swapOperandsToTransposeIntrinsic(rewriter, pvGeneric);
    qkSchedule = transposeSchedule(rewriter, qkSchedule);
    pvSchedule = transposeSchedule(rewriter, pvSchedule);
  }

  if (failed(setContractionAnchor(qkSchedule, rewriter, qkMatmul))) {
    return failure();
  }

  // Do not promote lhs of pvMatmul if we are reusing the intrinsic output.
  bool promoteLhs = !reuseIntrinsicOutput;
  bool promoteRhs = true;
  if (transposeIntrinsic) {
    std::swap(promoteLhs, promoteRhs);
  }

  return setContractionAnchor(pvSchedule, rewriter, pvMatmul, promoteLhs,
                              promoteRhs);
}

static Operation *getOpWithAttr(Operation *root, StringRef attr) {
  Operation *result = nullptr;
  WalkResult walkResult = root->walk([&](Operation *op) {
    if (op->hasAttr(attr)) {
      if (result) {
        return WalkResult::interrupt();
      }
      result = op;
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return nullptr;
  }
  return result;
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

    auto attentionQKMatmul = dyn_cast_or_null<linalg::LinalgOp>(
        getOpWithAttr(func, "attention_qk_matmul"));
    auto attentionPVMatmul = dyn_cast_or_null<linalg::LinalgOp>(
        getOpWithAttr(func, "attention_pv_matmul"));

    if (attentionQKMatmul && !attentionPVMatmul) {
      func->emitError("Expected attention attributes to be set properly");
      return signalPassFailure();
    }

    if (!attentionQKMatmul && attentionPVMatmul) {
      func->emitError("Expected attention attributes to be set properly");
      return signalPassFailure();
    }

    func->walk([&](linalg::LinalgOp linalgOp) {
      if (linalgOp == attentionQKMatmul || linalgOp == attentionPVMatmul) {
        return WalkResult::advance();
      }

      if (linalg::isaContractionOpInterface(linalgOp)) {
        contracts.push_back(linalgOp);
      } else if (succeeded(linalg::inferConvolutionDims(linalgOp))) {
        convs.push_back(linalgOp);
      }
      return WalkResult::advance();
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

    if (attentionQKMatmul && attentionPVMatmul) {
      if (failed(setAttentionMatmulAnchor(
              scheduleAttr, rewriter, attentionQKMatmul, attentionPVMatmul))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
