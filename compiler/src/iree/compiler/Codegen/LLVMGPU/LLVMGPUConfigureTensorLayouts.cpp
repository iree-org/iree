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

using IREE::VectorExt::NestedLayoutAttr;
using IREE::VectorExt::ToLayoutOp;
using IREE::VectorExt::VectorLayoutInterface;

namespace {

static SmallVector<bool> getPromotedOperands(Operation *op) {
  SmallVector<bool> promotedOperands(op->getNumOperands(), false);

  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!config) {
    return promotedOperands;
  }

  std::optional<SmallVector<int64_t>> promoteConfig =
      config.getPromotedOperandList();
  if (!promoteConfig) {
    return promotedOperands;
  }

  for (int64_t operand : promoteConfig.value()) {
    promotedOperands[operand] = true;
  }

  return promotedOperands;
}

static IREE::GPU::MmaInterfaceAttr getIntrinsic(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  IREE::GPU::MmaInterfaceAttr mmaIntrinsic = config.getMmaKind();
  assert(mmaIntrinsic && "Cannot find intrinsic in lowering config.");
  return mmaIntrinsic;
}

static int64_t getSubgroupMCount(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  return *config.getSubgroupMCount();
}

static int64_t getSubgroupNCount(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  return *config.getSubgroupNCount();
}

static LogicalResult setContractionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                          SmallVector<bool> promotedOperands,
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
      rewriter.create<ToLayoutOp>(loc, lhs, aLayout, schedule.getIntrinsic());
  auto layoutedRhs =
      rewriter.create<ToLayoutOp>(loc, rhs, bLayout, schedule.getIntrinsic());
  auto layoutedAcc =
      rewriter.create<ToLayoutOp>(loc, acc, cLayout, schedule.getIntrinsic());

  // Promote matmul lhs and rhs.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  if (promotedOperands[0]) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[1]) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[2]) {
    layoutedAcc.setSharedMemoryConversion(true);
  }

  contract->setOperand(0, layoutedLhs.getResult());
  contract->setOperand(1, layoutedRhs.getResult());
  contract->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(contract);
  auto toLayout = rewriter.create<ToLayoutOp>(loc, contract->getResult(0),
                                              cLayout, schedule.getIntrinsic());
  rewriter.replaceAllUsesExcept(contract->getResult(0), toLayout.getResult(),
                                toLayout);

  return success();
}

static LogicalResult setConvolutionAnchor(IREE::GPU::MMAScheduleAttr schedule,
                                          SmallVector<bool> promotedOperands,
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
  auto layoutedLhs =
      rewriter.create<ToLayoutOp>(loc, lhs, aLayout, schedule.getIntrinsic());
  auto layoutedRhs =
      rewriter.create<ToLayoutOp>(loc, rhs, bLayout, schedule.getIntrinsic());
  auto layoutedAcc =
      rewriter.create<ToLayoutOp>(loc, acc, cLayout, schedule.getIntrinsic());

  // Promote matmul lhs and rhs.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  if (promotedOperands[0]) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[1]) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[2]) {
    layoutedAcc.setSharedMemoryConversion(true);
  }

  conv->setOperand(0, layoutedLhs.getResult());
  conv->setOperand(1, layoutedRhs.getResult());
  conv->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(conv);
  auto toLayout = rewriter.create<ToLayoutOp>(loc, conv->getResult(0), cLayout,
                                              schedule.getIntrinsic());
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

static LogicalResult setAttentionMatmulAnchor(RewriterBase &rewriter,
                                              linalg::LinalgOp qkMatmul,
                                              linalg::LinalgOp pvMatmul) {

  IREE::GPU::MMAScheduleAttr qkSchedule =
      rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(getIntrinsic(qkMatmul),
                                                   getSubgroupMCount(qkMatmul),
                                                   getSubgroupNCount(qkMatmul));

  IREE::GPU::MMAScheduleAttr pvSchedule =
      rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(getIntrinsic(pvMatmul),
                                                   getSubgroupMCount(pvMatmul),
                                                   getSubgroupNCount(pvMatmul));

  // Check if the intrinsic output for qkMatmul can be reused for pvMatmul.
  // We know that pvMatmul takes result of qkMatmul as it's lhs.
  // If the intrinsic output of pvMatmul can be used as rhs of pvMatmul,
  // we swap operands of both contracts to get output as transposed intrinsic.
  bool reuseIntrinsicOutput = false;
  bool transposeIntrinsic = false;

  auto qkIntrinsic = cast<IREE::GPU::MMAAttr>(qkSchedule.getIntrinsic());
  auto pvIntrinsic = cast<IREE::GPU::MMAAttr>(pvSchedule.getIntrinsic());
  IREE::GPU::MMASingleSubgroupLayout lhsLayout =
      pvIntrinsic.getASingleSubgroupLayout();
  IREE::GPU::MMASingleSubgroupLayout rhsLayout =
      pvIntrinsic.getBSingleSubgroupLayout();
  IREE::GPU::MMASingleSubgroupLayout outLayout =
      qkIntrinsic.getCSingleSubgroupLayout();

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

  SmallVector<bool> promotedQKOperands = getPromotedOperands(qkMatmul);
  SmallVector<bool> promotedPVOperands = getPromotedOperands(pvMatmul);

  // Do not promote lhs of pvMatmul if we are reusing the intrinsic output.
  promotedPVOperands[0] = !reuseIntrinsicOutput;

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

    // Swap promoted operands.
    std::swap(promotedQKOperands[0], promotedQKOperands[1]);
    std::swap(promotedPVOperands[0], promotedPVOperands[1]);
  }

  if (failed(setContractionAnchor(qkSchedule, promotedQKOperands, rewriter,
                                  qkMatmul))) {
    return failure();
  }

  return setContractionAnchor(pvSchedule, promotedPVOperands, rewriter,
                              pvMatmul);
}

// Apply the permuted projection map to the layout.
static IREE::VectorExt::VectorLayoutInterface
getLayoutForMap(VectorLayoutInterface layout, AffineMap map) {
  // Project out unusued dims in layout.
  SmallVector<bool> projectedDims(layout.getRank(), false);
  for (int dim : getUnusedDimsBitVector(map).set_bits()) {
    projectedDims[dim] = true;
  }
  IREE::VectorExt::VectorLayoutInterface projectedLayout =
      layout.project(projectedDims);

  // Transpose dims in layout.
  AffineMap permMap = compressUnusedDims(map);
  SmallVector<int64_t> identity =
      llvm::to_vector(llvm::seq<int64_t>(permMap.getNumDims()));
  SmallVector<int64_t> perm = applyPermutationMap<int64_t>(permMap, identity);
  return projectedLayout.permute(perm);
}

static LogicalResult setDerivedThreadConfigLayout(
    IREE::GPU::DerivedThreadConfigAttr config, linalg::LinalgOp linalgOp,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  int64_t opRank = linalgOp.getNumLoops();

  SmallVector<int64_t> elementTile = config.getStaticTilingLevelSizes(
      static_cast<unsigned>(IREE::GPU::TilingLevel::Thread), linalgOp);

  SmallVector<int64_t> opShape = linalgOp.getStaticLoopRanges();
  for (auto [index, size, element] : llvm::enumerate(opShape, elementTile)) {
    if (ShapedType::isDynamic(size)) {
      linalgOp->emitError() << "Cannot set layouts for dynamic loop ranges";
      return failure();
    }

    if (size % element != 0) {
      linalgOp->emitError()
          << "Operation with unsupported number of elements. "
             "Chosen vector tile sizes for operation are not "
             "divisible by operation loop ranges at dim: "
          << index << ", size=" << size << ", vector size = " << element;
      return failure();
    }

    size /= element;
  }

  SmallVector<int64_t> threadTile(opRank, 1);
  SmallVector<int64_t> threadStrides(opRank, 0);

  int64_t residualThreads = ShapedType::getNumElements(workgroupSize);
  int64_t currStride = 1;

  for (auto [tile, stride, size] :
       llvm::reverse(llvm::zip(threadTile, threadStrides, opShape))) {
    int64_t threadBlock;
    if (residualThreads % size == 0) {
      threadBlock = size;
    } else if (size % residualThreads == 0) {
      threadBlock = residualThreads;
    } else {
      linalgOp->emitError() << "Operation with unsupported number of elements.";
      return failure();
    }

    tile = threadBlock;
    stride = currStride;
    size /= threadBlock;

    currStride *= threadBlock;
    residualThreads /= threadBlock;
  }

  SmallVector<int64_t> subgroupTile(opRank, 1);
  SmallVector<int64_t> subgroupStrides(opRank, 0);
  SmallVector<int64_t> outerTile(opRank, 1);

  MLIRContext *context = rewriter.getContext();
  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupTile, opShape, outerTile, threadTile, elementTile,
      subgroupStrides, threadStrides);

  Location loc = linalgOp.getLoc();

  rewriter.setInsertionPointAfter(linalgOp);
  for (OpResult result : linalgOp->getResults()) {
    VectorLayoutInterface resultLayout =
        getLayoutForMap(layout, linalgOp.getIndexingMapMatchingResult(result));
    auto toLayout = rewriter.create<ToLayoutOp>(loc, result, resultLayout);
    rewriter.replaceAllUsesExcept(result, toLayout, toLayout);
  }

  return success();
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
    FunctionOpInterface func = getOperation();
    IRRewriter rewriter(func);

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(func);
    if (!maybeWorkgroupSize) {
      func->emitOpError()
          << "unable to query workgroup_size information from entry point";
      return signalPassFailure();
    }

    if (failed(setDerivedConfigLayouts(func, maybeWorkgroupSize.value(),
                                       rewriter))) {
      return signalPassFailure();
    }

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

    for (linalg::LinalgOp contract : contracts) {
      SmallVector<bool> promotedOperands = getPromotedOperands(contract);
      auto contractionSchedule = rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
          getIntrinsic(contract), getSubgroupMCount(contract),
          getSubgroupNCount(contract));
      if (failed(setContractionAnchor(contractionSchedule, promotedOperands,
                                      rewriter, contract))) {
        return signalPassFailure();
      }
    }

    for (linalg::LinalgOp conv : convs) {
      SmallVector<bool> promotedOperands = getPromotedOperands(conv);
      auto convSchedule = rewriter.getAttr<IREE::GPU::MMAScheduleAttr>(
          getIntrinsic(conv), getSubgroupMCount(conv), getSubgroupNCount(conv));
      if (failed(setConvolutionAnchor(convSchedule, promotedOperands, rewriter,
                                      conv))) {
        return signalPassFailure();
      }
    }

    if (attentionQKMatmul && attentionPVMatmul) {
      if (failed(setAttentionMatmulAnchor(rewriter, attentionQKMatmul,
                                          attentionPVMatmul))) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult setDerivedConfigLayouts(FunctionOpInterface funcOp,
                                        ArrayRef<int64_t> workgroupSize,
                                        RewriterBase &rewriter) {
    SmallVector<linalg::LinalgOp> candidates;
    funcOp->walk([&](linalg::LinalgOp op) {
      auto config = dyn_cast_or_null<IREE::GPU::DerivedThreadConfigAttr>(
          getLoweringConfig(op));
      if (config) {
        candidates.push_back(op);
      }
    });

    for (linalg::LinalgOp candidate : candidates) {
      auto config = dyn_cast_or_null<IREE::GPU::DerivedThreadConfigAttr>(
          getLoweringConfig(candidate));
      assert(config);
      if (failed(setDerivedThreadConfigLayout(config, candidate, workgroupSize,
                                              rewriter))) {
        return failure();
      }
    }

    return success();
  }
};
} // namespace

} // namespace mlir::iree_compiler
