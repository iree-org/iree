// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-padding-level"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYPADDINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyPaddingLevelPass final
    : impl::GPUApplyPaddingLevelPassBase<GPUApplyPaddingLevelPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

/// For a reduction dimension of a linalg operation `linalgOp`,
///
/// 1) `loopIndex` is the index of the loop in `linalgOp` that is reduced,
/// 2) `operand` is the first operand indexed by the reduction dimension,
/// 3) `operandDim` is the dimension of the operand that's reduced.
///
/// Example. Consider the generic below with 2 reduction dimensions.
///
/// ```mlir
/// #map = affine_map<(d0, d1) -> (d1, d0)>
/// #map1 = affine_map<(d0, d1) -> (d0, d1)>
/// #map2 = affine_map<(d0, d1) -> ()>
/// [...]
/// %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
///                      iterator_types = ["reduction", "reduction"]}
///                      ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>)
///                      outs(%arg2 : tensor<f16>)
/// [...]
/// ```
///
/// d0 has reduction info (loopIndex = 0, operand = %arg0, operandDim = 1),
/// d1 has reduction info (loopIndex = 1, operand = %arg0, operandDim = 0).
namespace {
struct ReductionDimInfo {
public:
  unsigned loopIndex;
  unsigned operandDim;
  Value operand;
};
} // namespace

static FailureOr<SmallVector<ReductionDimInfo>>
getReductionInfo(linalg::LinalgOp linalgOp) {
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  unsigned numReductionDims =
      llvm::count(iteratorTypes, utils::IteratorType::reduction);
  SmallVector<ReductionDimInfo> reductionDimInfos;
  reductionDimInfos.reserve(numReductionDims);

  for (unsigned loopIdx = 0, endIdx = iteratorTypes.size(); loopIdx < endIdx;
       ++loopIdx) {
    if (!linalg::isReductionIterator(iteratorTypes[loopIdx]))
      continue;

    AffineExpr loopIdxExpr = getAffineDimExpr(loopIdx, linalgOp.getContext());
    for (auto [operandIndex, indexingMap] : llvm::enumerate(indexingMaps)) {
      if (std::optional<unsigned> index =
              indexingMap.getResultPosition(loopIdxExpr)) {
        Value operand = linalgOp->getOperand(operandIndex);
        reductionDimInfos.push_back({loopIdx, index.value(), operand});
        break;
      }
    }
  }

  if (reductionDimInfos.size() != numReductionDims)
    return failure();

  return reductionDimInfos;
}

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {
  // 1.a. Get padding values. The default should be poison, instead of 0.
  //
  // TODO(newling) pad with poison. Requires
  //
  // https://github.com/iree-org/iree/pull/21573
  // https://github.com/llvm/llvm-project/pull/152003
  // https://github.com/iree-org/iree/issues/21575
  //
  // The linalg operations can be padded with any value because we rewrite the
  // basic block to select the reduction identity for the yielded value if the
  // index corresponds to the padded part of the tensor.
  //
  // Non-linalg operations require special handling.
  // TODO: Extract the special handling into an upstream PaddingOpInterface.

  SmallVector<Attribute> paddingValues;
  for (Value operand : tilingInterfaceOp.getOperation()->getOperands()) {
    paddingValues.push_back(
        rewriter.getZeroAttr(getElementTypeOrSelf(operand.getType())));
  }

  // 1.b. Special adjustment for OnlineAttention mask padding.
  if (auto onlineAttentionOp = dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(
          tilingInterfaceOp.getOperation())) {
    TypedValue<ShapedType> mask = onlineAttentionOp.getMask();
    if (!mask) {
      tilingInterfaceOp.emitRemark(
          "failed to pad op: requires a mask operand to pad to the "
          "proper value. Consider materializing the mask operand "
          "explicitly.");
      return failure();
    }
    Type maskEltType = getElementTypeOrSelf(mask.getType());
    if (!llvm::isa<FloatType>(maskEltType)) {
      tilingInterfaceOp.emitRemark(
          "failed to pad op: -inf requires a float type");
      return failure();
    }
    int64_t idx = onlineAttentionOp.getMaskMutable()
                      .getAsOperandRange()
                      .getBeginOperandIndex();
    const auto &fltSemantics = cast<FloatType>(maskEltType).getFloatSemantics();
    paddingValues[idx] = rewriter.getFloatAttr(
        maskEltType, APFloat::getInf(fltSemantics, /*Negative=*/true));
  }

  // 2. Get padding sizes from tileSizes.
  SmallVector<int64_t> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getStaticTilingLevelSizes(llvm::to_underlying(tilingLevel),
                                     tilingInterfaceOp);
  SmallVector<OpFoldResult> padSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);

  // 3. Set options.
  linalg::PadTilingInterfaceOptions options =
      linalg::PadTilingInterfaceOptions()
          .setPaddingSizes(padSizes)
          .setPaddingValues(paddingValues)
          .setPadToMultipleOf(true);

  LLVM_DEBUG(DBGS() << "Start padding " << *tilingInterfaceOp << "\n";
             DBGS() << "--with tile sizes: "
                    << llvm::interleaved_array(options.paddingSizes) << "\n";
             DBGS() << "--with padding values: "
                    << llvm::interleaved_array(options.paddingValues) << "\n";
             DBGS() << "--with padToMultipleOf: " << options.padToMultipleOf
                    << "\n");

  // For linalg ops, we will rewrite the basic block in a way that means padded
  // parts of tensors are never read. This is useful to avoid inferring what
  // padding values should be for non-trivial basic blocks.
  FailureOr<SmallVector<ReductionDimInfo>> reductionDimInfo;
  if (auto linalgOp =
          dyn_cast<linalg::LinalgOp>(tilingInterfaceOp.getOperation())) {
    reductionDimInfo = getReductionInfo(linalgOp);
    if (failed(reductionDimInfo)) {
      tilingInterfaceOp.emitWarning("failed to map reduction dimensions");
      return failure();
    }
  }

  // 4. Pad.
  SmallVector<tensor::PadOp> padOps;
  FailureOr<TilingInterface> maybePaddedOp =
      linalg::rewriteAsPaddedOp(rewriter, tilingInterfaceOp, options, padOps);
  if (failed(maybePaddedOp)) {
    tilingInterfaceOp.emitWarning("failed to pad op");
    return failure();
  }

  TilingInterface paddedOp = *maybePaddedOp;

  if (auto paddedLinalgOp =
          dyn_cast<linalg::LinalgOp>(paddedOp.getOperation())) {
    Block *block = paddedLinalgOp.getBlock();

    SmallVector<Operation *> reductions;
    for (auto [index, initOpOperand] :
         llvm::enumerate(paddedLinalgOp.getDpsInitsMutable())) {
      SmallVector<Operation *> combinerOps;
      matchReduction(paddedLinalgOp.getRegionOutputArgs(), index, combinerOps);
      reductions.insert(reductions.begin(), combinerOps.begin(),
                        combinerOps.end());
    }

    for (Operation *reduction : reductions) {
      std::optional<TypedAttr> reductionIdentity =
          arith::getNeutralElement(reduction);
      if (!reductionIdentity.has_value()) {
        paddedOp.emitWarning("failed to get neutral element for reduction");
        return failure();
      }

      // Get the sizes of the reduction dimensions before padding:
      rewriter.setInsertionPoint(paddedOp.getOperation());
      SmallVector<std::pair<unsigned, Value>> reductionDimSizes;
      assert(succeeded(reductionDimInfo) &&
             "obtained with confirmation earlier");
      for (auto &&dimInfo : reductionDimInfo.value()) {
        Value redDimSize = rewriter.create<tensor::DimOp>(
            paddedOp.getLoc(), dimInfo.operand, dimInfo.operandDim);
        reductionDimSizes.push_back({dimInfo.loopIndex, redDimSize});
      }

      // Add a check within the block to see if the current iteration over the
      // loops is inside or outside the padded part of the iteration space.
      rewriter.setInsertionPoint(reduction);
      SmallVector<Value> conds;
      for (auto &&[redDim, redDimSize] : reductionDimSizes) {
        Value redDimIndex =
            linalg::IndexOp::create(rewriter, paddedOp.getLoc(), redDim);
        Value cond = arith::CmpIOp::create(rewriter, paddedOp.getLoc(),
                                           arith::CmpIPredicate::ult,
                                           redDimIndex, redDimSize);
        conds.push_back(cond);
      }
      Value reductionIdentityValue = rewriter.create<arith::ConstantOp>(
          paddedOp.getLoc(), reductionIdentity.value());
      assert(conds.size() > 0);
      Value cond = conds[0];
      for (Value nxtCond : llvm::drop_begin(conds, 1)) {
        cond = rewriter.create<arith::AndIOp>(paddedOp.getLoc(), cond, nxtCond);
      }

      // Find the reduction op operand that is reduced with the carried output.
      if (reduction->getNumOperands() != 2) {
        paddedOp.emitWarning("expected a reduction operation with 2 operands");
        return failure();
      }
      Value carry = block->getArguments().back();
      unsigned uncarryIndex = reduction->getOperand(0) == carry ? 1 : 0;
      Value uncarried = reduction->getOperand(uncarryIndex);

      // Select the reduction identity value if in the padding region.
      Value selected = arith::SelectOp::create(
          rewriter, paddedOp.getLoc(), cond, uncarried, reductionIdentityValue);
      IRMapping mapping;
      mapping.map(reduction->getOperand(uncarryIndex), selected);
      Operation *redClone = rewriter.clone(*reduction, mapping);
      rewriter.replaceOp(reduction, redClone);
    }
  }

  // 5. For each PadOp, create a linalg::CopyOp to allow dim propagations.
  for (tensor::PadOp padOp : padOps) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(padOp);

    // Record users for RAUW before creating new users.
    llvm::SmallDenseSet<Operation *> users(padOp.getResult().getUsers().begin(),
                                           padOp.getResult().getUsers().end());

    RankedTensorType tensorTy = padOp.getResultType();
    int64_t rank = tensorTy.getRank();
    SmallVector<OpFoldResult> sizes(rank, OpFoldResult());
    for (int64_t i = 0; i < rank; ++i) {
      sizes[i] = rewriter.createOrFold<tensor::DimOp>(paddedOp->getLoc(),
                                                      padOp.getResult(), i);
      if (auto v = dyn_cast<Value>(sizes[i]))
        sizes[i] = getAsOpFoldResult(v);
    }

    Value out = rewriter.create<tensor::EmptyOp>(
        paddedOp.getLoc(), sizes, getElementTypeOrSelf(tensorTy));
    auto copied = rewriter.create<linalg::CopyOp>(paddedOp.getLoc(),
                                                  padOp.getResult(), out);
    rewriter.replaceUsesWithIf(padOp.getResult(), copied.getResult(0),
                               [&](OpOperand &opOperand) {
                                 return users.contains(opOperand.getOwner());
                               });
  }

  return success();
}

void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  for (TilingInterface op : targetOps) {
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op, tilingLevel);
  }
}

} // namespace mlir::iree_compiler
