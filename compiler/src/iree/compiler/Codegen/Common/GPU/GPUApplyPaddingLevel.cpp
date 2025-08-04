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

/// For each reduction dimension in the linalg operation `op`, get a tuple
/// containing:
///
/// 1) loop index of linalg op that's reduced,
/// 2) an operand that is indexed by the reduction dimension,
/// 3) dimension of operand that's reduced.
///
/// Example:
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
/// ```
///
/// The above operation has 2 reduction dimensions (d0 and d1).
///
/// d0 corresponds to operand dimension 1 of %arg0, and
/// d1 corresponds to operand dimension 0 of %arg0.
///
/// So [{0, %arg0, 1}, {1, %arg0, 0}] is returned.
static SmallVector<std::tuple<unsigned, Value, unsigned>>
findReductionDims(linalg::LinalgOp op) {

  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();

  SmallVector<std::tuple<unsigned, Value, unsigned>> triplets;
  triplets.reserve(llvm::count(iteratorTypes, utils::IteratorType::reduction));

  for (unsigned loopIdx = 0; loopIdx < iteratorTypes.size(); ++loopIdx) {
    if (!linalg::isReductionIterator(iteratorTypes[loopIdx]))
      continue;

    AffineExpr loopIdxExpr = getAffineDimExpr(loopIdx, op.getContext());
    bool found = false;
    for (auto operandIter : llvm::enumerate(indexingMaps)) {
      if (std::optional<unsigned> index =
              operandIter.value().getResultPosition(loopIdxExpr)) {
        Value operand = op->getOperand(operandIter.index());
        triplets.push_back({loopIdx, operand, index.value()});
        found = true;
        break;
      }
    }

    assert(found && "failed to find operand with reduction dimension");
  }

  return triplets;
}

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {

  // 1.a. Get padding values.
  // TODO(newling)
  // 1) pad with poison.
  // 2) remove special logic for online attention.
  // This can be done when
  //
  // https://github.com/iree-org/iree/pull/21573 and
  // https://github.com/llvm/llvm-project/pull/152003 and
  // https://github.com/iree-org/iree/issues/21575
  //
  // are landed/resolved.
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
  auto options = linalg::PadTilingInterfaceOptions()
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

  SmallVector<std::tuple<unsigned, Value, unsigned>> reductionDims;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingInterfaceOp.getOperation());
  if (linalgOp) {
    reductionDims = findReductionDims(linalgOp);
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

  if (linalgOp) {

    // Just before the new linalg operation, insert operations to get the
    // reduction dimension sizes.
    SmallVector<std::pair<unsigned, Value>> redDimSizes;
    rewriter.setInsertionPoint(paddedOp.getOperation());
    for (auto &&abc : reductionDims) {
      auto [redDim, operand, redDimPos] = abc;
      Value redDimSize =
          rewriter.create<tensor::DimOp>(paddedOp.getLoc(), operand, redDimPos);
      redDimSizes.push_back({redDim, redDimSize});
    }

    auto *block = cast<linalg::LinalgOp>(paddedOp.getOperation()).getBlock();
    Operation *yield = block->getTerminator();
    if (yield->getNumOperands() != 1) {
      paddedOp.emitWarning("expected a single yield operand");
      return failure();
    }
    Value yielded = yield->getOperand(0);
    Operation *reduction = yielded.getDefiningOp();
    if (!reduction) {
      paddedOp.emitWarning("expected a reduction operation before yield");
      return failure();
    }
    auto zero = arith::getNeutralElement(reduction);
    if (!zero.has_value()) {
      paddedOp.emitWarning("failed to get neutral element for reduction");
      return failure();
    }

    rewriter.setInsertionPoint(reduction);
    SmallVector<Value> conds;
    for (auto &&[redDim, redDimSize] : redDimSizes) {
      Value redDimIndex =
          linalg::IndexOp::create(rewriter, paddedOp.getLoc(), redDim);
      Value cond = arith::CmpIOp::create(rewriter, paddedOp.getLoc(),
                                         arith::CmpIPredicate::ult, redDimIndex,
                                         redDimSize);
      conds.push_back(cond);
    }

    Value zeroValue =
        rewriter.create<arith::ConstantOp>(paddedOp.getLoc(), zero.value());

    // Merge the conds with andi operations.
    assert(conds.size() > 0);
    Value cond = conds[0];
    for (unsigned i = 1; i < conds.size(); ++i) {
      cond = arith::AndIOp::create(rewriter, paddedOp.getLoc(), cond, conds[i]);
    }

    // Find the reduction op operand that is the final block argument
    if (reduction->getNumOperands() != 2) {
      paddedOp.emitWarning("expected a reduction operation with 2 operands");
      return failure();
    }
    Value carry = block->getArguments().back();
    unsigned uncarryIndex = reduction->getOperand(0) == carry ? 1 : 0;
    Value uncarried = reduction->getOperand(uncarryIndex);
    Value selected = arith::SelectOp::create(rewriter, paddedOp.getLoc(), cond,
                                             uncarried, zeroValue);

    IRMapping mapping;
    mapping.map(reduction->getOperand(uncarryIndex), selected);
    auto redClone = rewriter.clone(*reduction, mapping);
    rewriter.replaceOp(reduction, redClone);
  }

  // mapping.map(reduction->getOperand(1 - uncarryIndex), c

  // update reduction's first operand to be selected.
  // reduction->setOperand(uncarryIndex, selected);

  // 5. For each PadOp, create a linalg::CopyOp to allow dim propagations.
  for (auto padOp : padOps) {
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
