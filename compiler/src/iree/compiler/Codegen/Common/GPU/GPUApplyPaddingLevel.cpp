// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Interfaces/TensorMaskingOpInterface.h"
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

static bool hasLoweringConfig(Operation *op) {
  if (!isa<TilingInterface>(op)) {
    return false;
  }
  return getLoweringConfig(op) != nullptr;
}

/// tensor.pad is not DPS, which causes issues with dimension reification. By
/// adding a no-op linalg.copy on their result, we make them DPS.
static void makePadDPS(RewriterBase &rewriter, tensor::PadOp padOp) {
  Location loc = padOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(padOp);

  // Record users for RAUW before creating new users.
  llvm::SmallDenseSet<Operation *> users(llvm::from_range,
                                         padOp.getResult().getUsers());
  RankedTensorType tensorTy = padOp.getResultType();
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(rewriter, loc, padOp.getResult());
  Value out = tensor::EmptyOp::create(rewriter, loc, sizes,
                                      getElementTypeOrSelf(tensorTy));
  auto copied = linalg::CopyOp::create(rewriter, loc, padOp.getResult(), out);
  rewriter.replaceUsesWithIf(padOp.getResult(), copied.getResult(0),
                             [&](OpOperand &opOperand) {
                               return users.contains(opOperand.getOwner());
                             });
}

struct MaskListener final : public RewriterBase::Listener {
  void notifyOperationInserted(Operation *op,
                               RewriterBase::InsertPoint previous) override {
    if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      pads.push_back(padOp);
    }
  }

  void notifyOperationErased(Operation *op) override {
    if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      llvm::erase(pads, padOp);
    }
  }

  SmallVector<tensor::PadOp> pads;
};

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {
  auto tensorMaskingOp =
      dyn_cast<TensorMaskingOpInterface>(tilingInterfaceOp.getOperation());
  if (!tensorMaskingOp) {
    return failure();
  }

  SmallVector<int64_t> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getStaticTilingLevelSizes(llvm::to_underlying(tilingLevel),
                                     tilingInterfaceOp);
  SmallVector<OpFoldResult> padSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(tilingInterfaceOp);
  FailureOr<SmallVector<Value>> result =
      tensorMaskingOp.getMaskedImplementation(rewriter, padSizes);
  if (failed(result)) {
    return failure();
  }

  rewriter.replaceOp(tilingInterfaceOp.getOperation(), result.value());

  return success();
}

void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  MaskListener maskListener;
  IRRewriter rewriter(funcOp, &maskListener);
  for (TilingInterface op : targetOps) {
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op, tilingLevel);
    // Propagate padding up by padding producers if possible.
    while (!maskListener.pads.empty()) {
      tensor::PadOp padOp = maskListener.pads.pop_back_val();

      auto resultVal = dyn_cast<OpResult>(padOp.getSource());
      if (!resultVal) {
        makePadDPS(rewriter, padOp);
        continue;
      }
      Operation *producer = resultVal.getOwner();
      if (hasLoweringConfig(producer)) {
        // Skip operations that have their own lowering config to avoid
        // incompatible padding.
        makePadDPS(rewriter, padOp);
        continue;
      }

      auto maskingIface = dyn_cast<TensorMaskingOpInterface>(producer);
      if (!maskingIface) {
        makePadDPS(rewriter, padOp);
        continue;
      }
      rewriter.setInsertionPointAfter(padOp);
      SmallVector<OpFoldResult> padSizes =
          tensor::getMixedSizes(rewriter, padOp.getLoc(), padOp.getResult());
      FailureOr<Value> paddedResult = maskingIface.maskAsProducer(
          rewriter, resultVal.getResultNumber(), padSizes);
      if (failed(paddedResult)) {
        makePadDPS(rewriter, padOp);
        continue;
      }
      DominanceInfo domInfo(padOp);
      rewriter.replaceUsesWithIf(
          padOp.getResult(), paddedResult.value(), [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return domInfo.properlyDominates(paddedResult.value(), user);
          });
    }
  }
}

} // namespace mlir::iree_compiler
