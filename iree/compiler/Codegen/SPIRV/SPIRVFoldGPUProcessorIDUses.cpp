// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVFoldGPUProcessorIDUses.cpp ------------------------------------===//
//
// This file implements patterns and passes for folding GPU processor ID uses.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-fold-gpu-procid-uses"

namespace mlir {
namespace iree_compiler {

namespace {

/// Returns true if the given `expr` is a linear expression over one
/// symbol/dimension.
///
/// Note that this function is not meant to check for all linear expression
/// cases. It only checks that:
/// 1) No mod/div operations,
/// 2) For mul operations, one of the operand is a constant.
/// Also this function assumes `expr` only contains one symbol/dimension.
bool isLinearExpr(AffineExpr expr) {
  switch (expr.getKind()) {
    case mlir::AffineExprKind::Add: {
      auto binExpr = expr.cast<AffineBinaryOpExpr>();
      return isLinearExpr(binExpr.getLHS()) && isLinearExpr(binExpr.getRHS());
    }
    case mlir::AffineExprKind::Mul: {
      auto binExpr = expr.cast<AffineBinaryOpExpr>();
      AffineExpr lhs = binExpr.getLHS();
      AffineExpr rhs = binExpr.getRHS();
      return (lhs.isa<AffineConstantExpr>() && isLinearExpr(rhs)) ||
             (rhs.isa<AffineConstantExpr>() && isLinearExpr(lhs));
    };
    case mlir::AffineExprKind::Mod:
    case mlir::AffineExprKind::FloorDiv:
    case mlir::AffineExprKind::CeilDiv:
      return false;
    case mlir::AffineExprKind::Constant:
    case mlir::AffineExprKind::DimId:
    case mlir::AffineExprKind::SymbolId:
      return true;
    default:
      llvm_unreachable("unhandled affine expr kind");
  }
}

/// Replaces the given `dim` in `expr` with a constant `value`.
AffineExpr replaceSymbolWithValue(AffineExpr expr, AffineSymbolExpr dim,
                                  int64_t value) {
  auto cstExpr = getAffineConstantExpr(value, expr.getContext());
  return expr.replace(dim, cstExpr);
}

/// Converts a dimension string to its corresponding index.
int dimensionToIndex(StringRef dimension) {
  return StringSwitch<int>(dimension).Case("x", 0).Case("y", 1).Case("z", 2);
}

IREE::HAL::ReturnOp getEntryPointReturnOp(Operation *op) {
  auto funcOp = op->getParentOfType<FuncOp>();
  auto variantOp =
      funcOp.getOperation()->getParentOfType<IREE::HAL::ExecutableVariantOp>();

  IREE::HAL::ExecutableEntryPointOp entryPointOp;
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    if (op.sym_name() == funcOp.getName()) {
      entryPointOp = op;
      break;
    }
  }
  if (!entryPointOp || !entryPointOp.getBody()) return {};

  Operation *terminator = entryPointOp.getBlock()->getTerminator();
  auto retOp = dyn_cast<IREE::HAL::ReturnOp>(terminator);
  if (!retOp || retOp.getNumOperands() != 3) return {};

  LLVM_DEBUG(llvm::dbgs() << "workgroup count function return op: " << retOp
                          << "\n");
  return retOp;
}

/// Gets the block processor ID's upper bound. This queries the workgroup count
/// function.
Optional<int64_t> getProcessorIDUpperBound(gpu::BlockIdOp blockIDOp) {
  auto retOp = getEntryPointReturnOp(blockIDOp);
  if (!retOp) return llvm::None;

  int index = dimensionToIndex(blockIDOp.dimension());
  IntegerAttr attr;
  if (!matchPattern(retOp.getOperand(index), m_Constant(&attr)))
    return llvm::None;

  return attr.getInt();
}

Optional<int64_t> getProcessorIDUpperBound(
    IREE::HAL::InterfaceWorkgroupIDOp blockIDOp) {
  auto retOp = getEntryPointReturnOp(blockIDOp);
  if (!retOp) return llvm::None;

  int index = blockIDOp.dimensionAttr().getInt();
  IntegerAttr attr;
  if (!matchPattern(retOp.getOperand(index), m_Constant(&attr)))
    return llvm::None;

  return attr.getInt();
}

/// Gets the thread processor ID's upper bound. This queries the SPIR-V entry
/// point ABI.
Optional<int64_t> getProcessorIDUpperBound(gpu::ThreadIdOp threadIDOp) {
  FuncOp funcOp = threadIDOp->getParentOfType<FuncOp>();
  auto abiAttr = funcOp->getAttrOfType<spirv::EntryPointABIAttr>(
      spirv::getEntryPointABIAttrName());
  if (!abiAttr) return llvm::None;

  int index = dimensionToIndex(threadIDOp.dimension());
  auto valueIt = abiAttr.local_size().getIntValues().begin() + index;
  return (*valueIt).getZExtValue();
}

/// Folds `affine.min` ops which has only one symbol operand, which is a
/// processor ID. For such cases we can use the processor ID's upper bound to
/// simplify the `affine.min`.
///
/// For example, this pattern can simplify the following IR:
/// ```
/// %id = "gpu.thread_id"() {dimension = "x"} : () -> index
/// ...
/// affine.min affine_map<()[s0] -> (3, s0 * -2 + 225)>()[%id]
/// ```
/// if the upper bound for thread ID along the x dimension is 112.
struct FoldAffineMinOverProcessorID : OpRewritePattern<AffineMinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "inspecting " << minOp << "\n");
    MLIRContext *context = minOp.getContext();
    auto dimensions = minOp.getDimOperands();
    auto symbols = minOp.getSymbolOperands();

    // We expect the affine.min op to only have one symbol operand.
    if (!llvm::hasSingleElement(symbols) || !dimensions.empty()) {
      return rewriter.notifyMatchFailure(
          minOp, "expected to only have one symbol operand");
    }

    // And the symbol operand should come from a GPU processor ID.
    Operation *symbolOp = symbols.front().getDefiningOp();
    auto symbol0 = getAffineSymbolExpr(0, context).cast<AffineSymbolExpr>();

    Optional<int64_t> ub;
    if (auto blockIDOp = dyn_cast<gpu::BlockIdOp>(symbolOp)) {
      ub = getProcessorIDUpperBound(blockIDOp);
    } else if (auto blockIDOp =
                   dyn_cast<IREE::HAL::InterfaceWorkgroupIDOp>(symbolOp)) {
      ub = getProcessorIDUpperBound(blockIDOp);
    } else if (auto threadIDOp = dyn_cast<gpu::ThreadIdOp>(symbolOp)) {
      ub = getProcessorIDUpperBound(threadIDOp);
    }
    if (!ub) {
      return rewriter.notifyMatchFailure(
          minOp, "failed to query processor ID upper bound");
    }
    LLVM_DEBUG(llvm::dbgs() << "processor ID '" << *symbolOp
                            << "' upper bound: " << *ub << "\n");

    // Look at each result expression. For expressions that are functions of
    // the input symbol, try to simplify it. We do this by replacing the
    // symbol with its lower and upper bound. This requires the result
    // expression to be a linear function of the input symbol.
    SmallVector<AffineExpr, 4> results;
    // The indices into `results` where the corresponding AffineExpr is a
    // constant from the original map. We need to keep track of this so later we
    // can probe whether the constant is the min.
    SmallVector<unsigned, 4> cstIndices;
    for (auto result : minOp.getAffineMap().getResults()) {
      if (auto cstResult = result.dyn_cast<AffineConstantExpr>()) {
        results.push_back(cstResult);
        cstIndices.push_back(results.size() - 1);
      } else if (isLinearExpr(result)) {
        results.push_back(simplifyAffineExpr(
            replaceSymbolWithValue(result, symbol0, 0), 0, 1));
        results.push_back(simplifyAffineExpr(
            replaceSymbolWithValue(result, symbol0, *ub - 1), 0, 1));
        LLVM_DEBUG({
          auto map = AffineMap::get(0, 1, results, context);
          llvm::dbgs() << "map after substituting with processor ID bounds: "
                       << map << "\n";
        });
      } else {
        // We cannot handle such cases. Just bail out on matching the pattern.
        return rewriter.notifyMatchFailure(
            minOp, "expected to have a linear function of the symbol");
      }
    }

    // Returns true if the given affine expression is a non-negative constant.
    auto isNonNegativeCstExpr = [](AffineExpr e) {
      if (auto cst = e.dyn_cast<AffineConstantExpr>())
        return cst.getValue() >= 0;
      return false;
    };

    // Check whether any of the original constant expressions, when subtracted
    // from all other expressions, produces only >= 0 constants. If so, it is
    // the min.
    for (auto cstIndex : cstIndices) {
      auto candidate = results[cstIndex].cast<AffineConstantExpr>();

      SmallVector<AffineExpr, 4> subExprs;
      subExprs.reserve(results.size());
      for (auto r : results) subExprs.push_back(r - candidate);

      AffineMap subMap =
          simplifyAffineMap(AffineMap::get(0, 1, subExprs, context));
      LLVM_DEBUG(llvm::dbgs() << "map by subtracting expr '" << candidate
                              << "': " << subMap << "\n");
      if (llvm::all_of(subMap.getResults(), isNonNegativeCstExpr)) {
        rewriter.replaceOpWithNewOp<ConstantIndexOp>(minOp,
                                                     candidate.getValue());
        return success();
      }
    }

    return failure();
  }
};

/// Tests processor ID use folding patterns.
struct SPIRVFoldProcessorIDUsesPass
    : public SPIRVFoldProcessorIDUsesBase<SPIRVFoldProcessorIDUsesPass> {
  SPIRVFoldProcessorIDUsesPass() = default;
  SPIRVFoldProcessorIDUsesPass(const SPIRVFoldProcessorIDUsesPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    populateFoldGPUProcessorIDUsesPatterns(context, patterns);
    (void)applyPatternsAndFoldGreedily(getOperation().getInnerModule(),
                                       std::move(patterns));
  }
};

};  // namespace

void populateFoldGPUProcessorIDUsesPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<FoldAffineMinOverProcessorID>(context);
  AffineMinOp::getCanonicalizationPatterns(patterns, context);
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVFoldProcessorIDUsesPass() {
  return std::make_unique<SPIRVFoldProcessorIDUsesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
