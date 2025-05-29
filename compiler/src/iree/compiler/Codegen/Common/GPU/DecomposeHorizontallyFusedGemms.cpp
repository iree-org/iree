// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSEHORIZONTALLYFUSEDGEMMSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct DecomposeHorizontallyFusedGemmsPass final
    : impl::DecomposeHorizontallyFusedGemmsPassBase<
          DecomposeHorizontallyFusedGemmsPass> {
  void runOnOperation() override;
};
} // namespace

//===---------------------------------------------------------------------===//
// Decompose horizontally fused gemm operations
// TODO: Eventually drop this if we end up creating an operation for the
// horizontally fused contractions.
//===---------------------------------------------------------------------===//

static LogicalResult captureUsedOperationsAndBlockArguements(
    linalg::LinalgOp linalgOp, SetVector<int64_t> &usedInputs,
    SetVector<Operation *> &usedOperations, int64_t resultNumber) {
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&](Operation *op) -> bool {
    return op->getBlock() == linalgOp.getBlock();
  };

  auto yieldOp = cast<linalg::YieldOp>(linalgOp.getBlock()->getTerminator());
  Value result = yieldOp.getOperand(resultNumber);

  [[maybe_unused]] LogicalResult ret =
      getBackwardSlice(result, &usedOperations, options);
  assert(ret.succeeded());

  // Get all block arguments used by the operations. If any of the arguments
  // used is a dpsInit argument other than resultNumber, return failure.
  for (Operation *op : usedOperations) {
    for (Value operand : op->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() != linalgOp.getBlock()) {
          continue;
        }

        int64_t argNumber = blockArg.getArgNumber();
        if (argNumber >= linalgOp.getNumDpsInputs() &&
            argNumber - linalgOp.getNumDpsInputs() != resultNumber) {
          return failure();
        }

        if (argNumber < linalgOp.getNumDpsInputs()) {
          usedInputs.insert(argNumber);
        }
      }
    }
  }

  return success();
}

// Since the `promotedOperands` changes that needs to be modified
// and transfered over to the decomposed ops.
static IREE::GPU::LoweringConfigAttr
getModifiedLoweringConfigForDecomposedGemmOp(
    RewriterBase &rewriter, IREE::GPU::LoweringConfigAttr origAttr,
    ArrayRef<unsigned> keptOperands) {
  std::optional<SmallVector<int64_t>> promotedOperandsList =
      IREE::GPU::getPromotedOperandList(origAttr);
  if (!promotedOperandsList) {
    return origAttr;
  }

  llvm::SmallDenseSet<int64_t> promotedOperandsSet(
      promotedOperandsList->begin(), promotedOperandsList->end());
  SmallVector<int64_t> newPromotedOperands;
  for (auto [index, origOperandNum] : llvm::enumerate(keptOperands)) {
    if (promotedOperandsSet.contains(origOperandNum)) {
      newPromotedOperands.push_back(index);
    }
  }
  return setPromotedOperandsList(rewriter.getContext(), origAttr,
                                 newPromotedOperands);
}

static LogicalResult
decomposeHorizontallyFusedGemmOperations(RewriterBase &rewriter,
                                         linalg::LinalgOp linalgOp) {
  assert(IREE::LinalgExt::isaHorizontallyFusedContraction(linalgOp) &&
         "expected op that is a horizontally fused contraction");

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(linalgOp);
  // Create num_results linalg.generics, each producing a single result (and
  // relying on canonicalizations to simplify).
  for (int64_t resultNumber : llvm::seq<int64_t>(linalgOp->getNumResults())) {
    rewriter.setInsertionPoint(linalgOp);

    auto yieldOp = cast<linalg::YieldOp>(linalgOp.getBlock()->getTerminator());
    Value result = yieldOp.getOperand(resultNumber);

    // Get all operations required to produce this result.
    SetVector<Operation *> usedOperations;
    SetVector<int64_t> usedInputs;
    if (failed(captureUsedOperationsAndBlockArguements(
            linalgOp, usedInputs, usedOperations, resultNumber))) {
      return failure();
    }

    // Create a new linalg.generic operation for this result.
    SmallVector<OpOperand *> inputs = llvm::map_to_vector(
        usedInputs, [&](int64_t x) { return linalgOp.getDpsInputOperand(x); });
    SmallVector<OpOperand *> inits = {linalgOp.getDpsInitOperand(resultNumber)};

    SmallVector<AffineMap> indexingMaps =
        llvm::map_to_vector(usedInputs, [&](int64_t x) {
          return linalgOp.getIndexingMapsArray()[x];
        });
    indexingMaps.push_back(linalgOp.getIndexingMapMatchingResult(
        linalgOp->getOpResult(resultNumber)));
    llvm::SmallBitVector unusedDims = getUnusedDimsBitVector(indexingMaps);
    indexingMaps = compressUnusedDims(indexingMaps);

    SmallVector<utils::IteratorType> iteratorTypes;
    for (int64_t i : llvm::seq<int64_t>(linalgOp.getNumLoops())) {
      if (!unusedDims.test(i)) {
        iteratorTypes.push_back(linalgOp.getIteratorTypesArray()[i]);
      }
    }

    SmallVector<Value> inputVals = llvm::map_to_vector(
        inputs, [](OpOperand *operand) { return operand->get(); });
    SmallVector<Value> initVals = llvm::map_to_vector(
        inits, [](OpOperand *operand) { return operand->get(); });
    auto newOp = rewriter.create<linalg::GenericOp>(
        linalgOp.getLoc(), TypeRange{inits[0]->get().getType()}, inputVals,
        initVals, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          Block *oldBody = linalgOp.getBlock();
          usedInputs.insert(resultNumber + linalgOp.getNumDpsInputs());

          IRMapping regionMapping;

          for (auto [oldBlockArgNum, newBlockArg] :
               llvm::zip_equal(usedInputs, blockArgs)) {
            regionMapping.map(oldBody->getArgument(oldBlockArgNum),
                              newBlockArg);
          }

          for (Operation *usedOperation : usedOperations) {
            b.clone(*usedOperation, regionMapping);
          }

          b.create<linalg::YieldOp>(loc, regionMapping.lookup(result));
        });

    // If on decomposition any dims are unused propagating lowering config isnt
    // well defined. So propagate lowering config only when no dim is unused.
    if (unusedDims.none()) {
      IREE::GPU::LoweringConfigAttr loweringConfigAttr =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
      if (loweringConfigAttr && getPromotedOperandList(loweringConfigAttr)) {
        SmallVector<unsigned> operandNums =
            llvm::map_to_vector(inputs, [](OpOperand *operand) {
              return operand->getOperandNumber();
            });
        auto range = llvm::map_range(inits, [](OpOperand *operand) {
          return operand->getOperandNumber();
        });
        operandNums.append(range.begin(), range.end());
        IREE::GPU::LoweringConfigAttr newGPUAttr =
            getModifiedLoweringConfigForDecomposedGemmOp(
                rewriter, loweringConfigAttr, operandNums);
        setLoweringConfig(newOp, newGPUAttr);
      }
    }

    rewriter.replaceAllUsesWith(linalgOp->getResult(resultNumber),
                                newOp.getResult(0));
  }

  rewriter.eraseOp(linalgOp);
  return success();
}

void DecomposeHorizontallyFusedGemmsPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  SmallVector<linalg::LinalgOp> horizontallyFusedOps;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (IREE::LinalgExt::isaHorizontallyFusedContraction(linalgOp)) {
      horizontallyFusedOps.push_back(linalgOp);
    }
  });

  for (auto linalgOp : llvm::make_early_inc_range(horizontallyFusedOps)) {
    if (failed(decomposeHorizontallyFusedGemmOperations(rewriter, linalgOp))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
