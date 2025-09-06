// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile-to-vector-size"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUTILETOVECTORSIZEPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

struct LLVMCPUTileToVectorSizePass final
    : impl::LLVMCPUTileToVectorSizePassBase<LLVMCPUTileToVectorSizePass> {
  using impl::LLVMCPUTileToVectorSizePassBase<
      LLVMCPUTileToVectorSizePass>::LLVMCPUTileToVectorSizePassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

static std::optional<SmallVector<int64_t>>
getTileSizesForEachDims(linalg::LinalgOp op) {
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(op);
  SmallVector<bool> scalableFlags = loweringConfig.getVectorScalableFlags();
  if (llvm::count(scalableFlags, true) > 0) {
    return std::nullopt;
  }

  std::optional<SmallVector<int64_t>> vectorSizes =
      loweringConfig.getVectorSizes();
  if (!vectorSizes) {
    return std::nullopt;
  }

  unsigned int numLoops = op.getNumLoops();
  SmallVector<int64_t> result(numLoops, 0);
  for (unsigned int dim = 0; dim < numLoops; ++dim) {
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    op.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return std::nullopt;
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    if (ShapedType::isStatic(dimSize)) {
      if (vectorSizes.value()[dim] < dimSize) {
        result[dim] = vectorSizes.value()[dim];
      }
      continue;
    }

    // If a tensor.extract_slice can not be found, the operand is not tiled at
    // all. It implies that the dimension is not yet tiled..
    if (!isa<tensor::EmptyOp, tensor::ExtractSliceOp>(
            firstOperand.getDefiningOp())) {
      result[dim] = vectorSizes.value()[dim];
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<DimBoundSize> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      FailureOr<int64_t> maybeDimBoundSize =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, {operand, operandDim},
              /*stopCondition=*/nullptr, /*closedUB=*/true);
      // Try the next pair if the bound can't be found.
      if (failed(maybeDimBoundSize)) {
        continue;
      }
      if (vectorSizes.value()[dim] < maybeDimBoundSize.value()) {
        result[dim] = vectorSizes.value()[dim];
      }
      break;
    }
  }

  return result;
}

void LLVMCPUTileToVectorSizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk([&](linalg::LinalgOp op) {
    // XXX(hanchung): fill is fine.
    if (isa<linalg::FillOp>(op)) {
      return;
    }
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
        getLoweringConfig(op);
    if (!loweringConfig) {
      return;
    }
    if (!loweringConfig.getVectorSizes().has_value()) {
      return;
    }
    candidates.push_back(op);
  });

  IRRewriter rewriter(context);
  for (linalg::LinalgOp op : candidates) {
    LDBG() << "candidate: " << op;
    std::optional<SmallVector<int64_t>> tileSizes = getTileSizesForEachDims(op);
    if (!tileSizes) {
      LDBG() << "all the dimensions are either tiled or target scalable tile "
                "sizes";
      continue;
    }
    if (llvm::all_of(tileSizes.value(), [](int64_t val) { return val == 0; })) {
      LDBG() << "skip the op because tile sizes are all zeros";
      continue;
    }
    LDBG() << "tileSizes: " << llvm::interleaved_array(tileSizes.value());

    auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
    scf::SCFTilingOptions options;
    setSCFTileSizes(options, tilingInterfaceOp, std::move(tileSizes.value()),
                    /*tileScalableFlags=*/{});
    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
    if (failed(tiledResults)) {
      LDBG() << "failed to tile the op";
      return signalPassFailure();
    }
    rewriter.replaceOp(op, tiledResults->replacements);
  }

  RewritePatternSet patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    LDBG() << "----- cleanup failed -----";
    return signalPassFailure();
  }
}
} // namespace
} // namespace mlir::iree_compiler
