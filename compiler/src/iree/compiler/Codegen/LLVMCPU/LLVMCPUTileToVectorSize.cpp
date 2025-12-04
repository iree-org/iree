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
#include "llvm/Support/LogicalResult.h"
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
  using Base::Base;

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

  unsigned numLoops = op.getNumLoops();
  std::optional<SmallVector<int64_t>> vectorSizes =
      loweringConfig.getVectorSizes();
  if (!vectorSizes || vectorSizes->size() != numLoops) {
    return std::nullopt;
  }
  LDBG() << "configured vector sizes: "
         << llvm::interleaved_array(vectorSizes.value());

  SmallVector<int64_t> result(numLoops, 0);
  for (unsigned dim = 0; dim < numLoops; ++dim) {
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    op.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return std::nullopt;
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize =
        cast<ShapedType>(firstOperand.getType()).getShape()[firstOperandDim];
    int64_t vectorDimSize = vectorSizes.value()[dim];
    if (ShapedType::isStatic(dimSize) && dimSize > vectorDimSize) {
      LDBG() << "set dim #" << dim << " size (" << dimSize
             << ") with vector size: " << vectorDimSize;
      result[dim] = vectorDimSize;
      continue;
    }

    // If a `tensor.extract_slice` op can not be found, the operand is not tiled
    // at all. It implies that the dimension is not yet tiled. `tensor.empty` is
    // part of tiling artifacts that can be used to infer tiling sizes.
    if (!isa_and_present<tensor::EmptyOp, tensor::ExtractSliceOp>(
            firstOperand.getDefiningOp())) {
      LDBG() << "set dim #" << dim
             << " size (untiled) with vector size: " << vectorDimSize;
      result[dim] = vectorDimSize;
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    std::optional<int64_t> maybeDimSize;
    FailureOr<DimBoundSize> maybeDimBound;
    for (auto [operand, operandDim] : operandDimPairs) {
      FailureOr<int64_t> maybeDimBoundSize =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, {operand, operandDim},
              /*stopCondition=*/nullptr, /*closedUB=*/true);
      if (succeeded(maybeDimBoundSize)) {
        maybeDimSize = maybeDimBoundSize.value();
        break;
      }
    }
    // Assume that the unknown dimension size implies the dimension is already
    // tiled. It means that the dimension is definitely tiled, but it is hard to
    // infer the tile size. It usually happens in fusion case, so the pass
    // assumes that it is not needed.
    if (maybeDimSize && maybeDimSize.value() > vectorDimSize) {
      LDBG() << "set dim #" << dim << " size (" << maybeDimSize.value()
             << ") with vector size: " << vectorDimSize;
      result[dim] = vectorDimSize;
    } else {
      LDBG() << "dim #" << dim << " either is tiled to vector size ("
             << vectorDimSize << ") or has complex size computation";
    }
  }

  return result;
}

void LLVMCPUTileToVectorSizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk([&](linalg::LinalgOp op) {
    // XXX(hanchung): linalg.fill usually follow the reduction consumer ops, so
    // the additional tiling is not needed. Otherwise, it results in an
    // additional loops before converting it to a vector. We may need to fix the
    // lowering config issue, but it is a fair stopgap in practice.
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
