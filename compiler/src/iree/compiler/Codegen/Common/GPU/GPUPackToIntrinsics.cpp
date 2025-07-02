// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPACKTOINTRINSICSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUPackToIntrinsicsPass final
    : impl::GPUPackToIntrinsicsPassBase<GPUPackToIntrinsicsPass> {
  void runOnOperation() override;
};
} // namespace

FailureOr<SmallVector<OpFoldResult>>
getPackedSizes(linalg::LinalgOp linalgOp, RewriterBase &rewriter,
               IREE::GPU::MmaInterfaceAttr kind) {
  SmallVector<int64_t> dims;
  SmallVector<SmallVector<unsigned, 2>> indices;
  auto createPackedSizes =
      [&rewriter, &linalgOp](SmallVector<int64_t> dims,
                             SmallVector<SmallVector<unsigned, 2>> indices)
      -> FailureOr<SmallVector<OpFoldResult>> {
    auto zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> packedSizes(linalgOp.getNumLoops(), zero);
    for (auto [dim, index] : llvm::zip_equal(dims, indices)) {
      if (index.empty()) {
        linalgOp.emitError()
            << "contraction like operation missing critical dimension\n";
        return failure();
      }
      packedSizes[index.back()] = rewriter.getIndexAttr(dim);
    }
    return packedSizes;
  };

  FailureOr<linalg::ScaledContractionDimensions> scaledContrDims =
      linalg::inferScaledContractionDims(linalgOp);
  if (succeeded(scaledContrDims)) {
    auto [m, n, k] = kind.getScaledMNKShape();
    indices = {scaledContrDims->m, scaledContrDims->n, scaledContrDims->k};
    return createPackedSizes({m, n, k}, indices);
  }

  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    linalgOp.emitError() << "failed to infer contraction dims\n";
    return failure();
  }
  auto [m, n, k] = kind.getMNKShape();
  indices = {scaledContrDims->m, scaledContrDims->n, scaledContrDims->k};
  return createPackedSizes({m, n, k}, indices);
}

LogicalResult packToIntrinsic(linalg::LinalgOp linalgOp,
                              RewriterBase &rewriter) {
  llvm::errs() << "[DEBUG] - packToIntrinsic\n";
  auto loweringConfig =
      getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
  assert(loweringConfig && "Packing unconfigured op");
  IREE::GPU::MmaInterfaceAttr kind = getMmaKind(loweringConfig);
  assert(kind && "Packing op without mma kind");
  FailureOr<SmallVector<OpFoldResult>> packedSizes =
      getPackedSizes(linalgOp, rewriter, kind);
  linalgOp.print(llvm::errs()), llvm::errs() << "\n!?! \n";
  FailureOr<linalg::PackResult> maybeResult =
      linalg::pack(rewriter, linalgOp, packedSizes.value());
  maybeResult->packedLinalgOp.print(llvm::errs()), llvm::errs() << "\n!??????????????????! \n";
  if (failed(maybeResult)) {
    return rewriter.notifyMatchFailure(linalgOp, "packing failed");
  }
  setLoweringConfig(maybeResult->packedLinalgOp, loweringConfig);
  return success();
}

struct ConvertToMultiMma final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "[DEBUG] - ConvertToMultiMma \n";
    linalgOp.print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "[DEBUG] - IN\n";
                                  auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return failure();
    }
    IREE::GPU::MmaInterfaceAttr kind = getMmaKind(loweringConfig);
    if (!kind) {
      return failure();
    }
    llvm::errs() << "[DEBUG] - NEEDS TO GET HERE\n";
    if (failed(convertContractionToInnerTiledMma(rewriter, linalgOp, kind))) {
      return failure();
    }
    return success();
  }
};

void GPUPackToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  llvm::errs() << "[DEBUG] - GPUPackToIntrinsicsPass\n";
  // Step 1. Pack candidate linalg ops to specified shapes.
  IRRewriter rewriter(funcOp);
  SmallVector<linalg::LinalgOp> packingCandidates;
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return;
    }
    if (!getMmaKind(loweringConfig)) {
      llvm::errs() << "[DEBUG] - !getMmaKind(loweringConfig)\n";
      return;
    }
    packingCandidates.push_back(linalgOp);
    llvm::errs() << "[DEBUG] - packingCandidates\n";
    linalgOp.print(llvm::errs()), llvm::errs() << "\n";
  });

  for (auto candidate : packingCandidates) {
    rewriter.setInsertionPoint(candidate);
    if (failed(packToIntrinsic(candidate, rewriter))) {
      funcOp.emitError() << "failed to pack operation marked with intrinsic\n";
      return signalPassFailure();
    }
  }

  // Step 2. Convert configured linalg ops to inner_tiled ops with multi-MMA
  // intrinsic kinds.
  {
    llvm::errs() << "convert to multimma\n";
    RewritePatternSet patterns(context);
    patterns.add<ConvertToMultiMma>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "failed to convert linalg to multi-MMA inner_tiled";
      return signalPassFailure();
    }
  }

  // Step 3. Run layout propagation patterns to pull in adjacent un-configured
  // ops.
  RewritePatternSet patterns(context);
  linalg::ControlPropagationFn control = [](OpOperand *opOperand) -> bool {
    Operation *producer = opOperand->get().getDefiningOp();
    Operation *consumer = opOperand->getOwner();
    return !getLoweringConfig(producer) && !getLoweringConfig(consumer);
  };

  linalg::populateDataLayoutPropagationPatterns(patterns, control);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
