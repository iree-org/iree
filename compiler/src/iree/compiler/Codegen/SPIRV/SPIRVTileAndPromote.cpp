// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndPromote.cpp --------------------------------------------===//
//
// This pass tiles promote Linalg ops with buffer semantics to use workgroup
// memory and then tiles to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-promote"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVTILEANDPROMOTEPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static LogicalResult
tileReductionLoops(mlir::FunctionOpInterface funcOp,
                   LinalgTransformationFilter filter,
                   const scf::SCFTileSizeComputationFunction &computeFn) {
  auto options =
      scf::SCFTilingOptions().setTileSizeComputationFunction(computeFn);
  return tileLinalgOpsWithFilter(funcOp, options, filter);
}

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

static LogicalResult
tileToInvocation(mlir::FunctionOpInterface funcOp,
                 LinalgTransformationFilter filter,
                 const linalg::TileSizeComputationFunction &computeFn) {
  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(computeFn)
                           .setDistributionOptions(distributionOptions);

  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Promotion patterns
//===----------------------------------------------------------------------===//

static const char promoteBothMarker[] = "promote_lhs_and_rhs";

static void populatePromotionPatterns(RewritePatternSet &patterns,
                                      StringAttr replaceMarker) {
  MLIRContext *context = patterns.getContext();
  auto baseOptions =
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setUseFullTileBuffers({false, false});
  auto promoteBothOptions = baseOptions.setOperandsToPromote({0, 1});

  LinalgTransformationFilter promoteBothFilter(
      {StringAttr::get(context, promoteBothMarker)}, replaceMarker);

  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>,
                  LinalgPromotionPattern<linalg::GenericOp>>(
      context, promoteBothOptions, promoteBothFilter);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

class SPIRVTileAndPromotePass final
    : public impl::SPIRVTileAndPromotePassBase<SPIRVTileAndPromotePass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;

private:
  /// Promotes C matrix to shared memory when necessary and returns success if
  /// no error happens.
  LogicalResult doPromoteCMatrix(mlir::FunctionOpInterface funcOp) const;
};

} // namespace

void SPIRVTileAndPromotePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  auto threadTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 1);
  if (failed(threadTileComputeFn))
    return signalPassFailure();
  auto reductionTileComputeFn = getSPIRVScfTileSizeComputeFn(funcOp, 2);
  if (failed(reductionTileComputeFn))
    return signalPassFailure();

  // Promote C matrix and propagate the potential fill producer into the
  // allocation. This needs to be done before reduction tiling.
  if (failed(doPromoteCMatrix(funcOp)))
    return signalPassFailure();

  StringLiteral markerAttrName = LinalgTransforms::kLinalgTransformMarker;
  auto workgroupMarker = StringAttr::get(context, getWorkgroupMemoryMarker());
  auto kTiledMarker = StringAttr::get(context, getWorkgroupKTiledMarker());

  { // Tile reduction dimensions.
    RewritePatternSet patterns(context);
    LinalgTransformationFilter filter(
        // Going through C matrix promotion we will have the marker..
        {workgroupMarker}, kTiledMarker);
    // Not going through C matrix promotion we will have no marker..
    filter.setMatchByDefault();
    if (failed(tileReductionLoops(funcOp, filter, *reductionTileComputeFn))) {
      funcOp.emitOpError() << "failed tiling reduction";
      return signalPassFailure();
    }
  }
  {
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After tiling reduction dimensions ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
      getWorkgroupSize(funcOp);
  if (!maybeWorkgroupSize) {
    funcOp.emitOpError(
        "failed to get workgroup size for tile and promote pass");
    return signalPassFailure();
  }

  SmallVector<int64_t> &workgroupSize = maybeWorkgroupSize.value();
  int64_t totalThreads = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
  if (!subgroupSize) {
    funcOp.emitError("failed to query subgroup size");
    return signalPassFailure();
  }

  // Only promote to workgroup size if there are multiple warps.
  if (totalThreads > *subgroupSize) {
    // Attach markers to contract ops to drive promotion.
    funcOp.walk([&](linalg::LinalgOp op) {
      if (isMatmulOrBatchMatmul(op)) {
        auto promoteMarker = StringAttr::get(context, promoteBothMarker);
        op->setAttr(markerAttrName, promoteMarker);
      }
    });

    RewritePatternSet promotionPatterns(context);
    populatePromotionPatterns(promotionPatterns, workgroupMarker);
    if (failed(applyPatternsGreedily(funcOp, std::move(promotionPatterns)))) {
      return signalPassFailure();
    }

    // Insert barriers before and after copies to workgroup memory.
    insertBarriersAroundSharedMemoryCopy(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After inserting barriers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // If we fail to promote (e.g., for cases where we just have one tile so
    // that there are no subview ops), clear markers to enable following steps.
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      auto marker = linalgOp->getAttrOfType<StringAttr>(markerAttrName);
      if (!marker)
        return WalkResult::advance();
      if (marker.getValue() == promoteBothMarker)
        linalgOp->removeAttr(markerAttrName);
      return WalkResult::advance();
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After promotion ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Attach markers to ops without them to drive tiling next.
  funcOp.walk([&](linalg::LinalgOp op) {
    auto marker = op->getAttrOfType<StringAttr>(markerAttrName);
    if (!marker || marker.getValue() != getCopyToWorkgroupMemoryMarker()) {
      op->setAttr(markerAttrName, workgroupMarker);
    }
  });

  if (!skipThreadLevel) { // Tile and distribute to invocations.
    LinalgTransformationFilter filter({workgroupMarker}, std::nullopt);
    if (failed(tileToInvocation(funcOp, filter, *threadTileComputeFn))) {
      funcOp.emitOpError() << "failed tiling and distributing to invocations";
      return signalPassFailure();
    }

    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
    populateFoldAffineMinInDistributedLoopsPatterns(patterns, numWorkgroups);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

LogicalResult SPIRVTileAndPromotePass::doPromoteCMatrix(
    mlir::FunctionOpInterface funcOp) const {
  MLIRContext *context = funcOp.getContext();
  if (!promoteCMatrix)
    return success();

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  SmallVector<Operation *> linalgOps;
  for (Operation *op : computeOps) {
    if (isa<linalg::FillOp>(op))
      continue; // Don't care
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      linalgOps.push_back(linalgOp);
    } else {
      return funcOp.emitError("unknown compute op ") << *op;
    }
  }

  if (linalgOps.size() > 2) {
    return funcOp.emitError("unhandled multiple matmul/generic cases");
  }

  // If there are no fused elementwise ops, we can avoid promoting C matrix.
  if (linalgOps.size() <= 1)
    return success();

  auto matmulOp = cast<linalg::LinalgOp>(linalgOps.front());
  auto genericOp = cast<linalg::GenericOp>(*linalgOps.back());

  auto matmulType =
      llvm::cast<MemRefType>(matmulOp.getDpsInitOperand(0)->get().getType());
  if (hasSharedMemoryAddressSpace(matmulType)) {
    // The matmul output is already in shared memory. This can happen when
    // bufferization decides an allocation is needed, e.g., matmul + arith.extf,
    // where the output have different element types. For such cases, don't need
    // to promote and propagate shared memory copy anymore. Just mark the
    // following generic op for distribution accordingly.
    setMarker(genericOp, getCopyToWorkgroupMemoryMarker());
    return success();
  }

  // If the fused elementwise ops are allowed to use cooperative types, we can
  // also avoid promoting C matrix.
  if (isCooperativeMatrixFusable(genericOp))
    return success();

  // Finally do promote C matrix.
  RewritePatternSet patterns(context);
  populateContractPromotionPatterns(patterns, {2});
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "--- After promoting C matrix ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  propagateSharedMemoryCopy(funcOp);
  LLVM_DEBUG({
    llvm::dbgs() << "--- After propagating shared memory copy ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
  return success();
}

} // namespace mlir::iree_compiler
