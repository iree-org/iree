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

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-promote"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

constexpr int kMaxVectorNumBits = 128;

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static void populateTilingReductionPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::LinalgTransformationFilter filter,
    const linalg::TileSizeComputationFunction &computeFn) {
  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(computeFn);
  TilingPatterns<linalg::BatchMatmulOp, linalg::MatmulOp,
                 linalg::GenericOp>::insert(patterns, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

static void populateTilingToInvocationPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::LinalgTransformationFilter filter,
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

  TilingPatterns<linalg::BatchMatmulOp, linalg::FillOp, linalg::GenericOp,
                 linalg::MatmulOp>::insert(patterns, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Promotion patterns
//===----------------------------------------------------------------------===//

static const char promoteBothMarker[] = "promote_lhs_and_rhs";

template <typename T>
using LinalgPromotionPattern =
    mlir::iree_compiler::IREE::LinalgExt::LinalgPromotionPattern<T>;

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

  IREE::LinalgExt::LinalgTransformationFilter promoteBothFilter(
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
    : public SPIRVTileAndPromoteBase<SPIRVTileAndPromotePass> {
 public:
  SPIRVTileAndPromotePass(bool promoteCMatrix, bool skipThreadLevel)
      : promoteCMatrix(promoteCMatrix), skipThreadLevel(skipThreadLevel) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) return failure();
    // Consider pass option too
    promoteCMatrix |= this->promoteC;
    skipThreadLevel |= this->skipThread;
    return success();
  }

  void runOnOperation() override;

 private:
  /// Promotes C matrix to shared memory when necessary and returns success if
  /// no error happens.
  LogicalResult doPromoteCMatrix(func::FuncOp funcOp) const;

  /// Returns true if the given generic op is an elementwise op that can use
  /// cooperative matrix type eventually.
  bool isCooperativeMatrixFusable(linalg::GenericOp genericOp) const;

  // Whether to promote C matrix to use shared memory.
  bool promoteCMatrix = false;
  // Whether to skip thread level tiling and distribution.
  bool skipThreadLevel = false;
};

}  // namespace

void SPIRVTileAndPromotePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp)) return;

  auto threadTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 1);
  if (failed(threadTileComputeFn)) return signalPassFailure();
  auto reductionTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 2);
  if (failed(reductionTileComputeFn)) return signalPassFailure();

  // Promote C matrix and propagate the potential fill producer into the
  // allocation. This needs to be done before reduction tiling.
  if (failed(doPromoteCMatrix(funcOp))) return signalPassFailure();

  StringLiteral markerAttrName =
      IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker;
  auto workgroupMarker = StringAttr::get(context, getWorkgroupMemoryMarker());
  auto kTiledMarker = StringAttr::get(context, getWorkgroupKTiledMarker());

  {  // Tile reduction dimensions.
    RewritePatternSet patterns(context);
    IREE::LinalgExt::LinalgTransformationFilter filter(
        // Going through C matrix promotion we will have the marker..
        {workgroupMarker}, kTiledMarker);
    // Not going through C matrix promotion we will have no marker..
    filter.setMatchByDefault();
    populateTilingReductionPatterns(patterns, filter, *reductionTileComputeFn);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError() << "failed tiling reduction";
      return signalPassFailure();
    }
  }
  {
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After tiling reduction dimensions ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
      exportOp->getWorkgroupSize().value(),
      [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
  int64_t totalThreads = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  Optional<int> subgroupSize = getSPIRVSubgroupSize(funcOp);
  if (!subgroupSize) {
    funcOp->emitError("failed to query subgroup size");
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
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(promotionPatterns)))) {
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
      if (!marker) return WalkResult::advance();
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

  if (!skipThreadLevel) {  // Tile and distribute to invocations.
    RewritePatternSet tilingPatterns(context);
    IREE::LinalgExt::LinalgTransformationFilter filter({workgroupMarker},
                                                       std::nullopt);
    populateTilingToInvocationPatterns(tilingPatterns, filter,
                                       *threadTileComputeFn);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(tilingPatterns)))) {
      funcOp.emitOpError() << "failed tiling and distributing to invocations";
      return signalPassFailure();
    }

    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    populateFoldAffineMinInDistributedLoopsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
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
    func::FuncOp funcOp) const {
  MLIRContext *context = funcOp.getContext();
  if (!promoteCMatrix) return success();

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  SmallVector<Operation *> linalgOps;
  for (Operation *op : computeOps) {
    if (isa<linalg::FillOp>(op)) continue;  // Don't care
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
  if (linalgOps.size() <= 1) return success();

  linalg::LinalgOp matmulOp = linalgOps.front();
  auto genericOp = cast<linalg::GenericOp>(*linalgOps.back());

  auto matmulType =
      matmulOp.getDpsInitOperand(0)->get().getType().cast<MemRefType>();
  if (isInWorkgroupMemory(matmulType)) {
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
  if (isCooperativeMatrixFusable(genericOp)) return success();

  // Finally do promote C matrix.
  RewritePatternSet patterns(context);
  populateContractPromotionPatterns(patterns, {2});
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
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

bool SPIRVTileAndPromotePass::isCooperativeMatrixFusable(
    linalg::GenericOp genericOp) const {
  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) return false;
  // Limit to outer dimension broadcasted cases for now.
  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    if (!map.isMinorIdentity()) return false;
  }

  for (Operation &op : genericOp.getBlock()->without_terminator()) {
    if (!isa<
            // These ops are directly allowed to use cooperative matrix types.
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::NegFOp,
            arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp,
            arith::UIToFPOp,
            // Special cases of these ops are directly allowed to sue
            // cooperative matrix types. Other cases can use a loop.
            arith::MulFOp>(op))
      return false;
  }
  return true;
}

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndPromotePass(
    bool promoteCMatrix, bool skipThreadLevel) {
  return std::make_unique<SPIRVTileAndPromotePass>(promoteCMatrix,
                                                   skipThreadLevel);
}

}  // namespace iree_compiler
}  // namespace mlir
