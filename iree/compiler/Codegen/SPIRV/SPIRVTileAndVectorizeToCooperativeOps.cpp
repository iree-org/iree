// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndVectorizeToCooperativeOps.cpp --------------------------===//
//
// This pass tiles Linalg ops with buffer semantics to subgroups and vectorizes
// them into vector ops suitable for lowering to SPIR-V cooperative ops.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-vectorize-to-cooperative-ops"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Subgroup tiling patterns
//===----------------------------------------------------------------------===//

/// Gets the chosen hardware cooperative op size attached to the given `op`
/// as CodeGen lowering configuration.
static SmallVector<int64_t> getTargetCooperativeOpSize(linalg::LinalgOp op) {
  return getTileSizes(op, 1);  // For subgroup level tiling
}

/// Deduces required subgroup counts along all workgroup tiled dimensions.
///
/// `op` should be an operation with a `lowering.config` attribute to specify
/// tiling sizes for the workgroup and subgroup.
static SmallVector<int64_t> deduceSubgroupCounts(linalg::LinalgOp op) {
  SmallVector<int64_t> workgroupTileSizes = getTileSizes(op, 0);
  SmallVector<int64_t> subgroupCounts(workgroupTileSizes.size(), 1);

  SmallVector<int64_t> subgroupTileSizes = getTileSizes(op, 1);

  assert(workgroupTileSizes.size() == subgroupTileSizes.size());
  for (int i = 0, e = subgroupTileSizes.size(); i < e; ++i) {
    assert(workgroupTileSizes[i] % subgroupTileSizes[i] == 0);
    subgroupCounts[i] = workgroupTileSizes[i] / subgroupTileSizes[i];
  }
  return subgroupCounts;
}

/// Computes subgroup IDs and counts for distribution.
///
/// GPU's subgroup ID builtin is a single number. We need to delinearize it to
/// all workgroup tiled dimensions for distribution at the subgroup level.
static SmallVector<linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> numSubgroups) {
  Type indexType = builder.getIndexType();
  Value subgroupId = builder.create<gpu::SubgroupIdOp>(loc, indexType);
  SmallVector<linalg::ProcInfo, 2> procInfo(numSubgroups.size());

  // subgroupID = id.z * count.y * count.x + id.y * count.x + id.x
  for (size_t i = 0, e = numSubgroups.size(); i != e; ++i) {
    Value nprocs = builder.create<arith::ConstantIndexOp>(loc, numSubgroups[i]);
    AffineExpr d0 = getAffineDimExpr(0, builder.getContext());
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    Value procId =
        makeComposedAffineApply(builder, loc, d0 % s0, {subgroupId, nprocs});
    procInfo[e - i - 1] = linalg::ProcInfo{procId, nprocs};
    subgroupId = builder.create<arith::DivSIOp>(loc, subgroupId, nprocs);
  }
  return procInfo;
}

/// Adds patterns to tile Linalg ops with workgroup markers to subgroups.
static void populateTilingToSubgroupPatterns(ArrayRef<int64_t> subgroupCounts,
                                             RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  auto getSubgroupProcInfoFn = [subgroupCounts](
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Range> parallelLoopRanges) {
    auto counts = llvm::to_vector<3>(subgroupCounts);
    // Only consider parallel dimensions for tiling and distribution. Reduction
    // dimension distribution needs synchronization. We'll use vector unroll
    // later to "tile" along reduction dimensions.
    unsigned size = std::min(parallelLoopRanges.size(), static_cast<size_t>(3));
    counts.resize(size, 1);
    return getSubgroupIdsAndCounts(builder, loc, counts);
  };

  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getSubgroupProcInfoFn;
  distributionOptions.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};

  auto setTileSizesFn = [](OpBuilder &builder, Operation *op) {
    SmallVector<int64_t> tileSizes = getTileSizes(op, 1);
    // Only consider parallel dimensions for tiling and distribution. Reduction
    // dimension distribution needs synchronization. We'll use vector unroll
    // later to "tile" along reduction dimensions.
    tileSizes.resize(
        std::min(cast<linalg::LinalgOp>(op).getNumParallelLoops(), 3u));
    return llvm::to_vector<4>(
        llvm::map_range(tileSizes, [&](int64_t v) -> Value {
          return builder.create<arith::ConstantIndexOp>(op->getLoc(), v);
        }));
  };
  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizeComputationFunction(setTileSizesFn)
          .setDistributionOptions(distributionOptions);

  auto filter = linalg::LinalgTransformationFilter(
      ArrayRef<Identifier>{}, Identifier::get(getVectorizeMarker(), context));
  linalg::TilingPatterns<linalg::FillOp, linalg::MatmulOp,
                         linalg::GenericOp>::insert(patterns, tilingOptions,
                                                    filter);
}

//===----------------------------------------------------------------------===//
// Vectorization patterns
//===----------------------------------------------------------------------===//

/// Adds patterns to vectorize Linalg ops with vectorization markers.
void populateVectorizationPatterns(MLIRContext *context,
                                   RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::ContractionOpInterface,
                                      linalg::FillOp, linalg::GenericOp>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context)));
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
}

/// Returns vector shape matching native cooperative op sizes for unrolling
/// high-D vectors.
Optional<SmallVector<int64_t, 4>> getCooperativeOpVectorShape(
    Operation *op, ArrayRef<int64_t> nativeShape) {
  // Unroll vector.contract ops according to native cooperative matrix size.
  if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    return llvm::to_vector<4>(nativeShape);
  }

  // Unrolling vector.contract generates vector.{insert|extract}_strided_slice
  // ops for the vector transfer ops associated with the original contract op.
  // We can use those to figure out how to unroll transfer ops accordingly
  // to match the native cooperative op sizes.
  //
  // A better way might be to inspect the SSA value chain to figure out how the
  // transfer ops are used (e.g., for cooperative matrix A/B/C matrix) and use
  // the corresponding cooperative matrix configuration.

  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    auto insert =
        writeOp.vector().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insert) return llvm::None;
    return llvm::to_vector<4>(insert.getSourceVectorType().getShape());
  }

  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    VectorType sliceType;
    for (Operation *users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract) return llvm::None;
      auto vecType = extract.getResult().getType().cast<VectorType>();
      if (sliceType && sliceType != vecType) return llvm::None;
      sliceType = vecType;
    }
    return llvm::to_vector<4>(sliceType.getShape());
  }

  return llvm::None;
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(ArrayRef<int64_t> cooperativeOpSize,
                                  RewritePatternSet &patterns) {
  auto getShapeFn = [cooperativeOpSize](Operation *op) {
    return getCooperativeOpVectorShape(op, cooperativeOpSize);
  };
  auto options = vector::UnrollVectorOptions().setNativeShapeFn(getShapeFn);
  vector::populateVectorUnrollPatterns(patterns, options);
}

/// Fuses vector.transpose into consumer vector.contract.
///
/// This is a workaround for SPIR-V backend limitations. SPIR-V vetorization
/// pass relies on unrolling to reduce instructions to a vector size we can
/// convert to SPIR-V. When vectorization creates transpose those block
/// unrolling and result in large vector we currently cannot lower. For now we
/// always merge the transpose into the contract op so that it can be unrolled.
//
// TODO(thomasraoux): Make transpose work with the current unrolling mechanism
// or replace unrolling.
class CombineContractTranspose final
    : public OpRewritePattern<vector::ContractionOp> {
 public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // Perform lhs + rhs transpositions to conform to matmul row-major
    // semantics. Bail out if the contraction cannot be put in this form.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    bool foundTranspose = false;
    std::array<Value, 3> sources = {op.lhs(), op.rhs(), op.acc()};
    SmallVector<AffineMap> newMaps;
    SmallVector<Value> newSources;
    for (auto source : llvm::enumerate(sources)) {
      auto map =
          op.indexing_maps()[source.index()].cast<AffineMapAttr>().getValue();
      auto tranposeOp = source.value().getDefiningOp<vector::TransposeOp>();
      if (!tranposeOp) {
        newSources.push_back(source.value());
        newMaps.push_back(map);
        continue;
      }
      SmallVector<int64_t, 3> perm;
      tranposeOp.getTransp(perm);
      SmallVector<AffineExpr> exprs(perm.size());
      for (auto remap : llvm::enumerate(perm)) {
        exprs[remap.value()] = map.getResult(remap.index());
      }
      newMaps.push_back(
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx));
      newSources.push_back(tranposeOp.vector());
      foundTranspose = true;
    }
    if (!foundTranspose) return failure();

    Value res = rewriter.create<vector::ContractionOp>(
        loc, newSources[0], newSources[1], newSources[2],
        rewriter.getAffineMapArrayAttr(newMaps), op.iterator_types());
    rewriter.replaceOp(op, res);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

class SPIRVTileAndVectorizeToCooperativeOpsPass final
    : public SPIRVTileAndVectorizeToCooperativeOpsBase<
          SPIRVTileAndVectorizeToCooperativeOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getOperation();

    // First we need to discover the CodeGen lowering configuration. It was
    // decided earlier and attached to a linalg op as an attribute.

    linalg::LinalgOp rootOp;
    funcOp.walk([&](linalg::ContractionOpInterface contractOp) {
      if (getLoweringConfig(contractOp)) {
        rootOp = cast<linalg::LinalgOp>(contractOp.getOperation());
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!rootOp) {
      funcOp.emitError(
          "expected a linalg::ContractionOpInterface op with "
          "lowering.config attribute");
      return signalPassFailure();
    }

    auto cooperativeOpSize = getTargetCooperativeOpSize(rootOp);
    auto subgroupCounts = deduceSubgroupCounts(rootOp);

    // Then tile and distribute to subgroups.

    {
      RewritePatternSet subgroupTilingPatterns(context);
      populateTilingToSubgroupPatterns(subgroupCounts, subgroupTilingPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(subgroupTilingPatterns)))) {
        return signalPassFailure();
      }

      RewritePatternSet canonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to subgroups ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Now vectorize and unroll to native cooperative sizes.

    {
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(context, vectorizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorizationPatterns)))) {
        return signalPassFailure();
      }

      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After vectorization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(cooperativeOpSize, vectorUnrollPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // At the last perform various canonicalization and cleanups.

    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting vector transfers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet canonicalizationPatterns(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After canonicalizing vectors ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // When using cooperative matrix we don't want to lower the contract,
    // instead we want to merge contract and transpose so that they can be
    // converted to cooperative matrix matmul op.
    RewritePatternSet combineTransposePatterns(context);
    combineTransposePatterns.add<CombineContractTranspose>(context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(combineTransposePatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After handling transposes ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createSPIRVTileAndVectorizeToCooperativeOpsPass() {
  return std::make_unique<SPIRVTileAndVectorizeToCooperativeOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
