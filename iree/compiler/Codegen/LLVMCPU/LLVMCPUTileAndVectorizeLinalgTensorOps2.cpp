// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tile-and-vectorize"

namespace mlir {
namespace iree_compiler {

namespace {
// Could just be linalg::TilingPattern with a ContractionOpInterface filter, but
// that is always templated on an op.
struct TileWorkgroups : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileWorkgroups(MLIRContext *context, linalg::LinalgTilingOptions options,
                 linalg::LinalgTransformationFilter marker)
      : LinalgBaseTilingPattern(context, options, marker) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto contractionOp = dyn_cast<linalg::ContractionOpInterface>(op);
    if (!contractionOp) return failure();

    linalg::TiledLinalgOp tiledLinalgOp;
    if (failed(Base::matchAndRewriteBase(op, rewriter, tiledLinalgOp))) {
      return failure();
    }
    rewriter.replaceOp(op, tiledLinalgOp.tensorResults);
    return success();
  }
};

}  // namespace

namespace {
struct LLVMCPUTileAndVectorizePass2
    : public LLVMCPUTileAndVectorize2Base<LLVMCPUTileAndVectorizePass2> {
  LLVMCPUTileAndVectorizePass2(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  LLVMCPUTileAndVectorizePass2(const LLVMCPUTileAndVectorizePass2 &pass) {
    lowerToVectors = pass.lowerToVectors;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;

 private:
  bool lowerToVectors;
};
}  // namespace

void LLVMCPUTileAndVectorizePass2::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- Before LLVMCPUTileAndVectorizePass2 ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Assume there is a single op with a lowering config we use to drive the
  // tiling decisions.
  IREE::Codegen::LoweringConfigAttr config;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (auto opConfig = getLoweringConfig(linalgOp)) {
      if (opConfig) {
        // Duplicate configurations.
        if (config) return signalPassFailure();
        config = opConfig;
      }
    }
  });

  // Skip vetorization if not all the ops can be vectorized.
  bool isVectorizable = true;
  {
    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return signalPassFailure();
    }

    for (auto op : computeOps) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (failed(linalg::vectorizeLinalgOpPrecondition(op))) {
          isVectorizable = false;
        }
      }
    }
  }

  // Query if first level of tiling is needed.
  auto l1TileSizes = config.getTileSizeVals(1);
  if (!l1TileSizes.empty() && isVectorizable) {
    // First tile and fuse paralell loops.
    OpBuilder builder(funcOp.getContext());
    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return signalPassFailure();
    }
    auto consumerOp = dyn_cast<linalg::LinalgOp>(computeOps.back());
    SmallVector<int64_t> consumerTileSize(
        l1TileSizes.begin(),
        l1TileSizes.begin() + consumerOp.getNumParallelLoops());
    auto identityIndicesOrder =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, consumerTileSize.size()));
    FailureOr<linalg::TileLoopNest> tileLoopNest =
        linalg::tileConsumerAndFuseProducers(
            builder, consumerOp, consumerTileSize, identityIndicesOrder);
    if (failed(tileLoopNest)) return signalPassFailure();
    consumerOp->replaceAllUsesWith(tileLoopNest->getRootOpReplacementResults());

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After tile and fuse paralell loops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Apply canoncalization
    {
      OwningRewritePatternList canonicalizationPatterns(&getContext());
      linalg::populateLinalgTilingCanonicalizationPatterns(
          canonicalizationPatterns);
      memref::DimOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                 context);
      scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After canonicalization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Tile reduction loops.
    {
      OwningRewritePatternList l1patterns(&getContext());
      l1patterns.insert<TileWorkgroups>(
          context,
          linalg::LinalgTilingOptions().setTileSizeComputationFunction(
              [](OpBuilder &builder,
                 Operation *operation) -> SmallVector<Value, 4> {
                auto tiles =
                    getTileSizes(builder, operation,
                                 static_cast<unsigned>(TilingLevel::L1Tiles));
                auto numParallelLoops =
                    dyn_cast<linalg::LinalgOp>(operation).getNumParallelLoops();
                auto zeroTileVal = builder.create<arith::ConstantIndexOp>(
                    operation->getLoc(), 0);
                SmallVector<Value> reductionTiles(tiles.size(), zeroTileVal);
                for (int i = numParallelLoops; i < tiles.size(); ++i) {
                  reductionTiles[i] = tiles[i];
                }
                return std::move(reductionTiles);
              }),
          linalg::LinalgTransformationFilter(
              ArrayRef<Identifier>{},
              Identifier::get(getVectorizeMarker(), context)));

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(l1patterns)))) {
        return signalPassFailure();
      }
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After tiling reduction loops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Apply canoncalization
    {
      OwningRewritePatternList canonicalizationPatterns(context);
      linalg::populateLinalgTilingCanonicalizationPatterns(
          canonicalizationPatterns);
      tensor::DimOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                 context);
      memref::DimOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                 context);
      scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }

      DEBUG_WITH_TYPE(DEBUG_TYPE, {
        llvm::dbgs() << "\n--- After canonicalization ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
  }

  if (!lowerToVectors || !isVectorizable) {
    return;
  }

  {
    // Set vectorization marker globally
    OpBuilder builder(funcOp.getContext());
    funcOp.walk([&](linalg::LinalgOp op) {
      op->setAttr("__internal_linalg_transform__",
                  builder.getStringAttr("vectorize"));
    });
  }

  // Apply vectorization patterns.
  {
    OwningRewritePatternList vectorizationPatterns(&getContext());
    linalg::insertVectorizationPatterns<linalg::ContractionOpInterface,
                                        linalg::GenericOp, linalg::CopyOp,
                                        linalg::FillOp>(
        vectorizationPatterns, linalg::LinalgVectorizationOptions(),
        linalg::LinalgTransformationFilter(
            Identifier::get(getVectorizeMarker(), context)));
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorizationPatterns)))) {
      return signalPassFailure();
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After vectorization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    // Fold consumer add ops into the contraction op itself.
    RewritePatternSet canonicalizationPatterns(context);
    vector::ContractionOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                       context);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs()
          << "\n--- After folding consumer add ops into contraction op "
             "iteself ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  // Apply vector unroll
  {
    RewritePatternSet vectorUnrollPatterns(context);
    vector::populateVectorUnrollPatterns(
        vectorUnrollPatterns, vector::UnrollVectorOptions().setNativeShape(
                                  config.getNativeVectorSizeVals()));
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(vectorUnrollPatterns)))) {
      return signalPassFailure();
    }
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- After vector unroll ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
  linalg::hoistRedundantVectorTransfers(funcOp);

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "--- After hoisting vector transfers ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Apply vector specific operation lowering.
  {
    vector::VectorTransformsOptions vectorTransformsOptions =
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct);
    OwningRewritePatternList vectorContractLoweringPatterns(&getContext());
    vectorContractLoweringPatterns.insert<
        vector::ContractionOpToOuterProductOpLowering,
        vector::ContractionOpToMatmulOpLowering, vector::ContractionOpLowering>(
        vectorTransformsOptions, context);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorContractLoweringPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorContractLoweringPatterns)))) {
      return signalPassFailure();
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After vector specific operatrion lowering ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUTileAndVectorizePass2(
    bool lowerToVectors) {
  return std::make_unique<LLVMCPUTileAndVectorizePass2>(lowerToVectors);
}

}  // namespace iree_compiler
}  // namespace mlir
