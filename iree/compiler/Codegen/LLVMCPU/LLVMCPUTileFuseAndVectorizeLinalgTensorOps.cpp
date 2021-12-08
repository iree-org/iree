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
#include "llvm/Support/Debug.h"
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

#define DEBUG_TYPE "iree-llvmcpu-tile-fuse-and-vectorize"

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
struct LLVMCPUTileFuseAndVectorizePass
    : public LLVMCPUTileFuseAndVectorizeBase<LLVMCPUTileFuseAndVectorizePass> {
  LLVMCPUTileFuseAndVectorizePass(bool vectorize = true)
      : lowerToVectors(vectorize) {}
  LLVMCPUTileFuseAndVectorizePass(const LLVMCPUTileFuseAndVectorizePass &pass) {
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

LogicalResult applyTileAndFuseCanonicalizationPatterns(FuncOp funcOp) {
  auto context = funcOp.getContext();
  OwningRewritePatternList patterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  tensor::DimOp::getCanonicalizationPatterns(patterns, context);
  memref::DimOp::getCanonicalizationPatterns(patterns, context);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
}  // namespace

void LLVMCPUTileFuseAndVectorizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- Before LLVMCPUTileFuseAndVectorize ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Assume there is a single op with a lowering config we use to drive the
  // tiling decisions.
  // TODO(hanchung): Speicify a callback to get tile sizes in tile+fuse after
  // upstream method supports it. Then we don't need extracting the config.
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

  // Tile and fuse Linalg ops.
  {
    OpBuilder builder(funcOp.getContext());
    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return signalPassFailure();
    }
    auto tileSizes =
        config.getTileSizeVals(static_cast<unsigned>(TilingLevel::L1Tiles));
    linalg::LinalgOp consumerOp;
    for (auto iter : llvm::reverse(computeOps)) {
      if (auto op = dyn_cast<linalg::LinalgOp>(iter)) {
        consumerOp = op;
        break;
      }
    }
    assert(consumerOp && "can't find consumerOp");
    SmallVector<int64_t> consumerTileSize(
        tileSizes.begin(),
        tileSizes.begin() + consumerOp.getNumParallelLoops());
    auto identityIndicesOrder =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, consumerTileSize.size()));
    FailureOr<linalg::TileLoopNest> tileLoopNest =
        linalg::tileConsumerAndFuseProducers(
            builder, consumerOp, consumerTileSize, identityIndicesOrder);
    if (failed(tileLoopNest)) return signalPassFailure();
    consumerOp->replaceAllUsesWith(tileLoopNest->getRootOpReplacementResults());

    // Apply canoncalization
    if (failed(applyTileAndFuseCanonicalizationPatterns(funcOp))) {
      return signalPassFailure();
    }
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After tile and fuse paralell loops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    OwningRewritePatternList tileReductionPatterns(&getContext());

    // TODO(hanchung): Add a pattern to fold the tensor.extract_slice op.
    // One-trip loop can be removed. But weird patterns could be generated and
    // can't be folded atm. E.g.,
    //   %a = linalg.init_tensor [%x, 4] : tensor<?x4xf32>
    //   %b = linalg.fill(%cst0, %a)
    //   %c = tensor.extract_slice %b[0, 0] [%x, 4] [1, 1]
    //
    // In this case, %c should be folded. Otherwise, it introduces memref.alloca
    // in bufferization.
    bool shouldTileReductionLoop = true;
    funcOp.walk([&](linalg::ContractionOpInterface op) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
      auto loopRanges = linalgOp.getStaticLoopRanges();
      if (loopRanges) {
        auto l1Tiles =
            getTileSizes(op, static_cast<unsigned>(TilingLevel::L1Tiles));
        for (int i = linalgOp.getNumParallelLoops(); i < l1Tiles.size(); ++i) {
          if (loopRanges.getValue()[i] != ShapedType::kDynamicSize &&
              l1Tiles[i] && loopRanges.getValue()[i] <= l1Tiles[i]) {
            shouldTileReductionLoop = false;
          }
        }
      }
    });
    if (shouldTileReductionLoop) {
      tileReductionPatterns.insert<TileWorkgroups>(
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

      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(tileReductionPatterns)))) {
        return signalPassFailure();
      }
      // Apply canoncalization
      if (failed(applyTileAndFuseCanonicalizationPatterns(funcOp))) {
        return signalPassFailure();
      }
      DEBUG_WITH_TYPE(DEBUG_TYPE, {
        llvm::dbgs() << "\n--- After tiling reduction loops ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
  }

  funcOp.walk([&](linalg::ContractionOpInterface op) {
    if (failed(linalg::vectorizeLinalgOpPrecondition(op))) {
      lowerToVectors = false;
    }
  });
  if (!lowerToVectors) {
    // Apply second level of tiling patterns if they are not vectorizable. This
    // will trigger LLVM auto-vectorization, which gains better performance.
    {
      funcOp.walk([&](linalg::ContractionOpInterface op) {
        setMarker(op, getWorkgroupL1TileMarker());
      });
      OwningRewritePatternList l2patterns(&getContext());
      l2patterns.insert<TileWorkgroups>(
          context,
          linalg::LinalgTilingOptions().setTileSizeComputationFunction(
              [](OpBuilder &builder, Operation *op) -> SmallVector<Value, 4> {
                return getTileSizes(
                    builder, op,
                    static_cast<unsigned>(TilingLevel::VectorTiles));
              }),
          linalg::LinalgTransformationFilter(
              Identifier::get(getWorkgroupL1TileMarker(), context),
              Identifier::get(getVectorizeMarker(), context)));

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(l2patterns)))) {
        return signalPassFailure();
      }
      DEBUG_WITH_TYPE(DEBUG_TYPE, {
        llvm::dbgs() << "\n--- After second level of tiling patterns ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    return;
  }

  {
    // Set vectorization marker globally
    OpBuilder builder(funcOp.getContext());
    funcOp.walk(
        [&](linalg::LinalgOp op) { setMarker(op, getVectorizeMarker()); });
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
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }

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
    // TODO(hanchung): Set different vector sizes for different operations. Also
    // it seems that `{16, 16, 16}` is not a good config. We should tune it.
    vector::populateVectorUnrollPatterns(
        vectorUnrollPatterns,
        vector::UnrollVectorOptions().setNativeShape(config.getTileSizeVals(
            static_cast<unsigned>(TilingLevel::VectorTiles))));

    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(vectorUnrollPatterns)))) {
      return signalPassFailure();
    }
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After vector unrolling ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

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

std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUTileFuseAndVectorizePass(
    bool lowerToVectors) {
  return std::make_unique<LLVMCPUTileFuseAndVectorizePass>(lowerToVectors);
}

}  // namespace iree_compiler
}  // namespace mlir
