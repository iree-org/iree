// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/GetNumWorkgroups.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Attributes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

namespace {
struct LinalgTileAndDistributePass
    : public PassWrapper<LinalgTileAndDistributePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREEDialect, AffineDialect,
                    scf::SCFDialect>();
  }
  LinalgTileAndDistributePass() = default;
  LinalgTileAndDistributePass(const LinalgTileAndDistributePass &pass) {}
  void runOnOperation() override;

 private:
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Set tile sizes to use"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
}  // namespace

namespace {
template <typename LinalgOpTy>
struct TileToCPUThreads : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileToCPUThreads(MLIRContext *context,
                   const linalg::LinalgDependenceGraph &dependenceGraph,
                   const CPUKernelDispatch &cpuKernelDispatch,
                   linalg::LinalgTilingOptions options,
                   linalg::LinalgMarker marker, PatternBenefit benefit = 1)
      : Base(LinalgOpTy::getOperationName(), context, options, marker, benefit),
        cpuKernelDispatch(cpuKernelDispatch) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Find the parent FuncOp before tiling. If tiling succeeds, the op will be
    // erased.
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    SmallVector<Value, 4> tensorResults;
    if (!funcOp ||
        failed(Base::matchAndRewriteBase(op, rewriter, tensorResults)) ||
        !tensorResults.empty() ||
        (funcOp.getAttr(getNumWorkgroupsFnAttrName()) &&
         failed(utils::createNumWorkgroupsFromResultShape(
             rewriter, cast<linalg::LinalgOp>(op), funcOp,
             getNumWorkgroupsFnAttrName(),
             cpuKernelDispatch.getTileSizes(op))))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
  CPUKernelDispatch cpuKernelDispatch;
};

template <typename LinalgOpTy>
struct TileAndFuseToCPUThreads
    : public linalg::LinalgTileAndFusePattern<LinalgOpTy> {
  using Base = linalg::LinalgTileAndFusePattern<LinalgOpTy>;
  TileAndFuseToCPUThreads(MLIRContext *context,
                          const linalg::LinalgDependenceGraph &dependenceGraph,
                          const CPUKernelDispatch &cpuKernelDispatch,
                          linalg::LinalgTilingOptions tilingOptions,
                          linalg::LinalgMarker marker,
                          PatternBenefit benefit = 1)
      : Base(context, dependenceGraph, tilingOptions,
             linalg::LinalgFusionOptions().setIndicesToFuse({2}), marker,
             marker,
             linalg::LinalgMarker(ArrayRef<Identifier>(),
                                  Identifier::get(getDeleteMarker(), context)),
             benefit),
        dependenceGraph(dependenceGraph),
        cpuKernelDispatch(cpuKernelDispatch) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    if (!funcOp || !dependenceGraph.hasDependentOperations(linalgOp) ||
        failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(utils::createNumWorkgroupsFromResultShape(
            rewriter, cast<linalg::LinalgOp>(op), funcOp,
            getNumWorkgroupsFnAttrName().str(),
            cpuKernelDispatch.getTileSizes(op)))) {
      return failure();
    }
    return success();
  }

  const linalg::LinalgDependenceGraph &dependenceGraph;
  CPUKernelDispatch cpuKernelDispatch;
};

}  // namespace

void LinalgTileAndDistributePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        Type indexType = builder.getIndexType();
        auto numParallelDims = parallelLoopRanges.size();
        SmallVector<linalg::ProcInfo, 2> procInfo(numParallelDims);
        for (int dim = 0; dim < numParallelDims; ++dim) {
          std::array<StringRef, 3> dimAttr{"x", "y", "z"};
          StringAttr attr =
              builder.getStringAttr(dimAttr[std::min<unsigned>(dim, 3)]);
          procInfo[numParallelDims - dim - 1] = {
              builder.create<IREE::WorkgroupIdOp>(loc, indexType, attr),
              builder.create<IREE::WorkgroupSizeOp>(loc, indexType, attr)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters}};

  CPUKernelDispatch cpuKernelDispatch;

  // Function to compute first level tiling values.
  std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>
      getOuterTileSizeFn =
          [&cpuKernelDispatch](OpBuilder &builder,
                               Operation *operation) -> SmallVector<Value, 4> {
    auto tileSizes = cpuKernelDispatch.getTileSizes(operation);
    if (tileSizes.empty()) return {};
    SmallVector<Value, 4> tileSizesVal;
    tileSizesVal.reserve(tileSizes.size());
    for (auto val : tileSizes) {
      tileSizesVal.push_back(
          builder.create<ConstantIndexOp>(operation->getLoc(), val));
    }
    return tileSizesVal;
  };

  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    // Compute the Linalg Dependence Graph.
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph =
        linalg::LinalgDependenceGraph::buildDependenceGraph(aliases, funcOp);

    OwningRewritePatternList patterns;

    auto linalgTilingOptions =
        linalg::LinalgTilingOptions()
            .setDistributionOptions(workgroupDistributionOptions)
            .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);
    tileSizes.empty()
        ? linalgTilingOptions.setTileSizeComputationFunction(getOuterTileSizeFn)
        : linalgTilingOptions.setTileSizes(ArrayRef<int64_t>(tileSizes));
    patterns.insert<TileAndFuseToCPUThreads<linalg::MatmulOp>,
                    TileToCPUThreads<linalg::MatmulOp>>(
        context, dependenceGraph, cpuKernelDispatch, linalgTilingOptions,
        linalg::LinalgMarker(ArrayRef<Identifier>(),
                             Identifier::get(getWorkgroupMarker(), context)));

    // Tile and distribute to CPU threads.
    applyPatternsAndFoldGreedily(funcOp, patterns);

    // Apply canonicalization patterns.
    OwningRewritePatternList canonicalizationPatterns;
    canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
    AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                               context);
    AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
    SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns, context);

    applyPatternsAndFoldGreedily(funcOp, canonicalizationPatterns);

    // Delete the ops that are marked for deletion.
    funcOp.walk([](linalg::LinalgOp linalgOp) {
      if (hasMarker(linalgOp.getOperation(), getDeleteMarker()))
        linalgOp.getOperation()->erase();
    });
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createLinalgTileAndDistributePass() {
  return std::make_unique<LinalgTileAndDistributePass>();
}

static PassRegistration<LinalgTileAndDistributePass> pass(
    "iree-codegen-llvm-linalg-tile-and-distribute",
    "Tile and distribute Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndDistributePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
