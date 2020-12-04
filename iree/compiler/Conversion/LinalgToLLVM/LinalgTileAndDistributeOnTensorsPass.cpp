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

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-tile-and-distribute-on-tensors"

namespace mlir {
namespace iree_compiler {

struct LinalgTileAndDistributeOnTensorsPass
    : public PassWrapper<LinalgTileAndDistributeOnTensorsPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREEDialect, AffineDialect,
                    scf::SCFDialect>();
  }
  LinalgTileAndDistributeOnTensorsPass() = default;
  LinalgTileAndDistributeOnTensorsPass(
      const LinalgTileAndDistributeOnTensorsPass &pass) {}
  void runOnOperation() override;

 private:
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Set tile sizes to use"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

static std::pair<Value, Value> buildWorkgroupOpPair(OpBuilder &b,
                                                    StringRef dim) {
  Type indexType = b.getIndexType();
  StringAttr attr = b.getStringAttr(dim);
  return {b.create<IREE::WorkgroupIdOp>(b.getInsertionPoint()->getLoc(),
                                        indexType, attr),
          b.create<IREE::WorkgroupSizeOp>(b.getInsertionPoint()->getLoc(),
                                          indexType, attr)};
}

// Rewrite pattern to ensure only ops with tensor semantics are tiled.
struct TileAndDistributeOnTensorsPattern
    : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileAndDistributeOnTensorsPattern(linalg::LinalgTilingOptions options,
                                    linalg::LinalgMarker marker,
                                    PatternBenefit benefit = 1)
      : Base(options, marker, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
    SmallVector<Value, 4> tensorResults;
    if (failed(Base::matchAndRewriteBase(op, rewriter, tensorResults)))
      return failure();
    // TODO: Wrap in sequentialized SPMD loops.
    rewriter.replaceOp(op, tensorResults);
    return success();
  }
};

void LinalgTileAndDistributeOnTensorsPass::runOnOperation() {
  if (tileSizes.empty()) return;
  ModuleOp module = getOperation();
  MLIRContext *context = module->getContext();

  // Distribution strategy along at most 3 dimensions with WorkgroupIdOp in
  // range [0, WorkgroupSizeOp).
  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        // TODO: drop magic names.
        std::array<StringRef, 3> dimStrs{"x", "y", "z"};
        auto numParallelDims = parallelLoopRanges.size();
        SmallVector<linalg::ProcInfo, 2> procInfo(numParallelDims);
        for (unsigned dim = 0; dim < std::min(numParallelDims, 3ul); ++dim) {
          auto p = buildWorkgroupOpPair(builder, dimStrs[dim]);
          procInfo[dim] = {p.first, p.second};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    // TODO: maybe activate when put in a real pipeline.
    // if (!isEntryPoint(funcOp)) continue;

    OwningRewritePatternList patterns;
    auto linalgTilingOptions =
        linalg::LinalgTilingOptions()
            .setDistributionOptions(workgroupDistributionOptions)
            .setLoopType(linalg::LinalgTilingLoopType::Loops)
            .setTileSizes(ArrayRef<int64_t>(tileSizes));
    assert(linalgTilingOptions.distribution.hasValue());

    // In the future, derive from LinalgTilingPattern to create sequentialized
    // SPMD loops.
    patterns.insert<TileAndDistributeOnTensorsPattern>(
        linalgTilingOptions,
        linalg::LinalgMarker(ArrayRef<Identifier>(),
                             Identifier::get(getWorkgroupMarker(), context)));
    // Add canonicalization patterns.
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns, context);
    patterns.insert<AffineMinCanonicalizationPattern>(context);
    applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createLinalgTileAndDistributeOnTensorsPass() {
  return std::make_unique<LinalgTileAndDistributeOnTensorsPass>();
}

static PassRegistration<LinalgTileAndDistributeOnTensorsPass> pass(
    "iree-codegen-llvm-linalg-tile-and-distribute-on-tensors",
    "Tile and distribute Linalg operations on tensors",
    [] { return std::make_unique<LinalgTileAndDistributeOnTensorsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
