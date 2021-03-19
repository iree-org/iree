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

//===- Transforms.cpp - Transformations common to all backends ------------===//
//
// Implements transformations that are common to all backends.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/Transforms.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse"

namespace mlir {
namespace iree_compiler {

/// Apply canonicalizations related to tiling to make promotion/vectorization
/// easier.
void applyCanonicalizationPatternsForTiling(MLIRContext *context,
                                            Operation *op) {
  OwningRewritePatternList canonicalizationPatterns;
  canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
  scf::ForOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  memref::SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                 context);
  (void)applyPatternsAndFoldGreedily(op, std::move(canonicalizationPatterns));
}

//===----------------------------------------------------------------------===//
// Helper functions for tile and fuse.
//===----------------------------------------------------------------------===//

/// Promotes views used to chain Linalg ops in `fusedOps`
/// into buffers using the allocation callback in `options`.
///
/// Once fused the fused views that are due to a RAW dependence can be promoted
/// to workgroup memory. This will make the intermediate storage dead.
static LogicalResult promoteFusedViews(OpBuilder &builder,
                                       ArrayRef<linalg::LinalgOp> fusedOps,
                                       const TileAndFuseOptions &options) {
  linalg::Aliases aliases;
  linalg::LinalgDependenceGraph dependenceGraph(aliases, fusedOps);
  auto fusableDependences =
      linalg::findAllFusableDependences(fusedOps, dependenceGraph);

  DenseSet<Value> promotedViews;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(*fusedOps.begin());

  // Scan the list of ops in reverse order. The fusion is performed by creating
  // a tiled version of the ops within the tiled loops of the last operation in
  // the sequence, and then proceeding up the sequence.
  for (linalg::LinalgOp op : llvm::reverse(fusedOps)) {
    auto dependences = fusableDependences.lookup(op);
    if (dependences.empty()) continue;
    if (!llvm::hasSingleElement(dependences)) {
      return op.emitError(
          "unable to promote ops with multiple fusable dependences");
    }
    auto dependence = dependences.front();
    Optional<unsigned> producerIdx = dependence.getDependentOpViewOperandNum();
    if (!producerIdx) {
      return op.emitError(
          "expected dependent view in producer to be an operand");
    }
    linalg::LinalgOp consumer =
        cast<linalg::LinalgOp>(dependence.getIndexingOp());
    unsigned consumerIdx = dependence.getIndexingOpViewOperandNum().getValue();
    Value consumerView = dependence.getIndexingValue();
    Value promotedView = nullptr;

    // If the view is already promoted, reuse that. The assumption is that the
    // view matches already.
    if (promotedViews.count(consumerView)) {
      promotedView = consumerView;
    } else if (dependence.dependenceType ==
               linalg::LinalgDependenceGraph::RAW) {
      memref::SubViewOp promotedViewProducer =
          op.getShapedOperand(*producerIdx).getDefiningOp<memref::SubViewOp>();
      assert(promotedViewProducer &&
             "expected producer to be a subview op as well");
      Optional<linalg::PromotionInfo> promotionInfo =
          linalg::promoteSubviewAsNewBuffer(
              builder, op.getLoc(), promotedViewProducer, options.allocationFn);
      if (!promotionInfo) {
        return op.emitError("unable to promote RAW dependence");
      }
      promotedView = promotionInfo->partialLocalView;
      consumer.getOperation()->setOperand(consumerIdx, promotedView);
      promotedViews.insert(promotedView);
    }
    if (!promotedView) continue;
    op.getOperation()->setOperand(*producerIdx, promotedView);
  }
  return success();
}

/// Tile+Fuse only tiles the loops that can be fused. Tile any of the unfused
/// loops in the operation based on the configuration.
static linalg::LinalgOp tileUnfusedLoops(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const std::set<unsigned> &fusedLoopDims, ArrayRef<int64_t> tileSizesRef) {
  SmallVector<int64_t, 4> tileSizes = llvm::to_vector<4>(tileSizesRef);
  tileSizes.resize(linalgOp.getNumLoops(), 0);
  // Linalg uses tile size = 0 for a loop to indicate not tiling that loop. Set
  // the fused loops to be untiled (since they are already tiled during fusion).
  for (unsigned loopNum : fusedLoopDims) {
    tileSizes[loopNum] = 0;
  }
  if (llvm::all_of(tileSizes, [](int64_t v) { return !v; })) return linalgOp;
  Optional<linalg::TiledLinalgOp> tiledOp = tileLinalgOp(
      builder, linalgOp,
      linalg::LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(
          linalg::LinalgTilingLoopType::ParallelLoops));
  if (!tiledOp) return nullptr;
  linalgOp.erase();
  return tiledOp->op;
}

/// Tiles the last operation in `fusableOps` and fuses all other operations with
/// it by creating tiled versions of each within the generated inter-tile loops.
static Optional<linalg::TiledAndFusedLinalgOps> tileAndFuseLinalgOps(
    OpBuilder &builder, FuncOp funcOp, ArrayRef<linalg::LinalgOp> fusableOps,
    const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<int64_t> tileSizes, const TileAndFuseOptions &options) {
  // Get the tile sizes to use from the last fusable op and the tile+fuse all
  // ops.
  linalg::LinalgTilingOptions tilingOptions;
  tilingOptions.setDistributionOptions(options.distributionOptions)
      .setTileSizes(tileSizes)
      .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);

  Optional<linalg::TiledAndFusedLinalgOps> tiledAndFusedOps = llvm::None;
  if (fusableOps.size() == 1) {
    linalg::LinalgOp linalgOp = fusableOps.front();
    Optional<linalg::TiledLinalgOp> tiledOp =
        tileLinalgOp(builder, linalgOp, tilingOptions);
    if (!tiledOp) {
      linalgOp.emitError("unable to tile operation");
      return llvm::None;
    }
    tiledAndFusedOps = linalg::TiledAndFusedLinalgOps{tiledOp->op, {}, {}, {}};
    auto seq = llvm::seq<unsigned>(0, tileSizes.size());
    tiledAndFusedOps->fusedLoopDims.insert(seq.begin(), seq.end());
    tiledAndFusedOps->fusedLoops.assign(tiledOp->loops.begin(),
                                        tiledOp->loops.end());
  } else {
    tiledAndFusedOps = tileAndFuseLinalgOps(builder, fusableOps,
                                            dependenceGraph, tilingOptions);
  }
  if (!tiledAndFusedOps) {
    funcOp.emitError("tile and fuse of linalg operations failed");
    return llvm::None;
  }

  // TODO(GH-4901) Only support static shapes here. This is on deprecation
  // path. So doing this till this is needed. Remove this part when switched
  // over to linalg on tensors.
  linalg::LinalgOp rootOp = fusableOps.back();
  Optional<SmallVector<int64_t, 4>> staticLoopRange =
      rootOp.getStaticLoopRanges();
  if (!staticLoopRange ||
      llvm::any_of(staticLoopRange.getValue(),
                   [](int64_t d) { return d == ShapedType::kDynamicSize; })) {
    rootOp.emitError("failed to find statlc loop bounds");
    return llvm::None;
  }
  // Extract static tile sizes.
  WorkgroupCountRegionBuilder regionBuilder =
      [&tileSizes, &tiledAndFusedOps, &staticLoopRange](
          OpBuilder &b, Location loc,
          std::array<Value, 3> workload) -> std::array<Value, 3> {
    Value one = b.create<ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> workgroupCounts;
    for (auto size : enumerate(tileSizes)) {
      if (!size.value() ||
          !tiledAndFusedOps->fusedLoopDims.count(size.index())) {
        continue;
      }
      Value extent =
          b.create<ConstantIndexOp>(loc, (*staticLoopRange)[size.index()]);
      auto map =
          AffineMap::get(0, 1, b.getAffineSymbolExpr(0).ceilDiv(size.value()));
      Value workgroupCount = linalg::applyMapToValues(b, loc, map, extent)[0];
      workgroupCounts.push_back(workgroupCount);
    }
    if (workgroupCounts.size() > 3) {
      workgroupCounts.resize(3);
    }
    workgroupCounts = llvm::to_vector<4>(llvm::reverse(workgroupCounts));
    workgroupCounts.resize(3, one);
    return {workgroupCounts[0], workgroupCounts[1], workgroupCounts[2]};
  };
  if (failed(defineWorkgroupCountRegion(builder, funcOp, regionBuilder))) {
    return llvm::None;
  }

  // Delete all the original operations.
  for (auto linalgOp : fusableOps) linalgOp.erase();

  // Add workgroup markers to all the tiled and fused operations.
  for (auto fusedProducer : tiledAndFusedOps->fusedProducers) {
    setMarker(fusedProducer, getWorkgroupMarker());
  }
  setMarker(tiledAndFusedOps->op, getWorkgroupMarker());

  return tiledAndFusedOps;
}

LogicalResult getLinalgOps(FuncOp funcOp,
                           SmallVectorImpl<linalg::LinalgOp> &linalgOps,
                           SmallVectorImpl<Operation *> &tiledLoops) {
  Region &region = funcOp.body();
  if (!llvm::hasSingleElement(region)) {
    return funcOp.emitError("unable dispatch function with multiple blocks");
  }
  Block *body = &region.front();
  auto forOps = body->getOps<scf::ForOp>();
  while (!forOps.empty()) {
    if (!llvm::hasSingleElement(forOps)) return failure();
    scf::ForOp forOp = *(forOps.begin());
    tiledLoops.push_back(forOp.getOperation());
    body = forOp.getBody();
    forOps = body->getOps<scf::ForOp>();
  }
  linalgOps = llvm::to_vector<4>(body->getOps<linalg::LinalgOp>());

  // Propagate markers to all ops. If one of the ops has a marker all ops in
  // this loop need to have marker since body of the loop maps to a workgroup.
  // TODO(ravishankarm): Temporary WAR till a better story w.r.t markers is
  // figured out.
  Optional<StringRef> marker = llvm::None;
  for (auto op : linalgOps) {
    if (hasMarker(op)) {
      assert(!marker || marker.getValue() == getMarkerOrNull(op) &&
                            "expected all markers within op to be the same");
      marker = getMarkerOrNull(op);
    }
  }
  if (marker.hasValue()) {
    for (auto op : linalgOps) {
      setMarker(op, marker.getValue());
    }
  }
  return success();
}

namespace {
static size_t kMaxHALDimensions = 3;

/// Sets the hal.interace.workgroup.size operation to the constant value passed
/// in as `workloadPerWorkgroup`. The number of entries in
/// `workloadPerWorkgroup` is at least as much as the dimensionality of the
/// workgroup. It is assumed that the inner-most loop is mapped to the fastest
/// varying dimension in flow.dispatch.workgroup_size.
class SetWorkgroupSizePattern
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  SetWorkgroupSizePattern(MLIRContext *context,
                          ArrayRef<int64_t> workloadPerWorkgroupRef,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadPerWorkgroup(llvm::to_vector<4>(
            workloadPerWorkgroupRef.size() > kMaxHALDimensions
                ? workloadPerWorkgroupRef.take_front(kMaxHALDimensions)
                : workloadPerWorkgroupRef)) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupSizeOp workgroupSizeOp,
      PatternRewriter &rewriter) const override {
    int64_t dim = workgroupSizeOp.dimension().getSExtValue();
    if (dim >= workloadPerWorkgroup.size()) {
      return workgroupSizeOp.emitRemark(
          "expected at least as many static tile sizes as the workgroup "
          "dimensionality");
    }
    rewriter.replaceOpWithNewOp<ConstantIndexOp>(workgroupSizeOp,
                                                 workloadPerWorkgroup[dim]);
    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadPerWorkgroup;
};
}  // namespace

LogicalResult defineWorkgroupCountRegion(
    OpBuilder &builder, FuncOp funcOp,
    WorkgroupCountRegionBuilder regionBuilder) {
  IREE::HAL::ExecutableEntryPointOp entryPointOp = getEntryPoint(funcOp);
  if (!entryPointOp) {
    return funcOp.emitOpError("unable to find corresponding entry point op");
  }
  Location loc = entryPointOp.getLoc();

  OpBuilder::InsertionGuard guard(builder);
  // Create the cloned operation but with a single region.
  builder.setInsertionPoint(entryPointOp);
  auto clonedOp = builder.create<IREE::HAL::ExecutableEntryPointOp>(
      loc, entryPointOp.sym_nameAttr(), entryPointOp.ordinalAttr(),
      entryPointOp.interfaceAttr(), entryPointOp.signatureAttr(),
      entryPointOp.workgroup_sizeAttr(), 1);
  Region *region = clonedOp.getBody();
  Block *entryBlock = builder.createBlock(region);
  // Add 3 index arguments for the workload.
  auto indexType = builder.getIndexType();
  std::array<Value, 3> workload = {entryBlock->addArgument(indexType),
                                   entryBlock->addArgument(indexType),
                                   entryBlock->addArgument(indexType)};
  std::array<Value, 3> workgroupCount = regionBuilder(builder, loc, workload);
  builder.create<IREE::HAL::ReturnOp>(loc, workgroupCount);
  entryPointOp.erase();
  return success();
}

LogicalResult materializeStaticLaunchInformation(
    FuncOp funcOp, ArrayRef<int64_t> workloadPerWorkgroup) {
  OwningRewritePatternList patterns;
  patterns.insert<SetWorkgroupSizePattern>(funcOp.getContext(),
                                           workloadPerWorkgroup);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }
  assert(workloadPerWorkgroup.size() <= 3 &&
         "workloadPerWorkgroup size greater than 3 not handled");
  WorkgroupCountRegionBuilder regionBuilder =
      [&workloadPerWorkgroup](
          OpBuilder &b, Location loc,
          std::array<Value, 3> workload) -> std::array<Value, 3> {
    Value one = b.create<ConstantIndexOp>(loc, 1);
    std::array<Value, 3> returnValues = {one, one, one};
    for (auto ts : llvm::enumerate(workloadPerWorkgroup)) {
      returnValues[ts.index()] = linalg::applyMapToValues(
          b, loc,
          AffineMap::get(0, 1, b.getAffineSymbolExpr(0).ceilDiv(ts.value())),
          workload[ts.index()])[0];
    }
    return returnValues;
  };
  OpBuilder builder(funcOp.getContext());
  return defineWorkgroupCountRegion(builder, funcOp, regionBuilder);
}

LogicalResult tileAndFuseLinalgBufferOps(
    FuncOp funcOp, ArrayRef<linalg::LinalgOp> linalgOps,
    const linalg::LinalgDependenceGraph &dependenceGraph,
    const LaunchConfig &launchConfig, const TileAndFuseOptions &options) {
  // Collect all operations that are to be tiled-and-fused.
  MLIRContext *context = funcOp.getContext();
  SmallVector<linalg::LinalgOp, 4> fusableOps;
  for (Operation *operation : linalgOps) {
    if (!launchConfig.hasTileSizes(operation)) continue;
    fusableOps.push_back(cast<linalg::LinalgOp>(operation));
  }
  if (fusableOps.empty()) return success();

  OpBuilder builder(context);
  ArrayRef<int64_t> tileSizes = launchConfig.getTileSizes(fusableOps.back(), 0);
  Optional<linalg::TiledAndFusedLinalgOps> tiledAndFusedOps =
      tileAndFuseLinalgOps(builder, funcOp, fusableOps, dependenceGraph,
                           tileSizes, options);
  if (!tiledAndFusedOps) {
    return funcOp.emitError("failed to tile and fuse operations");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Fusion on buffers ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  applyCanonicalizationPatternsForTiling(context, funcOp);

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Canonicalization ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  if (options.allocationFn) {
    SmallVector<linalg::LinalgOp, 4> promoteFusedViewOps =
        llvm::to_vector<4>(tiledAndFusedOps->fusedProducers);
    promoteFusedViewOps.push_back(tiledAndFusedOps->op);

    if (failed(promoteFusedViews(builder, promoteFusedViewOps, options))) {
      return failure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Promotion ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Tile the unfused loops. Set the tile sizes for the fused loops to be zero
  // to avoid tiling them again.
  for (linalg::LinalgOp &fusedOp : tiledAndFusedOps->fusedProducers) {
    ArrayRef<int64_t> fusedOpTileSizes = launchConfig.getTileSizes(fusedOp, 0);
    linalg::LinalgOp tiledOp = tileUnfusedLoops(
        builder, fusedOp, tiledAndFusedOps->fusedLoopDims, fusedOpTileSizes);
    if (!tiledOp) {
      return fusedOp.emitError("unable to tile unfused loops");
    }
  }
  linalg::LinalgOp tiledOp =
      tileUnfusedLoops(builder, tiledAndFusedOps->op,
                       tiledAndFusedOps->fusedLoopDims, tileSizes);
  if (!tiledOp) {
    return tiledAndFusedOps->op.emitError("unable to tile unfused loops");
  }

  applyCanonicalizationPatternsForTiling(context, funcOp);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
