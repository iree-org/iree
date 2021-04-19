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
#include "mlir/Dialect/Affine/Utils.h"
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

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir {
namespace iree_compiler {

/// Apply canonicalizations related to tiling to make promotion/vectorization
/// easier.
void applyCanonicalizationPatternsForTiling(MLIRContext *context,
                                            Operation *op) {
  OwningRewritePatternList canonicalizationPatterns(context);
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
  OwningRewritePatternList patterns(funcOp.getContext());
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

static bool isDivisible(Value v, int64_t dividend);

/// Return true if we can prove that affineMinOp result is positive and
/// divisible by the given |dividend|. This is true if all the the results of
/// the associated affine map are positive and divisible by |dividend|.
/// This speciically look for the following pattern:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
static bool affineMinOpDivisible(AffineMinOp minOp, int64_t dividend) {
  if (!minOp.getSymbolOperands().empty() ||
      minOp.getAffineMap().getNumResults() != 2)
    return {};
  Value iv;
  Value ub;
  Value lb;
  Value step;
  // Check if any of the dimensions is a ForOp or ParallelOp induction variable.
  for (auto dim : minOp.getDimOperands()) {
    auto ivArg = dim.dyn_cast<BlockArgument>();
    if (!ivArg) continue;
    Operation *containingOp = ivArg.getOwner()->getParentOp();
    auto forOp = dyn_cast_or_null<scf::ForOp>(containingOp);
    if (forOp && forOp.getInductionVar() == dim) {
      iv = dim;
      ub = forOp.upperBound();
      lb = forOp.lowerBound();
      step = forOp.step();
      break;
    }
    auto parallelOp = dyn_cast_or_null<scf::ParallelOp>(containingOp);
    if (!parallelOp) continue;
    for (auto inductionVar : llvm::enumerate(parallelOp.getInductionVars())) {
      if (inductionVar.value() == dim) {
        iv = dim;
        ub = parallelOp.upperBound()[inductionVar.index()];
        lb = parallelOp.lowerBound()[inductionVar.index()];
        step = parallelOp.step()[inductionVar.index()];
        break;
      }
    }
    if (iv) break;
  }
  if (!iv) return false;
  // Calculate the affine map representing `%ub - %iv`.
  AffineExpr ivDim;
  AffineExpr ubDim;
  for (auto dim : llvm::enumerate(minOp.getDimOperands())) {
    if (dim.value() == iv)
      ivDim = getAffineDimExpr(dim.index(), minOp.getContext());
    else if (dim.value() == ub)
      ubDim = getAffineDimExpr(dim.index(), minOp.getContext());
    else
      return false;
  }

  if (!ubDim) {
    if (auto cstUb = ub.getDefiningOp<ConstantIndexOp>())
      ubDim = getAffineConstantExpr(cstUb.getValue(), minOp.getContext());
    else
      return false;
  }
  AffineExpr diffExp = ubDim - ivDim;
  // Check that all the affine map results are either constant divisible by
  // `dividend` or equal to `%ub - %iv`.
  for (AffineExpr result : minOp.getAffineMap().getResults()) {
    if (auto cst = result.dyn_cast<AffineConstantExpr>()) {
      if (cst.getValue() <= 0 || cst.getValue() % dividend != 0) return false;
    } else {
      if (diffExp != result) return false;
    }
  }
  // Now check that for every value of the induction variable `%ub - %iv` is
  // divisible by `dividend`. It is true if the lower bounder, the upper bound
  // and the step are all divisible by `dividend`.
  std::array<Value, 3> loopOperands = {lb, step, ub};
  return llvm::all_of(loopOperands,
                      [dividend](Value v) { return isDivisible(v, dividend); });
}

/// Return true if we can prove that the value |v| is always divisible by the
/// constant |dividend|. Return false otherwise.
static bool isDivisible(Value v, int64_t dividend) {
  MLIRContext *ctx = v.getContext();
  // Create an expression (d0) -> (d0 % n) and try to simplify it.
  AffineExpr mod = getAffineDimExpr(0, ctx) % dividend;
  AffineMap modMap = AffineMap::get(1, 0, {mod}, ctx);
  SmallVector<Value> ops(1, v);
  fullyComposeAffineMapAndOperands(&modMap, &ops);
  canonicalizeMapAndOperands(&modMap, &ops);
  modMap = simplifyAffineMap(modMap);
  auto cst = modMap.getResult(0).dyn_cast<AffineConstantExpr>();
  if (cst) return (cst.getValue() == 0);
  // If the map doesn't fold to 0 but simplifies to (d0 %n) with d0 an
  // affine.min, check if all the results of the affine.min's map are divisible
  // by `dividend`.
  if (modMap.getResult(0) != mod) return false;
  assert(ops.size() == 1);
  auto minOp = ops[0].getDefiningOp<AffineMinOp>();
  return (minOp && affineMinOpDivisible(minOp, dividend));
}

/// Try to fold a affine.min op by matching the following form:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
/// With N a compile time constant. This operations can be replace by
/// `%cN = constant N : index` if we can prove that %lb, %step and %ub are
/// divisible by N.
static Optional<int64_t> foldAffineMin(AffineMinOp minOp) {
  AffineMap map = minOp.getAffineMap();
  int64_t constantResult = 0;
  for (AffineExpr result : map.getResults()) {
    if (auto cst = result.dyn_cast<AffineConstantExpr>())
      constantResult = cst.getValue();
  }
  if (constantResult == 0) return {};
  // If afine.min map's results are all positive and divisible by
  // `constantResult` then it can be replaced by `constantResult`.
  if (affineMinOpDivisible(minOp, constantResult)) return constantResult;
  return {};
}

/// Compose map with apply affine ops and try to simplify it.
static void combineAndSimplifyMap(AffineMap &map, SmallVectorImpl<Value> &dims,
                                  SmallVectorImpl<Value> &symbols) {
  SmallVector<Value, 4> operands(dims.begin(), dims.end());
  operands.append(symbols.begin(), symbols.end());
  // Pull in affine.apply operations and compose them fully into the
  // result.
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  // Assign the results.
  dims.assign(operands.begin(), operands.begin() + map.getNumDims());
  symbols.assign(operands.begin() + map.getNumDims(), operands.end());
}

/// Replace dimensions and symbols with known range in the map expression.
// TODO: Use core function once the interface using a lambda lands.
static AffineMap substituteMin(AffineMap map, SmallVectorImpl<Value> &dims,
                               SmallVectorImpl<Value> &symbols,
                               GetMinMaxExprFn getMinMaxExpr) {
  combineAndSimplifyMap(map, dims, symbols);
  auto exprs = llvm::to_vector<4>(map.getResults());
  for (AffineExpr &expr : exprs) {
    bool substituted = true;
    while (substituted) {
      substituted = false;
      for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
        Value dim = dims[dimIdx];
        auto minMax = getMinMaxExpr(dim, dims, symbols);
        if (!minMax) continue;
        AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
        LLVM_DEBUG(DBGS() << "Subst: " << dim << " @ " << dimExpr << "\n");
        LLVM_DEBUG(DBGS() << "Before: " << expr << "\n");
        // Substitute occurrences of `dimExpr` by either the min expression or
        // the max expression depending on whether the value is used with a
        // positive or negative  coefficient.
        AffineExpr substitutedExpr =
            substWithMin(expr, dimExpr, minMax->first, minMax->second);
        LLVM_DEBUG(DBGS() << "After: " << substitutedExpr << "\n");
        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
      // Substitute symbols
      for (unsigned symIdx = 0; symIdx < symbols.size(); ++symIdx) {
        Value sym = symbols[symIdx];
        auto minMax = getMinMaxExpr(sym, dims, symbols);
        if (!minMax) continue;
        AffineExpr symExpr = getAffineSymbolExpr(symIdx, expr.getContext());
        LLVM_DEBUG(DBGS() << "Subst: " << sym << " @ " << symExpr << "\n");
        LLVM_DEBUG(DBGS() << "Before: " << expr << "\n");
        AffineExpr substitutedExpr =
            substWithMin(expr, symExpr, minMax->first, minMax->second);
        LLVM_DEBUG(DBGS() << "After: " << substitutedExpr << "\n");
        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
    }

    map = AffineMap::get(dims.size(), symbols.size(), exprs,
                         exprs.front().getContext());
    // Cleanup and simplify the results.
    // This needs to happen outside of the loop iterating on dims.size() since
    // it modifies dims.
    combineAndSimplifyMap(map, dims, symbols);
    // Assign the results.
    exprs.assign(map.getResults().begin(), map.getResults().end());
    LLVM_DEBUG(DBGS() << "Map simplified: " << map << "\n");
  }

  assert(!exprs.empty() && "Unexpected empty exprs");
  return AffineMap::get(dims.size(), symbols.size(), exprs, map.getContext());
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(scf::ForOp op, GetMinMaxExprFn getMinMax) {
  // Calculate the minimum value of ub - lb. If it is strictly positive it
  // means the loop will always run at least once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.lowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.upperBound());
  AffineExpr iterZero = ub - lb;
  auto map = AffineMap::get(dims.size(), 0, iterZero);
  AffineMap simplifiedMap = substituteMin(map, dims, symbols, getMinMax);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() > 0) return true;
  }
  return false;
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(scf::ForOp op, GetMinMaxExprFn getMinMax) {
  // Calculate the minimum of lb + step - ub. If it is positive it means the
  // loop never run more than once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.lowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.upperBound());
  AffineExpr step = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.step());
  AffineExpr iterOne = lb + step - ub;
  auto map = AffineMap::get(dims.size(), 0, iterOne);

  AffineMap simplifiedMap = substituteMin(map, dims, symbols, getMinMax);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() >= 0) return true;
  }
  return false;
}

namespace {

struct AffineMinDistributedSCFCanonicalizationPattern
    : public mlir::OpRewritePattern<mlir::AffineMinOp> {
  using OpRewritePattern<mlir::AffineMinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::AffineMinOp minOp, mlir::PatternRewriter &rewriter) const override {
    Optional<int64_t> cst = foldAffineMin(minOp);
    if (!cst) return failure();
    rewriter.replaceOpWithNewOp<ConstantOp>(minOp, rewriter.getIndexAttr(*cst));
    return failure();
  }
};

/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {
  SimplifyTrivialLoops(MLIRContext *context, GetMinMaxExprFn getMinMax)
      : OpRewritePattern<scf::ForOp>(context, 1), getMinMax(getMinMax) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle the case where we know that the loop doesn't run more than
    // once but the loop may not run at least once by replace the `loop` with an
    // `if`.
    if (!(alwaysRunsFirstIteration(op, getMinMax) &&
          neverRunsSecondIteration(op, getMinMax)))
      return failure();

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.lowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());
    replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
    return success();
  }

 private:
  GetMinMaxExprFn getMinMax;
};

}  // namespace

void populateAffineMinSCFCanonicalizationPattern(RewritePatternSet &patterns) {
  patterns.add<AffineMinDistributedSCFCanonicalizationPattern>(
      patterns.getContext());
}

void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              GetMinMaxExprFn getMinMaxFn) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext(), getMinMaxFn);
}
/// Pass to be able to test AffineMinDistributedSCFCanonicalizationPattern
/// individually.
struct AffineMinDistributedSCFCanonicalizationPass
    : public PassWrapper<AffineMinDistributedSCFCanonicalizationPass,
                         FunctionPass> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    RewritePatternSet foldPattern(&getContext());
    populateAffineMinSCFCanonicalizationPattern(foldPattern);
    FrozenRewritePatternSet frozenPatterns(std::move(foldPattern));

    // Explicitly walk and apply the pattern locally to avoid more general
    // folding on the rest of the IR.
    funcOp.walk([&frozenPatterns](AffineMinOp minOp) {
      (void)applyOpPatternsAndFold(minOp, frozenPatterns);
    });
  }
};

static PassRegistration<AffineMinDistributedSCFCanonicalizationPass> pass(
    "iree-codegen-affinemin-scf-canonicalization",
    "Pass to run pass cleaning up affineMinOp after tiling and distribute.",
    [] {
      return std::make_unique<AffineMinDistributedSCFCanonicalizationPass>();
    });

}  // namespace iree_compiler
}  // namespace mlir
