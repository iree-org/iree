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

//===- LinalgTilingOnBuffers.cpp - Tile and fuse Linalg on Buffers --------===//
//
// Implements a pass to tile and fuse linalg operations on buffers.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if the linalg op has padding attribute, and that it has
/// non-zero entries.
template <typename OpTy>
static bool hasPadding(OpTy op) {
  Optional<DenseIntElementsAttr> padding = op.padding();
  if (!padding) return false;
  return llvm::any_of(padding.getValue(),
                      [](APInt v) -> bool { return !v.isNullValue(); });
}

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, OperationPass<ModuleOp>> {
  LinalgTileAndFusePass() = default;
  LinalgTileAndFusePass(const SPIRVCodegenOptions &passedOptions) {
    options = passedOptions;
  }
  LinalgTileAndFusePass(const LinalgTileAndFusePass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    scf::SCFDialect, ShapeDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;

 private:
  SPIRVCodegenOptions options;

  // TODO: Find a common place to put these options. They are defined three
  // times, once here, once for the pass pipeline and once for the binary.
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Set tile sizes to use"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  ListOption<int64_t> workgroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Number of workgroups to dispatch for the SPIR-V module; at most "
          "three integers standarding for the x, y, and z dimension; "
          "additional arguments will be ignored (used only for testing)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  Option<bool> useWorkgroupMemory{
      *this, "use-workgroup-memory",
      llvm::cl::desc(
          "Enable use of workgroup memory in SPIR-V code generation pipeline"),
      llvm::cl::init(false)};

  Option<bool> useVectorization{
      *this, "use-vectorization",
      llvm::cl::desc(
          "Enable use of vectorization in SPIR-V code generation pipeline"),
      llvm::cl::init(false)};
};
}  // namespace

/// Apply canonicalizations related to tiling to make promotion/vectorization
/// easier.
static LogicalResult applyCanonicalizationPatterns(MLIRContext *context,
                                                   Operation *op) {
  OwningRewritePatternList canonicalizationPatterns;
  canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
  AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  return applyPatternsAndFoldGreedily(op, canonicalizationPatterns);
}

//===----------------------------------------------------------------------===//
// Patterns to tile computation to map to workgroups.
//===----------------------------------------------------------------------===//

/// Distribution options for operations when targeting workgroups.
static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
    [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
      return getGPUProcessorIdsAndCounts<gpu::BlockIdOp, gpu::GridDimOp>(
          builder, loc, parallelLoopRanges.size());
    },
    {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
     linalg::DistributionMethod::CyclicNumProcsEqNumIters,
     linalg::DistributionMethod::CyclicNumProcsEqNumIters}};

/// Returns true if the two maps have identical result exprs. One map might have
/// a different dimensionality compared to another, but it is assumed that the
/// dimension correspond exactly.
static bool indexingMapsMatch(AffineMap lhs, AffineMap rhs) {
  ArrayRef<AffineExpr> lhsResults = lhs.getResults();
  ArrayRef<AffineExpr> rhsResults = rhs.getResults();
  if (lhsResults.size() != rhsResults.size()) return false;
  for (unsigned i = 0, e = lhs.getNumResults(); i != e; ++i) {
    if (lhsResults[i] != rhsResults[i]) return false;
  }
  return true;
}

/// Inserts barrier between producer and consumer if the data crosses thread
/// boundary, i.e. if the data is produced by one thread and consumer by the
/// same thread, no barrier is inserted.
static LogicalResult insertBarrier(linalg::LinalgOp producer,
                                   unsigned producerIdx,
                                   linalg::LinalgOp consumer,
                                   unsigned consumerIdx, OpBuilder &builder) {
  if (indexingMapsMatch(producer.getOutputIndexingMap(producerIdx),
                        consumer.getIndexingMap(consumerIdx)))
    return success();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(consumer);
  return insertBarrier(builder, consumer.getLoc());
}

static SubViewOp getPromotedSubviewOfSameSize(SubViewOp subView, Value source,
                                              OpBuilder &builder) {
  Value zero = builder.create<ConstantIndexOp>(subView.getLoc(), 0);
  SmallVector<Value, 4> offsets(source.getType().cast<ShapedType>().getRank(),
                                zero);
  SmallVector<Value, 4> sizes =
      subView.getOrCreateSizes(builder, subView.getLoc());
  SmallVector<Value, 4> strides =
      subView.getOrCreateStrides(builder, subView.getLoc());
  return builder.create<SubViewOp>(subView.getLoc(), source, offsets, sizes,
                                   strides);
}

/// Promotes the fused index to use workgroup memory.
static LogicalResult promoteFusedView(linalg::LinalgOp producer,
                                      unsigned producerIdx,
                                      linalg::LinalgOp consumer,
                                      unsigned consumerIdx,
                                      OpBuilder &builder) {
  if (!producer.hasBufferSemantics() || !consumer.hasBufferSemantics()) {
    return consumer.emitError(
        "promotion only possible for operations with buffer semantics");
  }

  SubViewOp promotedConsumerSubview =
      consumer.getShapedOperands()[consumerIdx].getDefiningOp<SubViewOp>();
  if (!promotedConsumerSubview) {
    return consumer.emitError("expected operand ")
           << consumerIdx
           << " to be result of a subview operation, possible attempt of "
              "promotion without tiling";
  }

  MemRefType fusedProducerOperandType =
      producer.getOutputShapedType(producerIdx).cast<MemRefType>();
  if (fusedProducerOperandType.getMemorySpace() == getWorkgroupMemorySpace()) {
    return producer.emitError("unhandled producer operand already promoted");
  }
  SubViewOp promotedProducerSubview =
      producer.getOutputBuffers()[producerIdx].getDefiningOp<SubViewOp>();
  if (!promotedProducerSubview) {
    return producer.emitError("expected output ")
           << producerIdx
           << " to be result of a subview operation, possible attempt of "
              "promotion without fusion";
  }

  Optional<linalg::PromotionInfo> promotionInfo = llvm::None;

  // If the view in the consumer has already been promoted, use the same source.
  if (consumer.getShapedType(consumerIdx).cast<MemRefType>().getMemorySpace() ==
      getWorkgroupMemorySpace()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(producer);
    promotionInfo = {promotedConsumerSubview.source(), promotedConsumerSubview};
    SubViewOp subview = getPromotedSubviewOfSameSize(
        promotedProducerSubview, promotionInfo->fullLocalView, builder);
    producer.getOperation()->setOperand(producer.getNumInputs() + producerIdx,
                                        subview.getResult());
  } else {
    promotionInfo = linalg::promoteSubviewAsNewBuffer(
        builder, consumer.getLoc(), promotedConsumerSubview,
        allocateWorkgroupMemory);
    if (!promotionInfo) {
      return consumer.emitError("failed to promote operand ")
             << consumerIdx << " to new buffer";
    }
    consumer.getOperation()->setOperand(consumerIdx,
                                        promotionInfo->partialLocalView);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(producer);
    SubViewOp subview = getPromotedSubviewOfSameSize(
        promotedProducerSubview, promotionInfo->fullLocalView, builder);
    producer.getOperation()->setOperand(producer.getNumInputs() + producerIdx,
                                        subview.getResult());
    deallocateWorkgroupMemory(builder, promotionInfo->fullLocalView);
  }

  return insertBarrier(producer, producerIdx, consumer, consumerIdx, builder);
}

/// Fuses operations in `fusableOps` that can be fused with the
/// `tiledOp`. `fusableOps` is expected to be a sequence of linalg operations
/// with buffer semantic. `linalgOp` is the original untiled operation of this
/// op used to get dependence information from `dependenceGraph`. Note
/// `fusableOps` expects the last entry to be `linalgOp`.
static Optional<SmallVector<linalg::LinalgOp, 2>> fuseOperationChain(
    linalg::LinalgOp linalgOp, linalg::LinalgOp tiledOp,
    ArrayRef<Operation *> fusableOps,
    const linalg::LinalgDependenceGraph &dependenceGraph, OpBuilder &builder,
    bool doPromotion = true) {
  SmallVector<linalg::LinalgOp, 2> fusedLinalgOps(fusableOps.size(), nullptr);
  fusedLinalgOps.back() = tiledOp;
  for (unsigned i = fusableOps.size() - 1; i > 0; i--) {
    linalg::LinalgOp consumer = fusableOps[i];
    linalg::LinalgOp tiledConsumer = fusedLinalgOps[i];
    linalg::LinalgOp producer = fusableOps[i - 1];
    Optional<linalg::LinalgDependenceGraph::LinalgDependenceGraphElem>
        dependence = llvm::None;
    for (auto it : dependenceGraph.getDependentOperationsInto(consumer)) {
      if (it.dependentOpView.op == producer) {
        dependence = it;
        break;
      }
    }
    if (!dependence) {
      consumer.emitError(
          "unable to find dependence information to previous operation in "
          "chain");
      return llvm::None;
    }

    Optional<unsigned> producerIdx =
        producer.getIndexOfOutputBuffer(dependence->dependentOpView.view);
    Optional<unsigned> consumerIdx =
        consumer.getIndexOfShapedOperand(dependence->indexingView);
    assert(producerIdx && consumerIdx &&
           "unable to find index of the fused view in producer/consumer");
    builder.setInsertionPoint(tiledConsumer);
    edsc::ScopedContext context(builder, linalgOp.getLoc());
    linalg::LinalgOp fusedOp =
        fuse(builder, producer, *producerIdx, tiledConsumer, *consumerIdx);
    fusedLinalgOps[i - 1] = fusedOp;

    LLVM_DEBUG({
      llvm::dbgs() << "--- After fusion ---\n";
      linalgOp.getOperation()->getParentOp()->print(
          llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (doPromotion) {
      // Promote the shared view to use workgroup memory.
      if (failed(promoteFusedView(fusedOp, *producerIdx, tiledConsumer,
                                  *consumerIdx, builder))) {
        return llvm::None;
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After promotion ---\n";
      linalgOp.getOperation()->getParentOp()->print(
          llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
  return fusedLinalgOps;
}

struct TiledAndFusedLinalgOp {
  SmallVector<linalg::LinalgOp, 2> fusedOps;
  SmallVector<Operation *, 8> fusedLoops;
  SmallVector<SmallVector<Operation *, 8>, 2> unfusedLoops;
};

/// Tile + fuse the sequence of linalg operations in
/// `linalgOps`. `numFusedLoops` is the number of fused inter-tile loops to
/// generate assuming its valid to do so. `tileSizes` is a list of tile sizes to
/// use for each of the operation in `linalgOps`.
static Optional<TiledAndFusedLinalgOp> tileAndFuseLinalgOpsImpl(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<Operation *> linalgOps, ArrayRef<ArrayRef<int64_t>> tileSizes,
    unsigned numFusedLoops, bool doPromotion = true) {
  assert(!linalgOps.empty());
  assert(linalgOps.size() == tileSizes.size() &&
         "expected as many tile sizes as ops");
  OpBuilder builder(context);
  linalg::LinalgOp lastLinalgOp = cast<linalg::LinalgOp>(linalgOps.back());

  builder.setInsertionPoint(lastLinalgOp);

  // First tile the last operation by the number of fused loops.
  Optional<linalg::TiledLinalgOp> tiledOp = linalg::tileLinalgOp(
      builder, lastLinalgOp,
      linalg::LinalgTilingOptions()
          .setTileSizes(
              ArrayRef<int64_t>(tileSizes.back()).take_front(numFusedLoops))
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setDistributionOptions(workgroupDistributionOptions));
  if (!tiledOp) {
    lastLinalgOp.emitError("failed to tile operation");
    return llvm::None;
  }
  assert(tiledOp->tensorResults.empty() &&
         "unexpected tensor results for tiled operation");

  // Fuse the chain of linalg operations.
  Optional<SmallVector<linalg::LinalgOp, 2>> fusedOps =
      fuseOperationChain(lastLinalgOp, tiledOp->op, linalgOps, dependenceGraph,
                         builder, doPromotion);
  if (!fusedOps) {
    return llvm::None;
  }

  TiledAndFusedLinalgOp tiledAndFusedOp;
  tiledAndFusedOp.fusedLoops = tiledOp->loops;
  tiledAndFusedOp.fusedOps = *fusedOps;

  // Based on the tile sizes some of the loops that are not fused may need to be
  // tiled as well. Do those now.
  tiledAndFusedOp.unfusedLoops.resize(linalgOps.size());
  auto makeUnfusedTileSizes = [&numFusedLoops](ArrayRef<int64_t> opTileSizes) {
    SmallVector<int64_t, 4> tileSizes(opTileSizes.begin(), opTileSizes.end());
    for (unsigned i : llvm::seq<unsigned>(
             0, std::min<unsigned>(numFusedLoops, opTileSizes.size()))) {
      tileSizes[i] = 0;
    }
    return tileSizes;
  };

  for (unsigned i : llvm::seq<unsigned>(0, tileSizes.size())) {
    linalg::LinalgOp fusedOp = tiledAndFusedOp.fusedOps[i];
    assert(fusedOp && "expected fused operation");
    auto unfusedTileSize = makeUnfusedTileSizes(tileSizes[i]);
    if (llvm::any_of(unfusedTileSize,
                     [](int64_t size) -> bool { return size; })) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPoint(fusedOp);
      Optional<linalg::TiledLinalgOp> unfusedTilingOp = linalg::tileLinalgOp(
          builder, fusedOp,
          linalg::LinalgTilingOptions()
              .setTileSizes(unfusedTileSize)
              .setLoopType(linalg::LinalgTilingLoopType::Loops));
      if (!unfusedTilingOp) {
        fusedOp.emitError("failed to tile unfused loops of fused op");
        return llvm::None;
      }
      fusedOp.erase();
      tiledAndFusedOp.fusedOps[i] = unfusedTilingOp->op;
      tiledAndFusedOp.unfusedLoops[i] = unfusedTilingOp->loops;
    }
  }
  return tiledAndFusedOp;
}

static LogicalResult tileAndFuseLinalgOps(
    FuncOp funcOp, const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<Operation *> linalgOps, const LaunchConfig &launchConfig,
    bool doPromotion = true) {
  // Find the ops that are to be tiled and the op that forms the root operation.
  linalg::LinalgOp rootOperation = nullptr;
  for (Operation *op : linalgOps) {
    if (isa<linalg::MatmulOp, linalg::ConvOp, linalg::PoolingMaxOp,
            linalg::PoolingMinOp, linalg::PoolingSumOp>(op)) {
      rootOperation = cast<linalg::LinalgOp>(op);
      break;
    }
  }
  // If there is no root operation, there is nothing to do
  if (!rootOperation) return success();

  unsigned numFusedLoops = getNumOuterParallelLoops(rootOperation);
  SmallVector<ArrayRef<int64_t>, 4> tileSizes;
  for (Operation *op : linalgOps) {
    ArrayRef<int64_t> opTileSizes = launchConfig.getTileSizes(op, 0);
    if (opTileSizes.size() < numFusedLoops) {
      return op->emitError("invalid tile size for operation, need at least ")
             << opTileSizes.size() << " to be able to fuse with root operation";
    }
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    if (opTileSizes.size() > linalgOp.getNumLoops()) {
      opTileSizes = opTileSizes.take_front(linalgOp.getNumLoops());
    }
    tileSizes.push_back(opTileSizes);
  }
  Optional<TiledAndFusedLinalgOp> tiledAndFused =
      tileAndFuseLinalgOpsImpl(funcOp.getContext(), dependenceGraph, linalgOps,
                               tileSizes, numFusedLoops, doPromotion);
  if (!tiledAndFused) return failure();

  // // Update the launch information.
  // OpBuilder builder(funcOp.getContext());
  // if (failed(updateWorkGroupSize(funcOp, launchConfig.getWorkgroupSize())) ||
  //     (funcOp.getAttr(getNumWorkgroupsFnAttrName()) &&
  //      failed(createNumWorkgroupsFromResultShape(
  //          builder, rootOperation, funcOp,
  //          launchConfig.getTileSizes(rootOperation, 0))))) {
  //   return failure();
  // }

  // Mark the tiled and fused ops with "workgroup" marker.
  for (auto fusedOp : tiledAndFused->fusedOps) {
    setMarker(fusedOp, getWorkgroupMarker());
  }
  // Mark the original operations as to be deleted.
  for (auto originalOp : linalgOps) {
    setMarker(originalOp, getDeleteMarker());
  }

  return applyCanonicalizationPatterns(funcOp.getContext(), funcOp);
}

//===----------------------------------------------------------------------===//
// Patterns to promote subviews to workgroup memory
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to promote matmul operands to workgroup memory.
struct PromoteMatmulSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteMatmulSubviewsPattern(MLIRContext *context,
                               linalg::LinalgPromotionOptions options,
                               linalg::LinalgMarker marker,
                               PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}
};

/// Patterns to promote convolution operands to workgroup memory.
// TODO(ravishankarm): This pattern is only promoting the image subview to
// workgroup memory. In reality we should also be able to promote the filter
// subview to workgroup memory as well. Since none of the loops used to access
// the filter are tiled, this would mean the entire filter is moved to workgroup
// memory. Two reasons this is not done right now:
// 1) Linalg when tiling doesnt create a subview for the filter (since none of
//    its dimensions are tiled. This needs to be relaxed (maybe by using an
//    option.
// 2) Maybe there are better alternatives for handling filter (using different
//    StorageClasses, since for inference workloads these are model
//    constants. This is TBD.
struct PromoteConvolutionSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::ConvOp> {
  PromoteConvolutionSubviewsPattern(MLIRContext *context,
                                    linalg::LinalgPromotionOptions options,
                                    linalg::LinalgMarker marker,
                                    PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::ConvOp>(
            context,
            options.setOperandsToPromote({1}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}
};
}  // namespace

static void populatePromotionPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns) {
  patterns
      .insert<PromoteMatmulSubviewsPattern, PromoteConvolutionSubviewsPattern>(
          context,
          linalg::LinalgPromotionOptions()
              .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                            deallocateWorkgroupMemory)
              .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory),
          linalg::LinalgMarker(
              Identifier::get(getWorkgroupMarker(), context),
              Identifier::get(getWorkgroupMemoryMarker(), context)));
}

//===----------------------------------------------------------------------===//
// Patterns and methods for subgroup tiling.
//===----------------------------------------------------------------------===//

/// Computes the Value for subgroupID along each dimension given number of
/// subgroups `numSubGroups` along each dimension (x-first, y-second, z-third).
static SmallVector<linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> numSubgroups) {
  Type indexType = builder.getIndexType();
  Value subgroupId = builder.create<gpu::SubgroupIdOp>(loc, indexType);
  SmallVector<linalg::ProcInfo, 2> procInfo(numSubgroups.size());

  // subgroupID
  //   = id.z * nsubgroups.y * nsubgroups.x + id.y * nsubgroups.x + id.x
  using edsc::op::operator%;
  for (size_t i = 0, e = numSubgroups.size(); i != e; ++i) {
    Value nprocs = builder.create<ConstantIndexOp>(loc, numSubgroups[i]);
    Value procId = subgroupId % nprocs;
    procInfo[e - i - 1] = linalg::ProcInfo{procId, nprocs};
    subgroupId = builder.create<SignedDivIOp>(loc, subgroupId, nprocs);
  }
  return procInfo;
}

namespace {
/// Pattern to tile linalg.matmul for subgroups.
struct TileMatmulSubgroupPattern
    : public linalg::LinalgTilingPattern<linalg::MatmulOp> {
  using Base = linalg::LinalgTilingPattern<linalg::MatmulOp>;
  TileMatmulSubgroupPattern(MLIRContext *context,
                            linalg::LinalgTilingOptions options,
                            linalg::LinalgMarker marker,
                            PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit) {}
};
}  // namespace

/// Patterns for second level tiling to target subgroups.
static void populateTilingToSubgroupPatterns(
    MLIRContext *context, const LaunchConfig &launchConfig,
    OwningRewritePatternList &patterns) {
  std::function<SmallVector<Value, 4>(OpBuilder &, Operation *)>
      getInnerTileSizeFn =
          [&launchConfig](OpBuilder &builder,
                          Operation *operation) -> SmallVector<Value, 4> {
    ArrayRef<int64_t> tileSizes = launchConfig.getTileSizes(operation, 1);
    if (tileSizes.empty()) return {};
    SmallVector<Value, 4> tileSizesVal;
    tileSizesVal.reserve(tileSizes.size());
    for (auto val : tileSizes) {
      tileSizesVal.push_back(
          builder.create<ConstantIndexOp>(operation->getLoc(), val));
    }
    return tileSizesVal;
  };

  auto getSubgroupProcInfoFn = [&launchConfig](
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Range> parallelLoopRanges) {
    ArrayRef<int64_t> numSubgroups =
        launchConfig.getNumSubgroups().take_front(parallelLoopRanges.size());
    return getSubgroupIdsAndCounts(builder, loc, numSubgroups);
  };
  linalg::LinalgLoopDistributionOptions subgroupDistributionOptions = {
      getSubgroupProcInfoFn,
      {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters}};
  patterns.insert<TileMatmulSubgroupPattern>(
      context,
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(subgroupDistributionOptions),
      linalg::LinalgMarker(
          /*matchDisjunction=*/{Identifier::get(getWorkgroupMemoryMarker(),
                                                context),
                                Identifier::get(getWorkgroupMarker(), context)},
          /*replacement=*/Identifier::get(getVectorizeMarker(), context)));
}

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(MLIRContext *context,
                                          const LaunchConfig &launchConfig,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<linalg::LinalgVectorizationPattern<linalg::MatmulOp>>(
      context,
      linalg::LinalgMarker(Identifier::get(getVectorizeMarker(), context)));
}

//====---------------------------------------------------------------------===//
// Patterns for unrolling vectors.
//====---------------------------------------------------------------------===//

static void populateVectorUnrollPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<vector::UnrollVectorPattern<vector::ContractionOp>>(
      context,
      vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorSize));
  vector::populateVectorToVectorCanonicalizationPatterns(patterns, context);
  vector::populateVectorToVectorTransformationPatterns(patterns, context);
}

void LinalgTileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Override options with command line values.
  if (!tileSizes.empty())
    options.tileSizes.assign(tileSizes.begin(), tileSizes.end());
  if (!workgroupSize.empty())
    options.workgroupSize.assign(workgroupSize.begin(), workgroupSize.end());
  if (useWorkgroupMemory) options.useWorkgroupMemory = true;
  if (useVectorization) options.useVectorization = true;

  LLVM_DEBUG(
      llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";);
  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    Region &body = funcOp.getBody();
    if (!llvm::hasSingleElement(body.getBlocks())) {
      funcOp.emitError("unhandled dispatch function with multiple blocks");
      return signalPassFailure();
    }
    Block &block = body.front();
    auto linalgOps = block.getOps<linalg::LinalgOp>();
    if (linalgOps.empty()) continue;

    LaunchConfig launchConfig;
    SmallVector<Operation *, 4> linalgOpsVec(linalgOps.begin(),
                                             linalgOps.end());
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOpsVec);
    if (failed(launchConfig.init(context, dependenceGraph, options,
                                 linalgOpsVec))) {
      funcOp.emitError("unable to find launch configuration");
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "@func " << funcOp.getName() << ": # workgroup sizes: [";
      interleaveComma(launchConfig.getWorkgroupSize(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      for (auto op : linalgOps) {
        llvm::dbgs() << "\t" << op.getOperation()->getName() << " : ";
        TileSizesListType const &tileSizes = launchConfig.getTileSizes(op);
        llvm::dbgs() << "{";
        std::string sep = "";
        for (auto &level : enumerate(tileSizes)) {
          llvm::dbgs() << sep << level.index() << " : [";
          sep = ", ";
          interleaveComma(level.value(), llvm::dbgs());
          llvm::dbgs() << "]";
        }
        llvm::dbgs() << "}\n";
      }
    });

    // Tile + fuse + promote to map to workgroups.
    if (failed(tileAndFuseLinalgOps(funcOp, dependenceGraph, linalgOpsVec,
                                    launchConfig))) {
      return signalPassFailure();
    }

    // Delete the ops that are marked for deletion.
    funcOp.walk([](linalg::LinalgOp linalgOp) {
      if (hasMarker(linalgOp.getOperation(), getDeleteMarker()))
        linalgOp.getOperation()->erase();
    });

    LLVM_DEBUG({
      llvm::dbgs() << "--- After First level of tile+distribute ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (options.useWorkgroupMemory) {
      // The promotion patterns are put separate from the tiling patterns to
      // make sure that the allocated scratchspace memory is constant sizes
      // which requires some folding to trigger.
      OwningRewritePatternList promotionPatterns;
      populatePromotionPatterns(context, promotionPatterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, promotionPatterns)) ||
          failed(applyCanonicalizationPatterns(context, funcOp))) {
        funcOp.emitError("promotion to workgroup memory failed");
        return signalPassFailure();
      }

      LLVM_DEBUG({
        llvm::dbgs() << "--- After Promotion  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    if (options.useVectorization) {
      {
        OwningRewritePatternList secondLevelTilingPatterns;
        populateTilingToSubgroupPatterns(context, launchConfig,
                                         secondLevelTilingPatterns);
        if (failed(applyPatternsAndFoldGreedily(funcOp,
                                                secondLevelTilingPatterns)) ||
            failed(applyCanonicalizationPatterns(context, funcOp))) {
          funcOp.emitError("second level of tiling failed");
          return signalPassFailure();
        }

        LLVM_DEBUG({
          llvm::dbgs() << "--- After Second level Tiling  ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }

      {
        OwningRewritePatternList vectorizationPatterns;
        populateVectorizationPatterns(context, launchConfig,
                                      vectorizationPatterns);
        if (failed(
                applyPatternsAndFoldGreedily(funcOp, vectorizationPatterns)) ||
            failed(applyCanonicalizationPatterns(context, funcOp))) {
          funcOp.emitError("vectorization failed");
          return signalPassFailure();
        }
        LLVM_DEBUG({
          llvm::dbgs() << "--- After Vectorization ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }

      {
        OwningRewritePatternList vectorUnrollPatterns;
        populateVectorUnrollPatterns(context, vectorUnrollPatterns);
        if (failed(
                applyPatternsAndFoldGreedily(funcOp, vectorUnrollPatterns)) ||
            failed(applyCanonicalizationPatterns(context, funcOp))) {
          funcOp.emitError("unrolling vectors failed");
          return signalPassFailure();
        }
        LLVM_DEBUG({
          llvm::dbgs() << "--- After Vector Unroll ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }
    }

    launchConfig.finalize(funcOp);
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLinalgTileAndFusePass(
    const SPIRVCodegenOptions &options) {
  return std::make_unique<LinalgTileAndFusePass>(options);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
