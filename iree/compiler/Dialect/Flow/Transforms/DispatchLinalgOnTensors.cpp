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

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/DestructiveUpdateUtils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-flow-dispatch-linalg-on-tensors"

static llvm::cl::list<int64_t> clLinalgOnTensorsTileSizes(
    "iree-flow-dispatch-linalg-on-tensors-tile-sizes",
    llvm::cl::desc("Comma-separated list of tile sizes for tiling on tensors"),
    llvm::cl::CommaSeparated);

// TODO(ravishankarm): Remove this option after addressing fusion.
static llvm::cl::opt<bool> clLinalgOnTensorsEnableForceFusion(
    "iree-flow-dispatch-linalg-on-tensors-enable-fusion-always",
    llvm::cl::desc("Option to force fuse linalg operations on tensors"),
    llvm::cl::init(false));

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupAttr[] = "__fused_op__";

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static unsigned kNumMaxParallelDims = 3;

/// PatternRewriter that allows replacing only a subset of uses.
/// Since this only adds a method, it can just be static_cast'ed to when
/// applying a rewrite.
/// TODO(nicolasvasilache): upstream support for this is landing, rebase on that
struct PatternRewriterWithScopedReplaceOp : public PatternRewriter {
  void replaceOpWithinScope(Operation *op, ValueRange newValues, Block *block) {
    // Notify the rewriter subclass that we're about to replace this root.
    notifyRootReplaced(op);

    assert(op->getNumResults() == newValues.size() &&
           "incorrect # of replacement values");
    bool erase = true;
    SmallVector<Operation *, 4> ops;
    SmallVector<Value, 4> operands, repls;
    for (auto &use : op->getUses()) {
      if (!block->getParentOp()->isProperAncestor(use.getOwner())) {
        erase = false;
        continue;
      }
      OpResult opResult = use.get().cast<OpResult>();
      ops.push_back(use.getOwner());
      operands.push_back(use.get());
      repls.push_back(newValues[opResult.getResultNumber()]);
    }
    // Perform the actual replacements.
    for (auto it : llvm::zip(ops, operands, repls))
      std::get<0>(it)->replaceUsesOfWith(std::get<1>(it), std::get<2>(it));
    if (erase) {
      notifyOperationRemoved(op);
      op->erase();
    }
  }
};

struct DispatchLinalgOnTensorsPass
    : public PassWrapper<DispatchLinalgOnTensorsPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    AffineDialect, scf::SCFDialect, ShapeDialect>();
  }
  DispatchLinalgOnTensorsPass() = default;
  DispatchLinalgOnTensorsPass(const DispatchLinalgOnTensorsPass &pass) {}
  void runOnOperation() override;
};

/// Returns the number of consecutive outer loops that are "parallel". This is a
/// copy of the function from
/// iree/compiler/Conversion/CodegenUtils/FunctionUtils.h that is duplicated
/// here to avoid adding an build dependency.
static size_t getNumOuterParallelLoops(linalg::LinalgOp op) {
  return op.iterator_types()
      .getValue()
      .take_while([](Attribute attr) -> bool {
        return linalg::isParallelIteratorType(attr);
      })
      .size();
}

/// Returns the number of loops of the operation that are to be tiled.
static size_t getNumTilableLoops(linalg::LinalgOp op) {
  return std::min<size_t>(getNumOuterParallelLoops(op), kNumMaxParallelDims);
}

// Creates a flow.dispatch.workgroup op without arguments.
// All the necessary operands are transiently captured and rewritten late as
// operands. This greatly simplifies transformations into the resulting op.
static IREE::Flow::DispatchWorkgroupsOp buildOperandLessFlowDispatchWorkgroupOp(
    PatternRewriter &rewriter, linalg::LinalgOp root,
    linalg::LinalgOp &clonedRoot) {
  Location loc = root->getLoc();
  Value one = rewriter.create<ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> count = llvm::to_vector<4>(llvm::map_range(
      root.createLoopRanges(rewriter, loc), [](Range r) { return r.size; }));
  count.resize(getNumTilableLoops(root));
  count = llvm::to_vector<4>(llvm::reverse(count));
  count.resize(kNumMaxParallelDims, one);

  auto dispatchOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc, count, root->getResultTypes(), ValueRange{});
  Region &region = dispatchOp.body();
  Block *block = &region.front();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(block);
    clonedRoot = cast<linalg::LinalgOp>(rewriter.clone(*root.getOperation()));
    // Note: DispatchOutputStoreOp is an abstraction jump that consumes the SSA
    // value produced by `clonedRoot` but it does not comply with the semantics
    // of DispatchWorkgroupsOp which explicitly states:
    // "behavior is undefined if multiple workgroups store to the same regions
    // of the output tensors".
    // Similarly to sequentialized SPMD loops, the semantics is valid assuming a
    // sequential ordering of execution.
    // After destructive update rewrites, the abstraction gap disappears.
    for (auto it : llvm::zip(clonedRoot->getResults(),
                             dispatchOp.body().getArguments().take_back(
                                 clonedRoot->getNumResults()))) {
      rewriter.create<IREE::Flow::DispatchOutputStoreOp>(
          loc, std::get<0>(it), std::get<1>(it), llvm::None, llvm::None,
          llvm::None);
    }
    // TODO(nicolasvasilache): return `clonedRoot->getResults()` once we have
    // shape operands and we drop tie_shape.
    rewriter.create<IREE::Flow::ReturnOp>(loc);
  }
  LLVM_DEBUG(llvm::dbgs() << "Created dispatchOp shell " << *dispatchOp
                          << "\n");
  return dispatchOp;
}

// Only fuses the first producer for the purpose of connecting the pieces.
// The impl does not worry about the dispatchOp, operands and arguments are set
// in a post-pattern `legalizeDispatchWorkgroupOperands` function.
// To simplify the implementation of the dispatch region formation, we just
// clone the op that needs to be fused inside the dispatch region and just fuse
// that one. This avoid any concerns related to tensor operands that are only
// used for their DimOp. This is a canonicalization that is more involved than
// necessary across the boundary of regions without captures.
//
// TODO(nicolasvasilache): Enhance fusion.
//
// TODO(nicolasvasilache: This implementation jumps an abstraction gap as it
// knows that `clonedLinalgOp` has been tiled into `tiledLinalgOp`. In the case
// where a `rootOp`, i.e. the untiled original operation used to create the
// dispatch region, can be fused with its producer, this allows calling into a
// `fuseProducerOfTensor` to which we provide the producer by construction. This
// avoids an analysis that would need to reconstruct a destructive update from
// the loop nest + operations in order to get the producer of an `out` tensor.
// In the future, this analysis should be implemented in core but for now it is
// IREE-only.
static void pullInProducersInSameGroup(
    PatternRewriter &rewriter, IREE::Flow::DispatchWorkgroupsOp dispatchOp,
    ValueRange shapedOperands, linalg::TiledLinalgOp &tiledLinalgOp,
    int64_t groupNum) {
  // Scoped within DispatchWorkgroupOp.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&dispatchOp.getRegion().front());
  for (auto en : llvm::enumerate(shapedOperands)) {
    if (auto producer = en.value().getDefiningOp<linalg::LinalgOp>()) {
      IntegerAttr opGroupNum =
          producer.getOperation()->getAttrOfType<IntegerAttr>(kFusionGroupAttr);
      if (!opGroupNum || opGroupNum.getInt() != groupNum) continue;
      Operation *clonedOpToFuse = rewriter.clone(*producer);
      static_cast<PatternRewriterWithScopedReplaceOp &>(rewriter)
          .replaceOpWithinScope(producer, clonedOpToFuse->getResults(),
                                &dispatchOp.getRegion().front());
      // TODO: this is incorrect on general pattern failures, try pattern within
      // pattern.
      OpResult opResult = en.value().cast<OpResult>();
      auto maybeFusionInfo = linalg::fuseProducerOfTensor(
          rewriter, clonedOpToFuse->getResult(opResult.getResultNumber()),
          tiledLinalgOp.op.getShapedOpOperand(en.index()));
      if (!maybeFusionInfo.hasValue()) {
        rewriter.replaceOp(clonedOpToFuse, producer->getResults());
      }
      maybeFusionInfo->fusedProducer.getOperation()->removeAttr(
          kFusionGroupAttr);
    }
  }
}

// Add tie_shape for all outputs. This provides necessary information for
// a subsequent OutlineDispatchRegion2 pass invocation to work properly.
// TODO(nicolasvasilache): get rid of this once we have a proper shape +
// subshape in core and DispatchWorkgroupOp takes output shape parameters.
static SmallVector<Value, 4> createDispatchTieShapeOp(
    PatternRewriter &rewriter, linalg::LinalgOp linalgOp,
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  assert(dispatchOp->getNumResults() == linalgOp.getNumOutputs());
  MLIRContext *context = linalgOp->getContext();
  Location loc = linalgOp->getLoc();
  SmallVector<Value, 4> shapedResults;
  for (auto it : llvm::zip(linalgOp.getOutputs(), dispatchOp->getResults())) {
    // Insert DimOp and MakeRankedShapeOp just before the dispatchOp to play
    // nicely with a (much later) OutlineDispatchRegions2 which requires all
    // dims and shapes to dominate the dispatchOp region.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(dispatchOp);

    assert(std::get<0>(it).getType() == std::get<1>(it).getType());
    auto rankedTensorType = std::get<0>(it).getType().cast<RankedTensorType>();
    if (rankedTensorType.hasStaticShape()) {
      shapedResults.push_back(std::get<1>(it));
      continue;
    }
    auto rank = rankedTensorType.getRank();
    SmallVector<Value, 4> dims;
    dims.reserve(rank);
    for (unsigned d = 0, e = rank; d < e; ++d) {
      if (rankedTensorType.isDynamicDim(d)) {
        dims.push_back(rewriter.create<DimOp>(loc, std::get<0>(it), d));
      }
    }
    auto shapeOp = rewriter.create<Shape::MakeRankedShapeOp>(
        loc, Shape::RankedShapeType::get(rankedTensorType.getShape(), context),
        dims);

    // The TieShapeOp use the dispatchOp results.
    rewriter.setInsertionPointAfter(dispatchOp);
    shapedResults.push_back(
        rewriter.create<Shape::TieShapeOp>(loc, std::get<1>(it), shapeOp));
  }
  return shapedResults;
}

// Rewrite pattern to ensure only ops with tensor semantics are tiled.
struct TileAndDistributeOnTensorsPattern
    : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileAndDistributeOnTensorsPattern(linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter marker,
                                    PatternBenefit benefit = 1)
      : Base(options, marker, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
    IntegerAttr rootOpAttr = op->getAttrOfType<IntegerAttr>(kRootOpAttr);
    if (!rootOpAttr) return failure();

    linalg::LinalgOp clonedLinalgOp;
    IREE::Flow::DispatchWorkgroupsOp dispatchOp =
        buildOperandLessFlowDispatchWorkgroupOp(rewriter, linalgOp,
                                                clonedLinalgOp);
    // Scoped within DispatchWorkgroupOp.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(clonedLinalgOp);

    linalg::TiledLinalgOp tiledLinalgOp;
    LogicalResult tilingResult =
        Base::matchAndRewriteBase(clonedLinalgOp, rewriter, tiledLinalgOp);
    if (failed(tilingResult)) {
      // GreedyPatternRewriter is not transactional and does not stop on
      // failure. Must explicitly delete on all failure paths.
      rewriter.eraseOp(dispatchOp);
      return failure();
    }
    // Keep track of the shapedOperands for fusion.
    SmallVector<Value, 4> shapedOperands(clonedLinalgOp.getShapedOperands());
    rewriter.replaceOp(clonedLinalgOp, tiledLinalgOp.tensorResults);

    pullInProducersInSameGroup(rewriter, dispatchOp, shapedOperands,
                               tiledLinalgOp, rootOpAttr.getInt());

    tiledLinalgOp.op.getOperation()->removeAttr(kRootOpAttr);
    SmallVector<Value, 4> shapedResults =
        createDispatchTieShapeOp(rewriter, linalgOp, dispatchOp);
    rewriter.replaceOp(op, shapedResults);
    return success();
  }
};

template <typename OpTy>
static Value buildFlowWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

/// Returns true if an operation is to always be cloned into the dispatch region
/// when its value is used within it.
bool isAlwaysClonedIntoDispatchOp(Operation *op) {
  if (isa<linalg::InitTensorOp, linalg::TensorReshapeOp>(op)) {
    return true;
  }
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    return constantOp.getResult().getType().isIntOrFloat();
  }
  return false;
}

/// Computes the values that will be eventually be used within the dispatch
/// workgroup op but defined outside the op after all clonable operations are
/// cloned into the region. Returns (by reference) the clonable operations too,
/// in order in which they can be cloned within the region to satisfy use-def
/// relationships between them.
void getUsedValuesDefinedAboveAfterCloningOps(
    Region &region, llvm::SetVector<Value> &valuesDefinedAbove,
    llvm::SmallVector<Operation *, 4> &clonedOps) {
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  llvm::SetVector<Value> visited;
  SmallVector<Value, 4> worklist;
  worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
  valuesDefinedAbove.clear();
  while (!worklist.empty()) {
    Value outsideValue = worklist.pop_back_val();
    if (visited.count(outsideValue)) continue;
    visited.insert(outsideValue);
    Operation *definingOp = outsideValue.getDefiningOp();
    if (!definingOp || !isAlwaysClonedIntoDispatchOp(definingOp)) {
      valuesDefinedAbove.insert(outsideValue);
      continue;
    }
    clonedOps.push_back(definingOp);
    worklist.append(definingOp->operand_begin(), definingOp->operand_end());
  }
  // The cloned operations form a DAG. Return the cloned operations in reverse
  // so the leaves come first, and can be cloned in-order into the dispatch
  // region. Do the same for `valuesDefinedAbove` to keep return values
  // consistent with order gotten from `mlir::getUsedValuesDefinedAbove` (more
  // for a convention that correctness)
  clonedOps = llvm::to_vector<4>(llvm::reverse(clonedOps));
  llvm::SetVector<Value> reverseValuesDefinedAbove;
  for (auto value : llvm::reverse(valuesDefinedAbove)) {
    reverseValuesDefinedAbove.insert(value);
  }
  std::swap(reverseValuesDefinedAbove, valuesDefinedAbove);
}

/// Returns a valid insertion point for an operation based on the values used.
static Operation *getInsertionPoint(Block &block, ArrayRef<Value> usedValues) {
  Operation *insertAfter = &block.front();
  for (auto value : usedValues) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) continue;
    if (definingOp->getBlock() != &block) return nullptr;
    if (insertAfter->isBeforeInBlock(definingOp)) {
      insertAfter = definingOp;
    }
  }
  return insertAfter->getNextNode();
}

// After outlining in dispatch region we can rewrite the dispatch ops with
// proper captures.
// A later RematerializeDispatchConstants should be called to avoid passing
// unnecessary constant arguments.
static LogicalResult legalizeDispatchWorkgroupOperands(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Location loc = dispatchOp.getLoc();
  Block &block = dispatchOp.body().front();
  unsigned numOldBBArgs = block.getNumArguments();
  OpBuilder b = OpBuilder::atBlockBegin(&block);

  llvm::SetVector<Value> valuesDefinedAbove;
  llvm::SmallVector<Operation *, 4> clonedOps;
  getUsedValuesDefinedAboveAfterCloningOps(dispatchOp.body(),
                                           valuesDefinedAbove, clonedOps);

  BlockAndValueMapping map;
  // Replace valuesDefinedAbove by new BB args (including the op's operands).
  for (Value operand : valuesDefinedAbove) {
    if (auto rt = operand.getType().dyn_cast<RankedTensorType>()) {
      block.addArgument(IREE::Flow::DispatchInputType::get(
          rt.getShape(), rt.getElementType()));
    } else {
      block.addArgument(operand.getType());
    }

    Value bbArg = block.getArguments().back();
    Value repl = bbArg;
    if (bbArg.getType().isa<IREE::Flow::DispatchInputType>()) {
      repl = b.create<IREE::Flow::DispatchInputLoadOp>(loc, operand.getType(),
                                                       bbArg);
    } else if (bbArg.getType().isa<IREE::Flow::DispatchOutputType>()) {
      // TODO(nicolasvasilache): do something useful.
      continue;
    }
    map.map(operand, repl);
  }

  // The only existing arguments are for the outputs. Just need to add a new
  // argument for the outputs and remap the value to use the new argument.
  for (auto ba : block.getArguments().take_front(numOldBBArgs)) {
    assert(ba.getType().isa<IREE::Flow::DispatchOutputType>());
    map.map(ba, block.addArgument(ba.getType()));
  }

  auto getUsesOfValueOutsideOfDispatchOp = [&](Value v) {
    SmallPtrSet<Operation *, 4> res;
    for (Operation *user : v.getUsers())
      if (!dispatchOp->isAncestor(user)) res.insert(user);
    return res;
  };
  // Replace all uses of mapped values within the dispatch op to the new value.
  for (Value value : valuesDefinedAbove) {
    auto uses = getUsesOfValueOutsideOfDispatchOp(value);
    value.replaceAllUsesExcept(map.lookup(value), uses);
  }

  // Clone the marked operations.
  for (Operation *op : clonedOps) {
    SmallVector<Value, 2> clonedOpOperands = llvm::to_vector<2>(llvm::map_range(
        op->getOperands(), [&map](Value v) { return map.lookup(v); }));
    Operation *insertionPoint = getInsertionPoint(block, clonedOpOperands);
    if (!insertionPoint) {
      return op->emitOpError(
          "failed to find insetion point within dispatch workgroup op for "
          "cloned operation");
    }
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(insertionPoint);
    b.clone(*op, map);
    for (Value result : op->getResults()) {
      auto uses = getUsesOfValueOutsideOfDispatchOp(result);
      result.replaceAllUsesExcept(map.lookup(result), uses);
    }
  }

  for (Value ba : block.getArguments().take_front(numOldBBArgs)) {
    ba.replaceAllUsesWith(map.lookup(ba));
  }

  // Drop old BB args.
  block.eraseArguments(
      llvm::to_vector<4>(llvm::seq<unsigned>(0, numOldBBArgs)));

  // Set the values captured from above as the new operands.
  dispatchOp.operandsMutable().assign(llvm::to_vector<4>(valuesDefinedAbove));

  return success();
}

/// Checks if the `producer` can be fused into the dispatch region of the
/// `consumer`.
// TODO(ravishankarm): For now this is doing only some limited fusion. Ideally
// the way we create the dispatches itself should account for data access
// pattern within the consumer and producer. So probably need to do a pre-pass
// to decide what to fuse and then do the fusion. For now just use a simple
// default scheme to get the ball rolling.
static bool isProducerFusableWithConsumer(linalg::LinalgOp producer,
                                          linalg::LinalgOp consumer) {
  if (clLinalgOnTensorsEnableForceFusion) return true;
  // Fuse if the dependence from producer to consumer is to the `outs` operand
  // of the consumer and the producer is a parallel operation
  llvm::DenseSet<Value> producerResults(producer.getOperation()->result_begin(),
                                        producer.getOperation()->result_end());
  auto isProducerResultValue = [&producerResults](OpOperand &operand) -> bool {
    return producerResults.count(operand.get());
  };
  auto isElementWiseParallelOp = [](linalg::LinalgOp op) -> bool {
    return llvm::all_of(op.iterator_types(), [](Attribute attr) {
      return attr.cast<StringAttr>().getValue() ==
             toString(IteratorType::Parallel);
    });
  };
  return llvm::none_of(consumer.getInputOpOperands(), isProducerResultValue) &&
         llvm::any_of(consumer.getOutputOpOperands(), isProducerResultValue) &&
         isElementWiseParallelOp(producer);
}

/// For a given block partition the LinalgOps in the block into fusable
/// groups. All analysis of what to fuse happens here. For now this is just
/// hard-wiring from basic heuristic but this could be adapted to have 1) better
/// heuristics and 2) use a search approach to decide what all should be fused.
static void decideFusableLinalgOps(FuncOp funcOp) {
  unsigned numRootOps = 0;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  for (Block &block : funcOp) {
    auto linalgOps = block.getOps<linalg::LinalgOp>();

    // Start with a root operation. Everything will be "fused with it".
    for (linalg::LinalgOp linalgOp : linalgOps) {
      Operation *op = linalgOp.getOperation();
      if (op->hasAttr(kRootOpAttr) || op->hasAttr(kFusionGroupAttr)) continue;
      // For now only matmul op as root.
      if (!isa<linalg::MatmulOp>(op)) continue;
      unsigned currGroupNum = numRootOps++;
      op->setAttr(kRootOpAttr, builder.getI64IntegerAttr(currGroupNum));
      for (auto operand : linalgOp.getShapedOperands()) {
        auto producer = operand.getDefiningOp<linalg::LinalgOp>();
        if (!producer) continue;
        Operation *producerOp = producer.getOperation();
        if (!producerOp->hasAttr(kRootOpAttr) &&
            !producerOp->hasAttr(kFusionGroupAttr) &&
            isProducerFusableWithConsumer(producer, op)) {
          producerOp->setAttr(kFusionGroupAttr,
                              builder.getI64IntegerAttr(currGroupNum));
        }
      }
    }

    // Finally whatever is not marked for fusion becomes a root operation.
    for (linalg::LinalgOp linalgOp : linalgOps) {
      Operation *op = linalgOp.getOperation();
      if (op->hasAttr(kRootOpAttr) || op->hasAttr(kFusionGroupAttr)) continue;
      op->setAttr(kRootOpAttr, builder.getI64IntegerAttr(numRootOps++));
    }
  }
}

void DispatchLinalgOnTensorsPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  // `isEntryPoint` functions are the ones that are marked public.
  if (!funcOp.isPublic()) return;

  MLIRContext *context = funcOp->getContext();
  context->allowUnregisteredDialects(true);

  decideFusableLinalgOps(funcOp);

  // Distribution strategy along at most 3 dimensions with WorkgroupIdOp in
  // range [0, WorkgroupSizeOp).
  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();
        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0;
             dim < std::min<size_t>(numParallelDims, kNumMaxParallelDims);
             ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildFlowWorkgroupInfoOp<Flow::DispatchWorkgroupIDOp>(builder,
                                                                    dim),
              buildFlowWorkgroupInfoOp<Flow::DispatchWorkgroupCountOp>(builder,
                                                                       dim)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tileSizeFn = [&](OpBuilder &builder,
                        Operation *op) -> SmallVector<Value, 4> {
    auto numTiledLoops = getNumTilableLoops(cast<linalg::LinalgOp>(op));
    SmallVector<Value, 4> useTileSizes(numTiledLoops);
    if (!clLinalgOnTensorsTileSizes.empty()) {
      SmallVector<int64_t, 2> tileSizes(clLinalgOnTensorsTileSizes.begin(),
                                        clLinalgOnTensorsTileSizes.end());
      useTileSizes.resize(std::min<size_t>(tileSizes.size(), numTiledLoops));
      return llvm::to_vector<4>(llvm::map_range(
          ArrayRef<int64_t>(tileSizes).take_front(
              std::min<size_t>(tileSizes.size(), numTiledLoops)),
          [&](int64_t t) -> Value {
            return builder.create<ConstantIndexOp>(op->getLoc(), t);
          }));
    }
    for (size_t dim = 0; dim < numTiledLoops; ++dim) {
      useTileSizes[numTiledLoops - dim - 1] =
          buildFlowWorkgroupInfoOp<Flow::DispatchWorkgroupSizeOp>(builder, dim);
    }
    return useTileSizes;
  };

  // Use the workgroup size as a proxy for tile size here. At the flow level
  // this represents the "workload" per processors and is not necessarily tied
  // to the workgroup size specified by the backend.
  OwningRewritePatternList patterns;
  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setDistributionOptions(workgroupDistributionOptions)
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(tileSizeFn);
  assert(linalgTilingOptions.distribution.hasValue());

  patterns.insert<TileAndDistributeOnTensorsPattern>(
      linalgTilingOptions,
      // TODO(nicolavasilache): use refactored `getWorkgroupMarker()`
      linalg::LinalgTransformationFilter(
          ArrayRef<Identifier>(), Identifier::get("workgroup", context)));

  // Add canonicalization patterns.
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns, context);
  patterns.insert<linalg::AffineMinSCFCanonicalizationPattern>(context);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // After outlining in dispatch region we can rewrite the dispatch ops with
  // proper captures.
  if (funcOp
          .walk([&](IREE::Flow::DispatchWorkgroupsOp op) -> WalkResult {
            return legalizeDispatchWorkgroupOperands(op);
          })
          .wasInterrupted()) {
    return signalPassFailure();
  }

  // Rewrite destructive updates and ensure no remaining store remains to the
  // full output.
  bool fail =
      funcOp
          .walk([&](IREE::Flow::DispatchWorkgroupsOp op) {
            if (failed(rewriteLinalgDestructiveUpdates(op))) {
              funcOp.emitError("Failed to rewrite destructive updates in:\n")
                  << *op.getOperation();
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted();
  fail |=
      funcOp
          .walk([&](IREE::Flow::DispatchOutputStoreOp op) {
            if (op.offsets().empty() || op.sizes().empty() ||
                op.strides().empty()) {
              funcOp.emitError("Full-tensor DispatchOutputStoreOp remaining:\n")
                  << *op.getOperation();
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted();
  if (fail) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>> createDispatchLinalgOnTensorsPass() {
  return std::make_unique<DispatchLinalgOnTensorsPass>();
}

static PassRegistration<DispatchLinalgOnTensorsPass> pass(
    "iree-flow-dispatch-linalg-on-tensors-pass",
    "Dispatch Linalg operations on tensors by using tile and distribute",
    [] { return std::make_unique<DispatchLinalgOnTensorsPass>(); });

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
