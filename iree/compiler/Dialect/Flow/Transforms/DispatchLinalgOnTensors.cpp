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

#define DEBUG_TYPE "iree-linalg-tile-and-distribute-on-tensors"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

struct DispatchLinalgOnTensorsPass
    : public PassWrapper<DispatchLinalgOnTensorsPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    AffineDialect, scf::SCFDialect, ShapeDialect>();
  }
  DispatchLinalgOnTensorsPass() = default;
  DispatchLinalgOnTensorsPass(ArrayRef<int64_t> sizes) {
    this->tileSizes = sizes;
  };
  DispatchLinalgOnTensorsPass(const DispatchLinalgOnTensorsPass &pass) {}
  void runOnOperation() override;

 private:
  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Set tile sizes to use"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

static Operation *buildFlowWorkgroupDispatchOp(OpBuilder &b,
                                               linalg::LinalgOp root,
                                               linalg::LinalgOp &clonedRoot) {
  Location loc = root->getLoc();
  // TODO(nicolasvasilache): for now this is a 3-D grid of 1's.
  // In the future make constant and bind late into backend-specific values.
  SmallVector<Value, 1> count(3, b.create<ConstantIndexOp>(loc, 1));

  SmallVector<Value, 4> dispatchOperands;
  dispatchOperands.reserve(root->getNumOperands());
  dispatchOperands.append(root.getInputs().begin(), root.getInputs().end());
  for (OpOperand &opOperand : root.getOutputOpOperands()) {
    if (root.isInitTensor(&opOperand)) {
      dispatchOperands.push_back(opOperand.get());
    } else {
      dispatchOperands.push_back(
          b.create<Shape::GetRankedShapeOp>(loc, opOperand.get()));
    }
  }
  auto otherOperands = root.getAssumedNonShapedOperands();
  dispatchOperands.append(otherOperands.begin(), otherOperands.end());
  auto dispatchOp = b.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc, count, root->getResultTypes(), dispatchOperands);
  Region &region = dispatchOp.body();
  Block *block = &region.front();
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(block);
    clonedRoot = cast<linalg::LinalgOp>(b.clone(*root.getOperation()));
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
      b.create<IREE::Flow::DispatchOutputStoreOp>(loc, std::get<0>(it),
                                                  std::get<1>(it), llvm::None,
                                                  llvm::None, llvm::None);
    }
    // TODO(nicolasvasilache): return `clonedRoot->getResults()` once we have
    // shape operands and we drop tie_shape.
    b.create<IREE::Flow::ReturnOp>(loc);
  }
  return dispatchOp;
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

    linalg::LinalgOp clonedLinalgOp;
    Operation *dispatch =
        buildFlowWorkgroupDispatchOp(rewriter, linalgOp, clonedLinalgOp);

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(clonedLinalgOp);
    SmallVector<Value, 4> tensorResults;
    if (failed(Base::matchAndRewriteBase(clonedLinalgOp, rewriter,
                                         tensorResults))) {
      // Need to erase on failure to avoid infinite loop
      rewriter.eraseOp(dispatch);
      return failure();
    }
    rewriter.replaceOp(op, dispatch->getResults());
    rewriter.replaceOp(clonedLinalgOp, tensorResults);

    return success();
  }
};

static Value buildFlowWorkgroupIdOp(OpBuilder &b, unsigned dim) {
  return b.create<IREE::Flow::DispatchWorkgroupIDOp>(
      b.getInsertionPoint()->getLoc(), dim);
}

static Value buildFlowWorkgroupCountOp(OpBuilder &b, unsigned dim) {
  return b.create<IREE::Flow::DispatchWorkgroupCountOp>(
      b.getInsertionPoint()->getLoc(), dim);
}

// After outlining in dispatch region we can rewrite the dispatch ops with
// proper captures.
static void setDispatchWorkgroupOperands(IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  llvm::SetVector<Value> valuesSet;
  mlir::getUsedValuesDefinedAbove(dispatchOp.body(), valuesSet);
  ValueRange valuesDefinedAbove{valuesSet.getArrayRef()};

  // TODO(nicolasvasilache): Adapt if the flow op allows capture.
  SmallVector<Value, 4> originalDispatchOperands = dispatchOp.operands();
  llvm::SetVector<Value> originalDispatchOperandsSet(
      originalDispatchOperands.begin(), originalDispatchOperands.end());
  SmallVector<Value, 4> newOperands, clones;
  for (Value v : valuesDefinedAbove) {
    if (v.getDefiningOp<ConstantOp>()) {
      clones.push_back(v);
    } else if (!originalDispatchOperandsSet.contains(v)) {
      newOperands.push_back(v);
    }
  }
  // Need to insert new operands into operands and BB args at proper
  // positions.
  assert(newOperands.empty() && "unsupported new operands from capture");
  // dispatchOp.body().front().addArguments(ValueRange{newOperands}.getTypes());
  // dispatchOp.setOperands(originalDispatchOperands + newOperands);

  auto getUsesOfValueOutsideOfOp =
      [](Value v, Operation *op) -> SmallVector<Operation *, 4> {
    SmallVector<Operation *, 4> res;
    for (Operation *user : v.getUsers())
      if (!op->isProperAncestor(user)) res.push_back(user);
    return res;
  };

  // Replace operands by BB args.
  // TODO(nicolasvasilache): change BB arg type and use
  // flow.interface.tensor.load/store
  Location loc = dispatchOp.getLoc();
  OpBuilder b = OpBuilder::atBlockBegin(&dispatchOp.body().front());
  for (auto it : llvm::zip(originalDispatchOperands,
                           dispatchOp.body().front().getArguments())) {
    Value v = std::get<0>(it);
    Value bbArg = std::get<1>(it);
    auto uses = getUsesOfValueOutsideOfOp(v, dispatchOp);
    Value repl = bbArg;
    if (bbArg.getType().isa<IREE::Flow::DispatchInputType>()) {
      repl = b.create<IREE::Flow::DispatchInputLoadOp>(loc, v.getType(), bbArg);
    } else if (bbArg.getType().isa<IREE::Flow::DispatchOutputType>()) {
      // TODO(nicolasvasilache): do something useful
      continue;
    }
    v.replaceAllUsesExcept(
        repl, SmallPtrSet<Operation *, 8>(uses.begin(), uses.end()));
  }
  // TODO(nicolasvasilache): add newOperands to op.

  // Clone the constants inside the op.
  for (Value v : clones) {
    auto uses = getUsesOfValueOutsideOfOp(v, dispatchOp);
    v.replaceAllUsesExcept(
        b.clone(*v.getDefiningOp())->getResult(0),
        SmallPtrSet<Operation *, 8>(uses.begin(), uses.end()));
  }
}

void DispatchLinalgOnTensorsPass::runOnOperation() {
  if (tileSizes.empty()) return;

  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp->getContext();
  context->allowUnregisteredDialects(true);

  // Distribution strategy along at most 3 dimensions with WorkgroupIdOp in
  // range [0, WorkgroupSizeOp).
  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();
        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0;
             dim < std::min(numParallelDims, static_cast<size_t>(3)); ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildFlowWorkgroupIdOp(builder, dim),
              buildFlowWorkgroupCountOp(builder, dim)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  OwningRewritePatternList patterns;
  auto linalgTilingOptions =
      linalg::LinalgTilingOptions()
          .setDistributionOptions(workgroupDistributionOptions)
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizes(ArrayRef<int64_t>(tileSizes));
  assert(linalgTilingOptions.distribution.hasValue());

  patterns.insert<TileAndDistributeOnTensorsPattern>(
      linalgTilingOptions,
      // TODO(nicolavasilache): use refactored `getWorkgroupMarker()`
      linalg::LinalgMarker(ArrayRef<Identifier>(),
                           Identifier::get("workgroup", context)));

  // Add canonicalization patterns.
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns, context);
  patterns.insert<linalg::AffineMinSCFCanonicalizationPattern>(context);
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

  // After outlining in dispatch region we can rewrite the dispatch ops with
  // proper captures.
  funcOp.walk(setDispatchWorkgroupOperands);

  // Rewrite destructive updates.
  if (failed(rewriteLinalgDestructiveUpdates(funcOp)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> createDispatchLinalgOnTensorsPass(
    ArrayRef<int64_t> sizes) {
  return std::make_unique<DispatchLinalgOnTensorsPass>(sizes);
}

static PassRegistration<DispatchLinalgOnTensorsPass> pass(
    "iree-flow-dispatch-linalg-on-tensors-pass",
    "Dispatch Linalg operations on tensors by using tile and distribute",
    [] { return std::make_unique<DispatchLinalgOnTensorsPass>(); });

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
