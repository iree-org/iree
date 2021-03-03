// Copyright 2021 Google LLC
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

//===- ConcretizeTileAmongWorkgroupsPass.cpp ------------------------------===//
//
// This pass concretizes hal.interface.workgroup.* ops by replacing them with
// chosen constant values.
//
// During dispatch region formation in IREE Flow transformations, ops are tiled
// and distributed in an abstract way by using symbolic hal.interface.workgroup
// ops. That is because the same source region is compiled towards different
// target backends and each target backend could use different tiling and
// distribution schemes. However, after HAL interface materialization, the
// hal.executable.target is just meant for one target backend. We need to
// concretize the tiling and distribution in order to inject static information
// for further compilation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-concretize-tile-among-workgroups"

namespace mlir {
namespace iree_compiler {

namespace {

constexpr unsigned kWorkgroupDimCount = 3;

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static size_t getNumOuterParallelDims(linalg::LinalgOp op) {
  ArrayRef<Attribute> iterators = op.iterator_types().getValue();
  auto parallels = iterators.take_while(
      [](Attribute attr) { return linalg::isParallelIteratorType(attr); });
  return parallels.size();
}

/// Returns the root Linalg op that is used as the anchor for dispatch region
/// formation.
linalg::LinalgOp getRootLinalgOp(FuncOp funcOp) {
  linalg::LinalgOp rootOp;
  funcOp.walk([&rootOp](linalg::LinalgOp op) {
    if (op.getOperation()->hasAttr("iree.codegen.fushion.root_op")) {
      rootOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return rootOp;
}

/// Assuming the given `rootOp` is the tiled root Linalg op, returns the
/// original input/output types for all tiles.
///
/// Note: After the abstract tiling and distribution in Flow dispatch region
/// creation, the anchoring root op is already in a loop nest and works on a
/// tile. The full type for all tiles in the IR is not explicit anymore after
/// HAL interface is materialized. So go through the IR use chain to figure it
/// out. Otherwise we need to make even more assumptions in the following.
// TODO(antiagainst): This is quite fragile. We need a better way to pass the
// information down from the upper layer, which readily has it. Probably via
// linalg.tile op.
LogicalResult getInputOutputTypesForAllTiles(
    Operation *rootOp, SmallVectorImpl<Type> &inputTypes,
    SmallVectorImpl<Type> &outputTypes) {
  // There are operands providing shape for the results. Ignore them.
  for (auto operand :
       rootOp->getOperands().drop_back(rootOp->getNumResults())) {
    Operation *op = operand.getDefiningOp();
    if (auto subtensor = dyn_cast<SubTensorOp>(op)) {
      op = subtensor.source().getDefiningOp();
    }
    if (auto loadOp = dyn_cast<IREE::Flow::DispatchInputLoadOp>(op)) {
      auto type =
          loadOp.source().getType().cast<IREE::Flow::DispatchInputType>();
      inputTypes.push_back(
          RankedTensorType::get(type.getShape(), type.getElementType()));
    } else if (auto reshapeOp = dyn_cast<linalg::TensorReshapeOp>(op)) {
      inputTypes.push_back(reshapeOp.getResultType());
    } else {
      return failure();
    }
  }

  for (auto result : rootOp->getResults()) {
    auto uses = result.getUses();
    if (++uses.begin() != uses.end()) return failure();
    auto storeOp =
        dyn_cast<IREE::Flow::DispatchOutputStoreOp>(uses.begin()->getOwner());
    if (!storeOp) return failure();

    auto type =
        storeOp.target().getType().dyn_cast<IREE::Flow::DispatchOutputType>();
    if (!type) return failure();
    outputTypes.push_back(
        RankedTensorType::get(type.getShape(), type.getElementType()));
  }

  return success();
}

/// Assuming the given `rootOp` is the tiled root Linalg op, returns the
/// tile sizes for distributing to workgroups.
///
/// TODO(antiagainst): This pass can be shared between CPU and GPU. But the
/// following query scopes it to GPU for now.
llvm::Optional<ArrayRef<int64_t>> getTileSize(Operation *rootOp,
                                              ArrayRef<Type> inputTypes,
                                              ArrayRef<Type> outputTypes) {
  // Build necesary structures to query the tile sizes for distributing to
  // workgroups.
  linalg::Aliases aliases;
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  auto ops = rootOp->getBlock()->getOps<linalg::LinalgOp>();
  linalgOps.assign(ops.begin(), ops.end());
  linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
  SPIRVCodegenOptions options;

  // XXX: Launch configuration expects the original input/output type to decide
  // the configuration. But we have already tiled the Linalg ops here. Use an
  // attribute to send it over for now.
  const char inputTypeAttrName[] = "iree.codegen.original_input_types";
  const char outputTypeAttrName[] = "iree.codegen.original_output_types";
  rootOp->setAttr(inputTypeAttrName,
                  Builder(rootOp).getTypeArrayAttr(inputTypes));
  rootOp->setAttr(outputTypeAttrName,
                  Builder(rootOp).getTypeArrayAttr(outputTypes));

  Optional<LaunchConfig> launchConfig = initGPULaunchConfig(
      rootOp->getContext(), dependenceGraph, options, linalgOps);
  if (!launchConfig) {
    rootOp->emitError("unable to find launch configuration");
    return llvm::None;
  }

  ArrayRef<int64_t> tileSize = launchConfig->getTileSizes(rootOp, 0);

  // Clean up internal markers that are set during launch configuration
  // preparation.
  launchConfig->finalize(rootOp->getParentOfType<FuncOp>());

  return tileSize;
}

/// Replaces hal.interface.workgroup.size op with the constant value chosen
/// from tiling scheme.
class ConcretizeWorkgroupSizeOp final
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  ConcretizeWorkgroupSizeOp(MLIRContext *context,
                            SmallVector<int64_t, 4> workloadSize,
                            SmallVector<int64_t, 4> tileSize,
                            PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(std::move(workloadSize)),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex < kWorkgroupDimCount) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, rewriter.getIndexAttr(tileSize[dimIndex]));
      return success();
    }

    return failure();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

/// Replaces hal.interface.workgroup.count op with the constant value chosen
/// from tiling scheme.
class ConcretizeWorkgroupCountOp final
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupCountOp> {
 public:
  ConcretizeWorkgroupCountOp(MLIRContext *context,
                             SmallVector<int64_t, 4> workloadSize,
                             SmallVector<int64_t, 4> tileSize,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(workloadSize),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupCountOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex >= kWorkgroupDimCount) return failure();

    int64_t dimSize = workloadSize[dimIndex];
    int64_t dimTile = tileSize[dimIndex];

    if (dimSize == ShapedType::kDynamicSize || dimTile == 0) return failure();

    int64_t count = ceilDiv(dimSize, dimTile);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, rewriter.getIndexAttr(count));

    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

// Canonicalizes away a trip-one scf.for loop by inlining its body and removing
// the loop.
//
// This pattern is needed because in Flow abstract tiling and distribution we
// will create scf.for loops that distribute workload cyclically. After
// concretizing hal.interface.workgroup.* ops, these scf.for loops still remain,
// and they will be of the form:
//
//   %lb = mul %workgroup_id_{x|y|z}, %cst_tile_size_{x|y|z}
//   scf.for %iv = %lb to %cst_wokload_size_{x|y|z}
//                 step %cst_workload_size_{x|y|z} { ... }
//
// Such scf.for loops can be inlined if %lb is smaller than upper bound.
class RemoveTripOneLoop final : public OpRewritePattern<scf::ForOp> {
 public:
  RemoveTripOneLoop(MLIRContext *context, SmallVector<int64_t, 4> workloadSize,
                    SmallVector<int64_t, 4> tileSize,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(workloadSize),
        tileSize(std::move(tileSize)) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // Get constant upper bound and step values.
    IntegerAttr ub, step;
    if (!matchPattern(op.upperBound(), m_Constant(&ub)) ||
        !matchPattern(op.step(), m_Constant(&step))) {
      return failure();
    }

    // Require that they are the same.
    if (ub != step) return failure();

    // Now make sure the lower bound is smaller than upper bound. The lower
    // bound should be multiplying the workgroup ID with some constant.
    auto mulOp = op.lowerBound().getDefiningOp<MulIOp>();
    if (!mulOp) return failure();

    auto idOp = mulOp.lhs().getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>();
    IntegerAttr multipler;
    if (!idOp || !matchPattern(mulOp.rhs(), m_Constant(&multipler)))
      return failure();

    // We just need to make sure the max value of the workgroup ID multipled by
    // the multipler is smaller than the upper bound to guarantee one trip.
    unsigned dimIndex = idOp.dimension().getZExtValue();
    int64_t dimSize = workloadSize[dimIndex];
    int64_t dimTile = tileSize[dimIndex];

    if (dimSize == ShapedType::kDynamicSize) return failure();

    int64_t count = ceilDiv(dimSize, dimTile);
    assert(count > 0 && "expected at least one tile!");

    // ID should be in range [0, count).
    if ((count - 1) * multipler.getInt() >= ub.getInt()) {
      // Dead loop. It can actually be removed entirely. But we aren't expecting
      // it to happen here. Do not canonicalize for such case.
      return failure();
    }

    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.lowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());

    Block *block = &op.getLoopBody().front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(block, op, blockArgs);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);

    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadSize;
  SmallVector<int64_t, 4> tileSize;
};

/// Concretizes hal.interface.workgroup.* ops with constants from the chosen
/// tiling sheme when possible and perform loop canonicalization afterwards.
struct ConcretizeTileAmongWorkgroupsPass
    : public PassWrapper<ConcretizeTileAmongWorkgroupsPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  void runOnOperation() override {
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    ModuleOp module = targetOp.getInnerModule();

    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!funcOp.isPublic()) return;
      if (failed(runOnFunction(funcOp))) return signalPassFailure();
    }
  }

  LogicalResult runOnFunction(FuncOp funcOp) {
    MLIRContext &context = getContext();

    linalg::LinalgOp rootOp = getRootLinalgOp(funcOp);
    if (!rootOp) {
      LLVM_DEBUG(llvm::dbgs() << "unable to find root Linalg op\n");
      // It can happen for ops that are not abstractly tiled during dispatch
      // region formation. So don't trigger pass failure.
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "Root op: " << rootOp << "\n");

    size_t numTilableDims = getNumOuterParallelDims(rootOp);

    SmallVector<Type, 4> inputTypes, outputTypes;
    if (failed(
            getInputOutputTypesForAllTiles(rootOp, inputTypes, outputTypes))) {
      return rootOp.emitError("unable to find input/output type for all tiles");
    }

    if (outputTypes.size() != 1) {
      return rootOp.emitError("only support ops with one result right now");
    }

    // Flow/HAL processor id/size/count ops' indices follow the reverse order of
    // the shape dimensions.
    auto workloadSize = llvm::to_vector<4>(llvm::reverse(
        outputTypes.front().cast<ShapedType>().getShape().take_front(
            numTilableDims)));

    LLVM_DEBUG({
      llvm::dbgs() << "Queried workload size: ";
      llvm::interleaveComma(workloadSize, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    SmallVector<int64_t, 4> tileSize;
    if (auto sizes = getTileSize(rootOp, inputTypes, outputTypes)) {
      // The tile sizes are specified against the original dimension order of
      // the workload shape. But Flow/HAL processor id/size/count ops' are
      // created using the reverse order.
      tileSize =
          llvm::to_vector<4>(llvm::reverse(sizes->take_front(numTilableDims)));
    } else {
      return funcOp.emitError("failed to query tile size");
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Queried tile size: ";
      llvm::interleaveComma(tileSize, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    {
      OwningRewritePatternList patterns;
      patterns.insert<ConcretizeWorkgroupSizeOp, ConcretizeWorkgroupCountOp>(
          &context, workloadSize, tileSize);

      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    LLVM_DEBUG({
      llvm::dbgs()
          << "--- After concretizing hal.interface.workgroup.* ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      SmallVector<int64_t, 3> numWorkgroups;
      for (auto pair : llvm::zip(workloadSize, tileSize)) {
        auto workload = std::get<0>(pair);
        auto tile = std::get<1>(pair);
        if (workload == ShapedType::kDynamicSize || tile == 0) {
          numWorkgroups.push_back(ShapedType::kDynamicSize);
        } else {
          numWorkgroups.push_back(ceilDiv(workload, tile));
        }
      }

      numWorkgroups.resize(kWorkgroupDimCount, 1);

      // If all dimensions are known constant, then we can set the number of
      // workgroups directly.
      if (llvm::none_of(numWorkgroups, [](int64_t dim) {
            return dim == ShapedType::kDynamicSize;
          })) {
        OpBuilder builder(&context);
        WorkgroupCountRegionBuilder regionBuilder =
            [&](OpBuilder &builder, Location loc, std::array<Value, 3>) {
              Value one = builder.create<ConstantIndexOp>(loc, 1);
              std::array<Value, 3> returnValues;
              for (unsigned i = 0; i < kWorkgroupDimCount; ++i) {
                returnValues[i] =
                    builder.create<ConstantIndexOp>(loc, numWorkgroups[i]);
              }
              return returnValues;
            };
        if (failed(
                defineWorkgroupCountRegion(builder, funcOp, regionBuilder))) {
          return funcOp.emitError(
              "failed to set entry point region for number of workgroups");
        }
      }
    }

    {
      OwningRewritePatternList patterns;
      patterns.insert<RemoveTripOneLoop>(&context, workloadSize, tileSize);

      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    return success();
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createConcretizeTileAmongWorkgroupsPass() {
  return std::make_unique<ConcretizeTileAmongWorkgroupsPass>();
}

static PassRegistration<ConcretizeTileAmongWorkgroupsPass> pass(
    "iree-spirv-concretize-tile-among-workgroups",
    "Replace hal.interface.workgroup.* ops with constant values from chosen "
    "tiling and distribution scheme",
    [] { return std::make_unique<ConcretizeTileAmongWorkgroupsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
