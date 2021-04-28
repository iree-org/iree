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
// This pass concretizes hal.interface.workgroup ops by replacing them with
// constant values from the chosen tiling and distribution scheme.
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
// This pass performs the conretization in two modes:
//
// 1) Partically static: where have a concrete tiling and distirbution sheme
//    *but not* a full static original problem size (e.g., due to dynamic
//    shapes). Under such circumstances,  we can only replace ops like
//    hal.interface.workgroup.size ops and still need to compute the number
//    of workgroups using symbolic values.
// 2) Fully static: where we have a concrete tiling and distribution scheme
//    *and* the full static original problem size. Under such circumstances,
//    we can fully deduce the number of workgroups to dispatch and replace
//    hal.interface.workgroup.count ops with constant values too.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/LaunchConfig.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
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

static constexpr unsigned kMaxWorkgroupDimCount = 3;

static int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// Returns the root Linalg op that dictates tiling and distribution policy.
static linalg::LinalgOp getRootLinalgOp(FuncOp funcOp,
                                        const SPIRVCodegenOptions &options) {
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  SmallVector<Operation *, 4> tiledLoops;
  if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) return {};

  linalg::Aliases aliases;
  linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
  Optional<LaunchConfig> launchConfigOpt = initGPULaunchConfig(
      funcOp.getContext(), dependenceGraph, options, linalgOps);
  if (!launchConfigOpt) return {};

  LaunchConfig &launchConfig = *launchConfigOpt;
  Operation *rootOp =
      launchConfig.getRootOperation(llvm::to_vector<4>(llvm::map_range(
          linalgOps, [](linalg::LinalgOp op) { return op.getOperation(); })));

  // Clean up internal markers that are set during launch configuration
  // preparation.
  launchConfig.finalize(funcOp);

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
static std::tuple<SmallVector<Type>, SmallVector<Type>> getInputOutputTypes(
    linalg::LinalgOp rootOp) {
  SmallVector<Type> inputTypes, outputTypes;
  for (Value inputBuffer : rootOp.getInputBuffers()) {
    if (auto subviewOp = inputBuffer.getDefiningOp<memref::SubViewOp>()) {
      inputTypes.push_back(subviewOp.getViewSource().getType());
    } else if (auto allocOp = inputBuffer.getDefiningOp<memref::AllocOp>()) {
      inputTypes.push_back(allocOp.getType());
    } else {
      inputTypes.clear();
      break;
    }
  }

  for (Value outputBuffer : rootOp.getOutputBuffers()) {
    auto subviewOp = outputBuffer.getDefiningOp<memref::SubViewOp>();
    if (!subviewOp) {
      outputTypes.clear();
      break;
    }
    outputTypes.push_back(subviewOp.getViewSource().getType());
  }
  return std::make_tuple(std::move(inputTypes), std::move(outputTypes));
}

/// Assuming the given `rootOp` is the tiled root Linalg op, returns the
/// tile sizes for distributing to workgroups and the workgroups size for the
/// generated kernel.
///
/// TODO(antiagainst): This pass can be shared between CPU and GPU. But the
/// following query scopes it to GPU for now.
static LogicalResult getTileSizeAndWorkgroupSize(
    Operation *rootOp, ArrayRef<Type> inputTypes, ArrayRef<Type> outputTypes,
    SmallVector<int64_t, 4> &tileSize, SmallVector<int64_t, 4> &workgroupSize,
    const SPIRVCodegenOptions &options) {
  // Build necesary structures to query the tile sizes for distributing to
  // workgroups.
  linalg::Aliases aliases;
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  auto ops = rootOp->getBlock()->getOps<linalg::LinalgOp>();
  linalgOps.assign(ops.begin(), ops.end());
  linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);

  Optional<LaunchConfig> launchConfig = initGPULaunchConfig(
      rootOp->getContext(), dependenceGraph, options, linalgOps);
  if (!launchConfig) {
    return rootOp->emitError("unable to find launch configuration");
  }

  tileSize = llvm::to_vector<4>(launchConfig->getTileSizes(rootOp, 0));
  workgroupSize = llvm::to_vector<4>(launchConfig->getWorkgroupSize());

  // Clean up internal markers that are set during launch configuration
  // preparation.
  launchConfig->finalize(rootOp->getParentOfType<FuncOp>());

  return success();
}

namespace {
/// Replaces hal.interface.workgroup.count op with the constant value chosen
/// from tiling scheme.
class ConcretizeWorkgroupCountOp final
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupCountOp> {
 public:
  ConcretizeWorkgroupCountOp(MLIRContext *context,
                             ArrayRef<int64_t> numWorkgroups,
                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        numWorkgroups(numWorkgroups.begin(), numWorkgroups.end()) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupCountOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex >= numWorkgroups.size()) return failure();
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, rewriter.getIndexAttr(numWorkgroups[dimIndex]));

    return success();
  }

 private:
  SmallVector<int64_t, 4> numWorkgroups;
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
  RemoveTripOneLoop(MLIRContext *context, ArrayRef<int64_t> workloadSize,
                    ArrayRef<int64_t> tileSize, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadSize(workloadSize.begin(), workloadSize.end()),
        tileSize(tileSize.begin(), tileSize.end()) {}

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

    auto mulOp = op.lowerBound().getDefiningOp<AffineApplyOp>();
    if (!mulOp || mulOp.mapOperands().size() != 2) return failure();

    AffineExpr lhs, rhs;
    bindSymbols(op.getContext(), lhs, rhs);
    auto mulMap = AffineMap::get(0, 2, lhs * rhs);
    if (mulOp.getAffineMap() != mulMap) return failure();

    auto mulLhs = mulOp.mapOperands().front();
    auto mulRhs = mulOp.mapOperands().back();

    auto idOp = mulLhs.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>();
    IntegerAttr multipler;
    if (!idOp || !matchPattern(mulRhs, m_Constant(&multipler)))
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

static void removeOneTripTiledLoops(MLIRContext *context, FuncOp funcOp,
                                    linalg::LinalgOp rootLinalgOp,
                                    ArrayRef<int64_t> halWorkgroupSize) {
  if (rootLinalgOp.getNumOutputs() != 1) return;
  unsigned numParallelDims = getNumOuterParallelLoops(rootLinalgOp);
  unsigned numTiledDims =
      std::min<size_t>(numParallelDims, kMaxWorkgroupDimCount);

  Value untiledOutputOperand = getViewSource(rootLinalgOp.getOutput(0));
  ArrayRef<int64_t> outputShape =
      untiledOutputOperand.getType().cast<ShapedType>().getShape();
  if (outputShape.size() < numParallelDims) return;

  // TODO(ravishankarm, antiagainst): Its pure co-incidence that the
  // workload is derivable from the output shape. There is no requirement
  // for this but is the case for all operations we are interested in.
  auto workloadSize = llvm::to_vector<4>(llvm::reverse(
      outputShape.take_front(numParallelDims).take_back(numTiledDims)));
  if (llvm::any_of(workloadSize, [](int64_t dim) {
        return dim == ShapedType::kDynamicSize;
      })) {
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Queried workload size: ";
    llvm::interleaveComma(workloadSize, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  SmallVector<int64_t, 3> numWorkgroups;
  assert(halWorkgroupSize.size() == workloadSize.size());
  for (auto pair : llvm::zip(workloadSize, halWorkgroupSize)) {
    auto workload = std::get<0>(pair);
    auto size = std::get<1>(pair);
    numWorkgroups.push_back(ceilDiv(workload, size));
  }
  numWorkgroups.resize(kMaxWorkgroupDimCount, 1);
  WorkgroupCountRegionBuilder regionBuilder = [&](OpBuilder &b, Location loc,
                                                  std::array<Value, 3>) {
    std::array<Value, 3> returnValues;
    for (unsigned i = 0; i < kMaxWorkgroupDimCount; ++i) {
      returnValues[i] = b.create<ConstantIndexOp>(loc, numWorkgroups[i]);
    }
    return returnValues;
  };

  OpBuilder builder(context);
  if (failed(defineWorkgroupCountRegion(builder, funcOp, regionBuilder))) {
    return;
  }

  {
    OwningRewritePatternList workgroupCountPatterns(context);
    workgroupCountPatterns.insert<ConcretizeWorkgroupCountOp>(context,
                                                              numWorkgroups);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(workgroupCountPatterns));
  }
  {
    OwningRewritePatternList removeTripOneLoopPatterns(context);
    removeTripOneLoopPatterns.insert<RemoveTripOneLoop>(context, workloadSize,
                                                        halWorkgroupSize);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(removeTripOneLoopPatterns));
  }
}

/// Concretizes hal.interface.workgroup.* ops with constants from the chosen
/// tiling sheme when possible and perform loop canonicalization afterwards.
class ConcretizeTileAmongWorkgroupsPass
    : public PassWrapper<ConcretizeTileAmongWorkgroupsPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
 public:
  ConcretizeTileAmongWorkgroupsPass(const SPIRVCodegenOptions &options)
      : options(options) {}
  ConcretizeTileAmongWorkgroupsPass(
      const ConcretizeTileAmongWorkgroupsPass &that)
      : options(that.options) {
    inlineTripOneLoops = that.inlineTripOneLoops;
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    ModuleOp module = targetOp.getInnerModule();
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!funcOp.isPublic()) continue;
      (void)runOnFunction(funcOp);
    }
  }

 private:
  LogicalResult runOnFunction(FuncOp funcOp) {
    MLIRContext &context = getContext();

    // 1. Get the linalg operations within the function. The callee here
    // successed only for functions with single basic block.
    SmallVector<linalg::LinalgOp> linalgOps;
    SmallVector<Operation *> tiledLoops;
    if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
      return failure();
    }
    // If there are no Linalg ops. Nothing to do. Return.
    if (linalgOps.empty()) return success();

    // 2. Get the launch configuration to use for the function.
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
    Optional<LaunchConfig> launchConfig = initGPULaunchConfig(
        funcOp.getContext(), dependenceGraph, options, linalgOps);
    if (!launchConfig) {
      // Having no config implies that there is nothing to do here. Return
      return success();
    }

    // 3. The root operation determines the tile size to use. This has already
    // been computed by the launch configuration.
    // TODO(ravishankarm): The configuration actually makes sure that all tile
    // sizes for the parallel loops are consistent, but get the root operation
    // for now.
    Operation *rootOp =
        launchConfig->getRootOperation(llvm::to_vector<4>(llvm::map_range(
            linalgOps, [](linalg::LinalgOp op) { return op.getOperation(); })));

    unsigned numParallelDims = getNumOuterParallelLoops(rootOp);
    unsigned numTiledDims =
        std::min<size_t>(numParallelDims, kMaxWorkgroupDimCount);
    ArrayRef<int64_t> tileSizes = launchConfig->getTileSizes(rootOp, 0);
    if (tileSizes.size() < numParallelDims) {
      return rootOp->emitError(
                 "invalid tile size configuration, expected at least as many "
                 "as the number of tiled loops : ")
             << numParallelDims;
    }

    // TODO(ravishankarm): The flow tiling only tiles the inner parallel loops
    // by default. Using the same approach here. This spooky distant shake hand
    // needs to be resolved. Potentially can be made cleaner with use of
    // `linalg.tile` operation.
    tileSizes = tileSizes.take_front(numParallelDims).take_back(numTiledDims);
    if (llvm::any_of(tileSizes, [](int64_t ts) { return ts == 0; })) {
      return rootOp->emitError(
          "unhandled tile size setting of 0 for a loop that was tiled");
    }

    // 4. The hal.workgroup.size is a representation of the tile size. Note that
    // this is not the actual workgroup size used eventually. That is computed
    // by the launch configuration and is set below.
    auto halWorkgroupSize = llvm::to_vector<4>(llvm::reverse(tileSizes));

    LLVM_DEBUG({
      llvm::dbgs() << "Queried tile size: ";
      llvm::interleaveComma(tileSizes, llvm::dbgs());
      llvm::dbgs() << ", HAL workgroup size: ";
      llvm::interleaveComma(halWorkgroupSize, llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    // 4. Materialize the constant values for the hal.workgroup.size along
    // different dimensions.
    if (failed(materializeStaticLaunchInformation(funcOp, halWorkgroupSize))) {
      return funcOp.emitOpError(
          "failed to materialize static launch information");
    }

    // 5. Update the actual workgroup size to use based on launch configuraiton.
    if (failed(updateWorkGroupSize(funcOp, launchConfig->getWorkgroupSize()))) {
      return funcOp.emitOpError("failed to set workgroup size on function");
    }
    launchConfig->finalize(funcOp);

    if (inlineTripOneLoops) {
      removeOneTripTiledLoops(&context, funcOp, cast<linalg::LinalgOp>(rootOp),
                              halWorkgroupSize);
    }

    return success();
  }

 private:
  SPIRVCodegenOptions options;

  // TODO(#5034): Investigate whether there is a better way to prove tileability
  // and canonicalize affine.min ops, without matching against the specific
  // pattern involving loops.
  Option<bool> inlineTripOneLoops{
      *this, "inline-trip-one-loops",
      llvm::cl::desc(
          "Inline a loop's body if it can be proven to just have one trip"),
      llvm::cl::init(true)};
};

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createConcretizeTileAmongWorkgroupsPass(const SPIRVCodegenOptions &options) {
  return std::make_unique<ConcretizeTileAmongWorkgroupsPass>(options);
}

static PassRegistration<ConcretizeTileAmongWorkgroupsPass> pass(
    "iree-spirv-concretize-tile-among-workgroups",
    "Replace hal.interface.workgroup.* ops with constant values from chosen "
    "tiling and distribution scheme",
    [] {
      SPIRVCodegenOptions options = getSPIRVCodegenOptionsFromClOptions();
      return std::make_unique<ConcretizeTileAmongWorkgroupsPass>(options);
    });

}  // namespace iree_compiler
}  // namespace mlir
