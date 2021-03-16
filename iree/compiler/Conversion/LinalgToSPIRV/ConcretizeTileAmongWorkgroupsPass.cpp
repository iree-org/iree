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

namespace {

constexpr unsigned kWorkgroupDimCount = 3;

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static size_t getNumOuterParallelDims(linalg::LinalgOp op) {
  ArrayRef<Attribute> iterators = op.iterator_types().getValue();
  auto parallels = iterators.take_while(
      [](Attribute attr) { return linalg::isParallelIteratorType(attr); });
  return parallels.size();
}

/// Returns the root Linalg op that dictates tiling and distribution policy.
linalg::LinalgOp getRootLinalgOp(FuncOp funcOp) {
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  SmallVector<Operation *, 4> tiledLoops;
  if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) return {};

  SPIRVCodegenOptions options;
  options.enableVectorization = true;
  options.usingLinalgOnTensors = true;

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
LogicalResult getInputOutputTypesForAllTiles(
    linalg::LinalgOp rootOp, SmallVectorImpl<Type> &inputTypes,
    SmallVectorImpl<Type> &outputTypes) {
  for (Value inputBuffer : rootOp.getInputBuffers()) {
    auto subviewOp = inputBuffer.getDefiningOp<SubViewOp>();
    if (!subviewOp) return failure();
    inputTypes.push_back(subviewOp.getViewSource().getType());
  }

  for (Value outputBuffer : rootOp.getOutputBuffers()) {
    auto subviewOp = outputBuffer.getDefiningOp<SubViewOp>();
    if (!subviewOp) return failure();
    outputTypes.push_back(subviewOp.getViewSource().getType());
  }

  return success();
}

/// Assuming the given `rootOp` is the tiled root Linalg op, returns the
/// tile sizes for distributing to workgroups and the workgroups size for the
/// generated kernel.
///
/// TODO(antiagainst): This pass can be shared between CPU and GPU. But the
/// following query scopes it to GPU for now.
llvm::Optional<
    std::pair<llvm::SmallVector<int64_t, 4>, llvm::SmallVector<int64_t, 4>>>
getTileSizeAndWorkgroupSize(Operation *rootOp, ArrayRef<Type> inputTypes,
                            ArrayRef<Type> outputTypes) {
  // Build necesary structures to query the tile sizes for distributing to
  // workgroups.
  linalg::Aliases aliases;
  SmallVector<linalg::LinalgOp, 4> linalgOps;
  auto ops = rootOp->getBlock()->getOps<linalg::LinalgOp>();
  linalgOps.assign(ops.begin(), ops.end());
  linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);

  SPIRVCodegenOptions options;
  options.enableVectorization = true;
  options.usingLinalgOnTensors = true;

  // NOTE: Launch configuration expects the original input/output type to decide
  // the configuration. But we have already tiled the Linalg ops here. Use an
  // attribute to send it over for now.
  const char inputTypeAttrName[] = "iree.codegen.original_input_types";
  const char outputTypeAttrName[] = "iree.codegen.original_output_types";
  if (!inputTypes.empty()) {
    rootOp->setAttr(inputTypeAttrName,
                    Builder(rootOp).getTypeArrayAttr(inputTypes));
  }
  if (!outputTypes.empty()) {
    rootOp->setAttr(outputTypeAttrName,
                    Builder(rootOp).getTypeArrayAttr(outputTypes));
  }

  Optional<LaunchConfig> launchConfig = initGPULaunchConfig(
      rootOp->getContext(), dependenceGraph, options, linalgOps);
  if (!launchConfig) {
    rootOp->emitError("unable to find launch configuration");
    return llvm::None;
  }

  ArrayRef<int64_t> tileSize = launchConfig->getTileSizes(rootOp, 0);
  ArrayRef<int64_t> workgroupSize = launchConfig->getWorkgroupSize();

  // Clean up internal markers that are set during launch configuration
  // preparation.
  launchConfig->finalize(rootOp->getParentOfType<FuncOp>());

  return std::make_pair(llvm::to_vector<4>(tileSize),
                        llvm::to_vector<4>(workgroupSize));
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

    if (dimIndex < kWorkgroupDimCount && tileSize[dimIndex] != 0) {
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
        workloadSize(std::move(workloadSize)),
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
        workloadSize(std::move(workloadSize)),
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
      if (failed(runOnFunction(funcOp))) return signalPassFailure();
    }
  }

 private:
  LogicalResult runOnFunction(FuncOp funcOp) {
    MLIRContext &context = getContext();

    // 1. Get the root op first. We need it to figure out the original problem
    // size, which then affects the tiling and distribution policy.

    linalg::LinalgOp rootOp = getRootLinalgOp(funcOp);
    if (!rootOp) {
      LLVM_DEBUG(llvm::dbgs() << "unable to find root Linalg op\n");
      // It can happen for ops that are not abstractly tiled during dispatch
      // region formation. So don't trigger pass failure.
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "Root op: " << rootOp << "\n");

    size_t numTilableDims = getNumOuterParallelDims(rootOp);

    // 2. Figure out the original problem size.

    SmallVector<Type, 4> inputTypes, outputTypes;
    SmallVector<int64_t, 4> workloadSize;
    if (succeeded(
            getInputOutputTypesForAllTiles(rootOp, inputTypes, outputTypes))) {
      if (outputTypes.size() != 1) {
        return rootOp.emitError("only support ops with one result right now");
      }

      // Flow/HAL processor id/size/count ops' indices follow the reverse order
      // of the shape dimensions.
      workloadSize = llvm::to_vector<4>(llvm::reverse(
          outputTypes.front().cast<ShapedType>().getShape().take_front(
              numTilableDims)));
    } else {
      // This can happen for dynamic shapes.
      LLVM_DEBUG(llvm::dbgs()
                 << "unable to find input/output type for all tiles");

      inputTypes.clear();
      outputTypes.clear();

      workloadSize.assign(numTilableDims, ShapedType::kDynamicSize);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Queried workload size: ";
      llvm::interleaveComma(workloadSize, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // 3. Query the scheme for tiling among workgroups.

    SmallVector<int64_t, 4> tileSize;
    SmallVector<int64_t, 4> workgroupSize;

    // Try to use configuration from the command-line first for testing.
    tileSize.assign(options.tileSizes.begin(), options.tileSizes.end());
    tileSize.resize(numTilableDims, 0);
    workgroupSize.assign(options.workgroupSize.begin(),
                         options.workgroupSize.end());
    if (tileSize.empty() || workgroupSize.empty()) {
      auto sizes = getTileSizeAndWorkgroupSize(rootOp, inputTypes, outputTypes);
      if (sizes) {
        // The tile sizes are specified against the original dimension order of
        // the workload shape. But Flow/HAL processor id/size/count ops' are
        // created using the reverse order.
        tileSize = sizes->first;
        tileSize.resize(numTilableDims);
        tileSize = llvm::to_vector<4>(llvm::reverse(tileSize));
        workgroupSize = sizes->second;
      } else {
        return funcOp.emitError("failed to query tile size and workgroup size");
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Queried tile size: ";
      llvm::interleaveComma(tileSize, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // 4. Replace hal.interface.workgroup symbolic ops with constant values.

    {
      OwningRewritePatternList patterns;
      patterns.insert<ConcretizeWorkgroupSizeOp, ConcretizeWorkgroupCountOp>(
          &context, workloadSize, tileSize);

      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    LLVM_DEBUG({
      llvm::dbgs()
          << "--- After concretizing hal.interface.workgroup ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // 5. Set the entry point region for computing the number of workgroups
    // to dispatch. The region has symbolic arguments representing the workload.
    // So two modes here (see comments at the begining of this file).

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
      // workgroups directly. Otherwise, we need to generate the IR for
      // computing it using symbolic values.
      if (llvm::none_of(numWorkgroups, [](int64_t dim) {
            return dim == ShapedType::kDynamicSize;
          })) {
        OpBuilder builder(&context);
        WorkgroupCountRegionBuilder regionBuilder =
            [&](OpBuilder &builder, Location loc, std::array<Value, 3>) {
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
      } else {
        if (failed(materializeStaticLaunchInformation(funcOp, tileSize))) {
          return funcOp.emitOpError(
              "failed to materialize static launch information");
        }
      }
    }

    if (failed(updateWorkGroupSize(funcOp, workgroupSize))) {
      return funcOp.emitOpError("failed to set workgroup size on function");
    }

    // 6. Canonicalization and clean up.

    if (inlineTripOneLoops) {
      OwningRewritePatternList patterns;
      patterns.insert<RemoveTripOneLoop>(&context, workloadSize, tileSize);

      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
