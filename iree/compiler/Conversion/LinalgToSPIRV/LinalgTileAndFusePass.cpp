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

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse-buffer"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static constexpr unsigned kMaxWorkgroupRank = 3;

static ArrayRef<int64_t> dropTrailingOnes(ArrayRef<int64_t> vector) {
  if (vector.empty()) return vector;
  auto numTrailingOnes = 0;
  for (unsigned i = vector.size() - 1; i > 0; --i) {
    if (vector[i] != 1) {
      break;
    }
    numTrailingOnes++;
  }
  return vector.drop_back(numTrailingOnes);
}

/// Returns the number of "outer" parallel loops specified in the `linalgOp`.
static unsigned getNumOuterParallelLoops(linalg::LinalgOp linalgOp) {
  if (auto convOp = dyn_cast<linalg::ConvOp>(linalgOp.getOperation())) {
    Optional<DenseIntElementsAttr> padding = convOp.padding();
    if (padding) return convOp.getNumBatchDimensions();
  }
  return linalgOp.iterator_types()
      .getValue()
      .take_while([](Attribute attr) {
        return attr.cast<StringAttr>().getValue() ==
               getParallelIteratorTypeName();
      })
      .size();
}

/// Updates the workgroup size used for the dispatch region.
static LogicalResult updateWorkGroupSize(FuncOp funcOp,
                                         ArrayRef<int64_t> workGroupSize) {
  // Need to update both the surrounding FuncOp that has the spv.entry_point_abi
  // attribute, and the hal.executable.
  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body))
    return funcOp.emitError("unhandled dispatch function with multiple blocks");

  SmallVector<int32_t, 3> workGroupSizeVec = llvm::to_vector<3>(llvm::map_range(
      workGroupSize, [](int64_t v) { return static_cast<int32_t>(v); }));

  // TODO(ravishankarm, antiagainst): We should have at most one scf.parallel
  // op, but that is not the case till the splitting of kernels lands.
  unsigned numParallelLoops = 0;
  auto updateNumParallelLoops = [&numParallelLoops](unsigned nPar) {
    numParallelLoops =
        (!numParallelLoops ? nPar : std::min(numParallelLoops, nPar));
  };
  for (auto parallelLoop : body.front().getOps<scf::ParallelOp>()) {
    updateNumParallelLoops(parallelLoop.getNumLoops());
  }
  // If there are no parallel loops, there might be linalg ops that arent
  // tiled. Use that to get the number of parallel loops.
  for (auto linalgOp : body.front().getOps<linalg::LinalgOp>()) {
    updateNumParallelLoops(getNumOuterParallelLoops(linalgOp));
  }
  workGroupSizeVec.resize(numParallelLoops);
  LLVM_DEBUG({
    llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";
    llvm::dbgs() << "# workgroup sizes at end: [";
    interleaveComma(workGroupSizeVec, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });
  MLIRContext *context = funcOp.getContext();
  workGroupSizeVec.resize(3, 1);
  funcOp.setAttr(spirv::getEntryPointABIAttrName(),
                 spirv::getEntryPointABIAttr(workGroupSizeVec, context));
  return success();
}

/// Returns the tile sizes to use by default based on number of dimension of
/// parallelism.
static void getDefaultTileSizes(unsigned numDims,
                                SmallVectorImpl<int64_t> &tileSizes) {
  tileSizes.clear();
  switch (numDims) {
    case 0:
      return;
    case 1:
      tileSizes.push_back(32);
      return;
    case 2:
      tileSizes.push_back(4);
      tileSizes.push_back(32);
      return;
    default:
      break;
  }
  tileSizes.push_back(2);
  tileSizes.push_back(2);
  tileSizes.push_back(32);
}

/// Returns the tile sizes to use for given list of linalg operations.
static LogicalResult getTileSizesImpl(ArrayRef<linalg::LinalgOp> linalgOps,
                                      SmallVectorImpl<int64_t> &tileSizes) {
  // For now hard-coding some special cases. The working assumption is that the
  // tile sizes will be driven by ops like matmul, convolution, etc.
  unsigned numParallelLoops = kMaxWorkgroupRank;
  for (linalg::LinalgOp op : linalgOps) {
    // If there is no marker on this op (i.e. a marker to prevent tile), add an
    // explicit marker to indicate that the op is to be tiled. Makes subsequent
    // lowering simpler.
    if (!hasMarker(op)) setWorkGroupMarker(op);
    numParallelLoops = std::min(numParallelLoops, getNumOuterParallelLoops(op));
  }
  getDefaultTileSizes(numParallelLoops, tileSizes);
  return success();
}

/// Returns the tile size to use for a linalg operation by following
/// `workGroupSize`, if provided, or the default otherwise.
static LogicalResult getTileSizes(ArrayRef<linalg::LinalgOp> linalgOps,
                                  ArrayRef<int64_t> workGroupSize,
                                  SmallVectorImpl<int64_t> &tileSizes) {
  tileSizes.clear();
  if (!workGroupSize.empty()) {
    workGroupSize = dropTrailingOnes(workGroupSize);
    auto rev = reverse(workGroupSize);
    tileSizes.assign(rev.begin(), rev.end());
  } else if (failed(getTileSizesImpl(linalgOps, tileSizes))) {
    return failure();
  }
  // Linalg convention is to use 0 for no tiling. If the workgroup size is
  // 1, then dont tile along that dimension. So overriding 1 to 0.
  for (auto &tileSize : tileSizes)
    if (tileSize == 1) tileSize = 0;
  return success();
}

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

/// Check that all uses of elements of `values` are within the `operation`.
static bool allUsesInOperation(ArrayRef<Value> values, Operation *operation) {
  for (auto value : values) {
    for (auto &use : value.getUses())
      if (use.getOwner() != operation) return false;
  }
  return true;
}

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, FunctionPass> {
  LinalgTileAndFusePass(ArrayRef<int64_t> workGroupSize = {})
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {}
  void runOnFunction() override;

 private:
  SmallVector<int64_t, 3> workGroupSize;
};

/// Pattern to tile linalg operations if they have the workgroup marker.
template <typename LinalgOp>
struct TileLinalgOpPattern : public linalg::LinalgTilingPattern<LinalgOp> {
  using linalg::LinalgTilingPattern<LinalgOp>::LinalgTilingPattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!hasWorkGroupMarker(op)) return failure();
    if (succeeded(linalg::LinalgTilingPattern<LinalgOp>::matchAndRewrite(
            op, rewriter)))
      return success();
    // Update the marker to map to global invocation ID.
    rewriter.startRootUpdate(op);
    setNoTileMarker(op);
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};
}  // namespace

void LinalgTileAndFusePass::runOnFunction() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getFunction();

  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body.getBlocks())) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }
  Block &block = body.front();
  auto linalgOps = block.getOps<linalg::LinalgOp>();
  if (linalgOps.empty()) return;

  // Get the tile sizes to use for the lowering.
  SmallVector<int64_t, 3> tileSizes;
  SmallVector<linalg::LinalgOp, 1> opsVec(linalgOps.begin(), linalgOps.end());
  if (failed(getTileSizes(opsVec, workGroupSize, tileSizes)))
    return signalPassFailure();

  LLVM_DEBUG({
    llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";
    llvm::dbgs() << "# workgroup sizes at start: [";
    interleaveComma(workGroupSize, llvm::dbgs());
    llvm::dbgs() << "]\ntile sizes: [";
    interleaveComma(tileSizes, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  OwningRewritePatternList tilingPatterns;
  tilingPatterns.insert<TileLinalgOpPattern<linalg::ConvOp>,
                        TileLinalgOpPattern<linalg::CopyOp>,
                        TileLinalgOpPattern<linalg::FillOp>,
                        TileLinalgOpPattern<linalg::GenericOp>,
                        TileLinalgOpPattern<linalg::IndexedGenericOp>,
                        TileLinalgOpPattern<linalg::MatmulOp>,
                        TileLinalgOpPattern<linalg::PoolingMaxOp>,
                        TileLinalgOpPattern<linalg::PoolingMinOp>,
                        TileLinalgOpPattern<linalg::PoolingSumOp>>(
      context,
      linalg::LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(
          linalg::LinalgTilingLoopType::ParallelLoops),
      linalg::LinalgMarker(getWorkGroupMarker(), getWorkItemMarker()));
  applyPatternsAndFoldGreedily(getOperation(), tilingPatterns);

  // Update the workgroup size to be consistent with the tile sizes used. Note
  // the tile sizes are ordered from outer most to inner most loops. The
  // heuristic is to map the inner loops to x, the next outer (if it exists) to
  // y, and the next outer (if it exists) to z. So tile sizes are reversed to
  // get the workgroup size.
  SmallVector<int64_t, 3> updatedWorkGroupSize(reverse(tileSizes));
  updatedWorkGroupSize.resize(3, 1);
  if (failed(updateWorkGroupSize(funcOp, updatedWorkGroupSize)))
    return signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createLinalgTileAndFusePass(
    ArrayRef<int64_t> workGroupSize) {
  return std::make_unique<LinalgTileAndFusePass>(workGroupSize);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
