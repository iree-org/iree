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

static constexpr unsigned kMaxWorkgroupRank = 3;

ArrayRef<int64_t> dropTrailingOnes(ArrayRef<int64_t> vector) {
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

/// Updates the workgroup size used for the dispatch region.
LogicalResult updateWorkGroupSize(Operation *op,
                                  ArrayRef<int64_t> workGroupSize) {
  // Need to update both the surrounding FuncOp that has the spv.entry_point_abi
  // attribute, and the hal.executable.
  FuncOp funcOp =
      (isa<FuncOp>(op) ? cast<FuncOp>(op) : op->getParentOfType<FuncOp>());
  MLIRContext *context = op->getContext();
  SmallVector<int32_t, 3> workGroupSizeVec(llvm::map_range(
      workGroupSize,
      [](int64_t value) { return static_cast<int32_t>(value); }));
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

/// Returns the tile size to use for a linalg operation by following
/// `workGroupSize`, if provided, or the default otherwise.
static void getTileSizes(unsigned numParallelLoops,
                         ArrayRef<int64_t> workGroupSize,
                         SmallVectorImpl<int64_t> &tileSizes) {
  tileSizes.clear();
  numParallelLoops = std::min(numParallelLoops, kMaxWorkgroupRank);
  if (!workGroupSize.empty()) {
    workGroupSize = dropTrailingOnes(workGroupSize);
    auto rev = reverse(workGroupSize.take_front(numParallelLoops));
    tileSizes.assign(rev.begin(), rev.end());
    tileSizes.resize(numParallelLoops, 0);
  } else {
    getDefaultTileSizes(numParallelLoops, tileSizes);
  }
  // Linalg convention is to use 0 for no tiling. If the workgroup size is
  // 1, then dont tile along that dimension. So overriding 1 to 0.
  for (auto &tileSize : tileSizes)
    if (tileSize == 1) tileSize = 0;
}

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

/// Tiling pattern which special cases pooling behavior to account for the fake
/// memref case.
template <typename LinalgPoolingOp>
struct TileLinalgPoolingOpPattern
    : public linalg::LinalgTilingPattern<LinalgPoolingOp> {
  TileLinalgPoolingOpPattern(MLIRContext *context,
                             linalg::LinalgTilingOptions options,
                             linalg::LinalgMarker marker,
                             PatternBenefit benefit = 1)
      : linalg::LinalgTilingPattern<LinalgPoolingOp>(context, options, marker,
                                                     benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    // Linalg pooling ops has a fake window_dimension memref that has at least
    // two uses. However, it doesn't affect because it is just a fake memref.
    assert(isa<linalg::PoolingMaxOp>(op) || isa<linalg::PoolingMinOp>(op) ||
           isa<linalg::PoolingSumOp>(op));

    auto linalgOp = cast<linalg::LinalgOp>(op);
    auto buffers = linalgOp.getInputsAndOutputBuffers();
    // The second buffer is a fake memref.
    if (!(*buffers.begin()).hasOneUse()) return failure();
    buffers = llvm::drop_begin(buffers, 2);

    // Check that all buffers have uses only in this op. In that case, just
    // tile the operation.
    // TODO(ravishankarm) : Use Linalg dependence graph information to make
    // this decision.
    for (Value buffer : buffers)
      if (!allUsesInOperation(buffer, op)) return failure();
    return linalg::LinalgTilingPattern<LinalgPoolingOp>::matchAndRewrite(
        linalgOp, rewriter);
  }
};
}  // namespace

void LinalgTileAndFusePass::runOnFunction() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getFunction();

  Region &body = funcOp.getBody();
  // Only handle single block functions.
  if (body.getBlocks().size() != 1) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }
  Block &block = body.front();
  auto linalgOps = block.getOps<linalg::LinalgOp>();
  if (linalgOps.empty()) return;

  // Compute the minimum number of outer parallel loops across linalg
  // operations. This gives the dimensionality of tiling to be used.
  unsigned numParallelLoops = kMaxWorkgroupRank;
  for (linalg::LinalgOp op : linalgOps)
    numParallelLoops = std::min(numParallelLoops, getNumOuterParallelLoops(op));

  // Get the tile sizes to use for the lowering.
  SmallVector<int64_t, 3> tileSizes;
  getTileSizes(numParallelLoops, workGroupSize, tileSizes);

  LLVM_DEBUG({
    llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";
    llvm::dbgs() << "# parallel loops: " << numParallelLoops;
    llvm::dbgs() << "\nworkgroup sizes: [";
    interleaveComma(workGroupSize, llvm::dbgs());
    llvm::dbgs() << "]\ntile sizes: [";
    interleaveComma(tileSizes, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  OwningRewritePatternList patterns;
  patterns.insert<linalg::LinalgTilingPattern<linalg::ConvOp>,
                  linalg::LinalgTilingPattern<linalg::CopyOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>,
                  linalg::LinalgTilingPattern<linalg::IndexedGenericOp>,
                  linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  TileLinalgPoolingOpPattern<linalg::PoolingMaxOp>,
                  TileLinalgPoolingOpPattern<linalg::PoolingMinOp>,
                  TileLinalgPoolingOpPattern<linalg::PoolingSumOp>>(
      context,
      linalg::LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(
          linalg::LinalgTilingLoopType::ParallelLoops),
      linalg::LinalgMarker(getWorkItemMarker()));
  applyPatternsAndFoldGreedily(getOperation(), patterns);

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
