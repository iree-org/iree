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
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace iree_compiler {

static StringRef getWorkGroupMarker() { return "spirv_workgroup"; }

static constexpr unsigned kMaxWorkgroupRank = 3;

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

/// Checks if an operation already has an attribute with this marker. If set it
/// implies this op shouldnt be tiled with the same marker.
static bool hasMarker(Operation *op) {
  auto tilingAttr = op->getAttrOfType<StringAttr>(
      linalg::LinalgTransforms::kLinalgTransformMarker);
  return tilingAttr != nullptr;
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

/// Base class for Linalg tiling patterns. All classes that derive from this
/// need to implement an apply method that will tile the operation with the
/// following signature.
///
/// LogicalResult apply(LinalgOp op, SmallVectorImpl<int64_t> &tileSizes,
///                     PatternRewriter &rewriter) const
template <typename DerivedClass, typename LinalgOp>
struct LinalgTilingPattern : public OpRewritePattern<LinalgOp> {
  LinalgTilingPattern(MLIRContext *context, ArrayRef<int64_t> tileSizes,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<LinalgOp>(context, benefit), tileSizes(tileSizes) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics()) return failure();
    // Currently we are only doing one-level tiling, so a single marker is
    // enough. This might need to move into derived classes.
    if (hasMarker(linalgOp.getOperation())) return failure();

    if (failed(static_cast<const DerivedClass *>(this)->apply(
            linalgOp, tileSizes, rewriter)))
      return failure();
    rewriter.eraseOp(linalgOp);
    return success();
  }

 private:
  ArrayRef<int64_t> tileSizes;
};

/// If there is nothing to fuse the linalg op with, then just tiles it.
template <typename LinalgOp>
struct TileLinalgOpPattern
    : public LinalgTilingPattern<TileLinalgOpPattern<LinalgOp>, LinalgOp> {
  using LinalgTilingPattern<TileLinalgOpPattern<LinalgOp>,
                            LinalgOp>::LinalgTilingPattern;
  LogicalResult apply(LinalgOp linalgOp, ArrayRef<int64_t> tileSizes,
                      PatternRewriter &rewriter) const {
    // Check that all input and outputs have a single use (this op). In that
    // case, there is nothing to tile and fuse with. So just tile it.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp.getOperation());
    if (!llvm::all_of(linalgOp.getInputsAndOutputBuffers(),
                      [](Value arg) { return arg.hasOneUse(); }))
      return failure();
    return linalg::tileLinalgOpToParallelLoopsAndSetMarker(
        rewriter, linalgOp.getOperation(), tileSizes, getWorkItemMarker(),
        /*permutation=*/{});
  }
};

/// Tile and fuse linalg operations.
template <typename LinalgOp>
struct TileAndFuseLinalgOpPattern
    : public LinalgTilingPattern<TileAndFuseLinalgOpPattern<LinalgOp>,
                                 LinalgOp> {
  using LinalgTilingPattern<TileAndFuseLinalgOpPattern<LinalgOp>,
                            LinalgOp>::LinalgTilingPattern;
  LogicalResult apply(LinalgOp linalgOp, ArrayRef<int64_t> tileSizes,
                      PatternRewriter &rewriter) const {
    SmallVector<int64_t, 1> operandIndicesToFuse;
    for (auto buffer : llvm::enumerate(linalgOp.getInputsAndOutputBuffers())) {
      // If a buffer has multiple uses, then it is a candidate for fusion.
      if (!buffer.value().hasOneUse())
        operandIndicesToFuse.push_back(buffer.index());
    }
    if (operandIndicesToFuse.empty()) return failure();
    return linalg::tileAndFuseLinalgOpToParallelLoopsAndSetMarker(
        rewriter, linalgOp, tileSizes, operandIndicesToFuse,
        getWorkItemMarker());
  }
};
}  // namespace

void LinalgTileAndFusePass::runOnFunction() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getFunction();
  if (!isDispatchFuncImpl(funcOp)) return;

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
  // operations. This gives the dimensionality of tiling to be used .
  unsigned numParallelLoops = kMaxWorkgroupRank;
  for (linalg::LinalgOp op : linalgOps)
    numParallelLoops = std::min(numParallelLoops, getNumOuterParallelLoops(op));

  // Get the tile sizes to use for the lowering.
  SmallVector<int64_t, 3> tileSizes;
  getTileSizes(numParallelLoops, workGroupSize, tileSizes);

  OwningRewritePatternList patterns;
  patterns.insert<TileLinalgOpPattern<linalg::GenericOp>,
                  TileLinalgOpPattern<linalg::IndexedGenericOp>,
                  TileLinalgOpPattern<linalg::MatmulOp>,
                  TileLinalgOpPattern<linalg::ConvOp>,
                  TileAndFuseLinalgOpPattern<linalg::GenericOp>>(context,
                                                                 tileSizes);
  applyPatternsAndFoldGreedily(getOperation(), patterns);

  // Check that there are single loop.parallel operation at the top most level
  // that will get mapped to thread blocks/workgroups.
  auto forLoops = block.getOps<loop::ParallelOp>();
  if (numParallelLoops > 0 && !llvm::hasSingleElement(forLoops) &&
      (*forLoops.begin()).getNumLoops() != numParallelLoops) {
    funcOp.emitError(
        "unable to generate the tiled loop structure to map to workgroups");
    return signalPassFailure();
  }

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
    "iree-linalg-tile-and-fuse", "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });
}  // namespace iree_compiler
}  // namespace mlir
