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
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse-buffer"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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

/// Returns true if the linalg op has padding attribute, and that it has
/// non-zero entries.
template <typename OpTy>
static bool hasPadding(OpTy op) {
  Optional<DenseIntElementsAttr> padding = op.padding();
  if (!padding) return false;
  return llvm::any_of(padding.getValue(),
                      [](APInt v) -> bool { return !v.isNullValue(); });
}

namespace {

/// Computes tile sizes (and workgroup size) to use based on operations within
/// the function, and resource constraints on the module.
class TileSizeCalculator {
 public:
  TileSizeCalculator(FuncOp funcOp)
      : resourceLimits(spirv::lookupTargetEnv(funcOp).getResourceLimits()) {
    if (DenseIntElementsAttr attr = spirv::lookupLocalWorkGroupSize(funcOp)) {
      for (auto val : attr.getValues<APInt>())
        workgroupSize.push_back(val.getSExtValue());
    }
    workgroupSize.resize(3, 1);
  }

  /// Set tile sizes to use.
  void setTileSizes(ArrayRef<int64_t> sizes) {
    tileSizes.assign(sizes.begin(), sizes.end());
  }

  /// Set workgroup size to use.
  void setWorkgroupSize(ArrayRef<int64_t> sizes) {
    workgroupSize.assign(sizes.begin(), sizes.end());
  }

  /// Compute the tile sizes based on the Linalg Ops within the dispatch region.
  LogicalResult inferTileAndWorkgroupSize(ArrayRef<linalg::LinalgOp> linalgOps);

  /// Get the current tile size computed.
  ArrayRef<int64_t> getTileSizes() const { return tileSizes; }

  /// Returns the workgroup size to use based on the tile sizes.
  ArrayRef<int64_t> getWorkgroupSize() const { return workgroupSize; }

 private:
  /// Current tile size configuration.
  SmallVector<int64_t, 4> tileSizes;

  /// Workgroup size to use.
  SmallVector<int64_t, 3> workgroupSize;

  /// Attribute for device constraints.
  spirv::ResourceLimitsAttr resourceLimits;
};
}  // namespace

LogicalResult TileSizeCalculator::inferTileAndWorkgroupSize(
    ArrayRef<linalg::LinalgOp> linalgOps) {
  tileSizes.clear();
  if (linalgOps.empty()) {
    tileSizes = {1, 1, 1};
    workgroupSize = {1, 1, 1};
    return success();
  }
  // The tile size will be driven by operations like matmul, conv, etc. within
  // the list. So see what operation exists in the list to decide the tile size.
  // If there are two such operations in the list, return error.
  enum OpInfo : uint32_t {
    None = 0x0,
    Convolution = 0x1,
    Matmul = 0x2,
    Pooling = 0x4,
    BatchMatmul = 0x8,
  };
  uint32_t opInfo = OpInfo::None;
  for (linalg::LinalgOp linalgOp : linalgOps) {
    Operation *op = linalgOp.getOperation();
    if (isa<linalg::ConvOp>(op))
      opInfo |= OpInfo::Convolution;
    else if (isa<linalg::MatmulOp>(op))
      opInfo |= OpInfo::Matmul;
    else if (isa<linalg::BatchMatmulOp>(op))
      opInfo |= OpInfo::BatchMatmul;
    else if (isa<linalg::PoolingMaxOp>(op))
      opInfo |= OpInfo::Pooling;
    else if (isa<linalg::PoolingMinOp>(op))
      opInfo |= OpInfo::Pooling;
    else if (isa<linalg::PoolingSumOp>(op))
      opInfo |= OpInfo::Pooling;
  }
  // If there are no tilable ops, there is nothing to do here.
  if (!opInfo) return success();

  Operation *linalgOp = *(linalgOps.begin());
  if (llvm::countPopulation(opInfo) != 1)
    return linalgOp->getParentOfType<FuncOp>().emitError(
        "unhandled fusion of ops in dispatch function");

  // TODO(ravishanarm, antiagainst): Only the maximum workgroup size is used
  // here for computing tile sizes. In reality we also need the maximum
  // workgroup memory size available (per workgroup) to compute the tile sizes
  // effectively.
  unsigned maxWorkgroupSize =
      resourceLimits.max_compute_workgroup_invocations().getInt();
  if (opInfo & OpInfo::Convolution) {
    int64_t tileSizeX = 32;
    int64_t tileSizeY = maxWorkgroupSize / 32;
    tileSizes = {1, tileSizeY, tileSizeX};
    workgroupSize = {tileSizeX, tileSizeY, 1};
    return success();
  }
  if (opInfo & OpInfo::Matmul) {
    // TODO: For now just hard wire this, but we can do better.
    tileSizes = {8, 8, 4};
    workgroupSize = {8, 8, 1};
    return success();
  }
  if (opInfo & OpInfo::BatchMatmul) {
    tileSizes = {2, 8, 8, 4};
    workgroupSize = {8, 8, 2};
    return success();
  }
  if (opInfo & OpInfo::Pooling) {
    int64_t tileSizeX = 32;
    int64_t tileSizeY = maxWorkgroupSize / 32;
    tileSizes = {tileSizeY, tileSizeX};
    workgroupSize = {tileSizeX, tileSizeY, 1};
    return success();
  }
  return linalgOp->getParentOfType<FuncOp>().emitError(
      "unable to find tile size for ops in this dispatch function");
}

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, FunctionPass> {
  LinalgTileAndFusePass(ArrayRef<int64_t> workgroupSize = {},
                        ArrayRef<int64_t> tileSizes = {},
                        bool useWorkgroupMem = false) {
    this->workgroupSize = workgroupSize;
    this->tileSizes = tileSizes;
    this->useWorkgroupMemory = useWorkgroupMem;
  }
  LinalgTileAndFusePass(const LinalgTileAndFusePass &pass) {}

  void runOnFunction() override;

 private:
  Option<bool> useWorkgroupMemory{
      *this, "use-workgroup-memory",
      llvm::cl::desc("Promote subviews to use workgroup memory"),
      llvm::cl::init(false)};

  ListOption<int64_t> workgroupSize{
      *this, "workgroup-size",
      llvm::cl::desc("Override the default workgroup size"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};

  ListOption<int64_t> tileSizes{
      *this, "tile-sizes", llvm::cl::desc("Set tile sizes to use"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

/// Pattern for tiling operations. Updates the workgroup size in the surrounding
/// function operation if tiling succeeds.
struct TileMatmulPattern
    : public linalg::LinalgTilingPattern<linalg::MatmulOp> {
  using Base = linalg::LinalgTilingPattern<linalg::MatmulOp>;
  TileMatmulPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                    ArrayRef<int64_t> workgroupSize, PatternBenefit benefit = 1)
      : Base(context, options,
             linalg::LinalgMarker(
                 ArrayRef<Identifier>(),
                 Identifier::get(getWorkgroupNumItemsGENumItersMarker(),
                                 context)),
             benefit),
        workgroupSize(workgroupSize.begin(), workgroupSize.end()) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    // Find the parent FuncOp before tiling. If tiling succeeds, the op will be
    // erased.
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(updateWorkGroupSize(funcOp, workgroupSize)))
      return failure();
    funcOp.setAttr(getWorkgroupCountAttrName(),
                   rewriter.getI32IntegerAttr(static_cast<int32_t>(
                       WorkgroupCountMethodology::ResultShape)));
    return success();
  }

  SmallVector<int64_t, 3> workgroupSize;
};

struct TileBatchMatmulPattern
    : public linalg::LinalgTilingPattern<linalg::BatchMatmulOp> {
  using Base = linalg::LinalgTilingPattern<linalg::BatchMatmulOp>;
  TileBatchMatmulPattern(MLIRContext *context,
                         linalg::LinalgTilingOptions options,
                         ArrayRef<int64_t> workgroupSize,
                         PatternBenefit benefit = 1)
      : Base(context, options,
             linalg::LinalgMarker(
                 ArrayRef<Identifier>(),
                 Identifier::get(getWorkgroupMarker(), context)),
             benefit),
        workgroupSize(workgroupSize.begin(), workgroupSize.end()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(updateWorkGroupSize(funcOp, this->workgroupSize))) {
      return failure();
    }
    funcOp.setAttr(getWorkgroupCountAttrName(),
                   rewriter.getI32IntegerAttr(static_cast<int32_t>(
                       WorkgroupCountMethodology::ResultShape)));
    return success();
  }

  SmallVector<int64_t, 3> workgroupSize;
};

/// Pattern for tiling convolution and pooling operations. Currently is just a
/// way to not tile when the operation has padding.
template <typename OpTy>
struct TileConvPoolPattern : public linalg::LinalgTilingPattern<OpTy> {
  using Base = linalg::LinalgTilingPattern<OpTy>;
  TileConvPoolPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                      ArrayRef<int64_t> workgroupSize,
                      PatternBenefit benefit = 1)
      : Base(context, options,
             linalg::LinalgMarker(
                 ArrayRef<Identifier>(),
                 Identifier::get(getWorkgroupMarker(), context)),
             benefit),
        workgroupSize(workgroupSize.begin(), workgroupSize.end()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (hasPadding(cast<OpTy>(op))) return failure();
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp || failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(updateWorkGroupSize(funcOp, this->workgroupSize)))
      return failure();
    funcOp.setAttr(getWorkgroupCountAttrName(),
                   rewriter.getI32IntegerAttr(static_cast<int32_t>(
                       WorkgroupCountMethodology::Default)));
    return success();
  }

  SmallVector<int64_t, 3> workgroupSize;
};

//===----------------------------------------------------------------------===//
// Patterns to promote subviews to workgroup memory
//===----------------------------------------------------------------------===//

/// Function used as callback for copyin/copyout in promotion pattern used to
/// promote subviews to workgroup memory when the number of threads is known to
/// be greater than equal to the number of iteration of loops the copy is
/// lowered to.
static LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

/// Pattern to promote matmul operands to workgroup memory.
struct PromoteMatmulSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteMatmulSubviewsPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      linalg::LinalgMarker marker = linalg::LinalgMarker(),
      PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1}).setUseFullTileBuffers(
                {false, false}),
            linalg::LinalgMarker(
                Identifier::get(getWorkgroupNumItemsGENumItersMarker(),
                                context),
                Identifier::get(getWorkgroupMemoryNumItemsGENumItersMarker(),
                                context)),
            benefit) {}
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
  PromoteConvolutionSubviewsPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      linalg::LinalgMarker marker = linalg::LinalgMarker(),
      PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::ConvOp>(
            context,
            options.setOperandsToPromote({1}).setUseFullTileBuffers(
                {false, false}),
            linalg::LinalgMarker(
                Identifier::get(getWorkgroupMarker(), context),
                Identifier::get(getWorkgroupMemoryMarker(), context)),
            benefit) {}
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

  TileSizeCalculator tileSizeCalculator(funcOp);
  if (tileSizes.empty()) {
    // Get the tile sizes to use for the lowering.
    SmallVector<int64_t, 3> tileSizes;
    SmallVector<linalg::LinalgOp, 1> opsVec(linalgOps.begin(), linalgOps.end());
    if (failed(tileSizeCalculator.inferTileAndWorkgroupSize(opsVec)))
      return signalPassFailure();
  } else {
    tileSizeCalculator.setTileSizes(tileSizes);
    if (!workgroupSize.empty())
      tileSizeCalculator.setWorkgroupSize(workgroupSize);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";
    llvm::dbgs() << "# workgroup sizes: [";
    interleaveComma(tileSizeCalculator.getWorkgroupSize(), llvm::dbgs());
    llvm::dbgs() << "]\ntile sizes: [";
    interleaveComma(tileSizeCalculator.getTileSizes(), llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  OwningRewritePatternList tilingPatterns;
  tilingPatterns
      .insert<TileConvPoolPattern<linalg::ConvOp>, TileMatmulPattern,
              TileBatchMatmulPattern, TileConvPoolPattern<linalg::PoolingMaxOp>,
              TileConvPoolPattern<linalg::PoolingMinOp>,
              TileConvPoolPattern<linalg::PoolingSumOp>>(
          context,
          linalg::LinalgTilingOptions()
              .setTileSizes(tileSizeCalculator.getTileSizes())
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
          tileSizeCalculator.getWorkgroupSize());
  applyPatternsAndFoldGreedily(getOperation(), tilingPatterns);

  if (useWorkgroupMemory) {
    // The promotion patterns are put separate from the tiling patterns to make
    // sure that the allocated scratchspace memory is constant sizes which
    // requires some folding to trigger.
    OwningRewritePatternList promotionPatterns;
    promotionPatterns.insert<PromoteMatmulSubviewsPattern,
                             PromoteConvolutionSubviewsPattern>(
        context,
        linalg::LinalgPromotionOptions()
            .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                          deallocateWorkgroupMemory)
            .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory));
    applyPatternsAndFoldGreedily(getOperation(), promotionPatterns);
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createLinalgTileAndFusePass(
    ArrayRef<int64_t> workgroupSize, ArrayRef<int64_t> tileSizes,
    bool useWorkgroupMemory) {
  return std::make_unique<LinalgTileAndFusePass>(workgroupSize, tileSizes,
                                                 useWorkgroupMemory);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
