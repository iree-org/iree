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
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse-buffer"

static std::string PromotionMarker = "promotion";

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

  /// Compute the tile sizes based on workgroup size specified.
  LogicalResult setTileSizesBasedOnWorkgroupSize(
      ArrayRef<int64_t> vWorkGroupSize) {
    if (!vWorkGroupSize.empty()) {
      vWorkGroupSize = dropTrailingOnes(vWorkGroupSize);
      workgroupSize.assign(vWorkGroupSize.begin(), vWorkGroupSize.end());
      auto rev = reverse(workgroupSize);
      tileSizes.assign(rev.begin(), rev.end());
    }
    return success();
  }

  /// Compute the tile sizes based on the Linalg Ops within the dispatch region.
  LogicalResult setTileSizesBasedOnOps(ArrayRef<linalg::LinalgOp> linalgOps);

  /// Get the current tile size computed.
  ArrayRef<int64_t> getTileSizes() const { return tileSizes; }

  /// Returns the workgroup size to use based on the tile sizes.
  ArrayRef<int64_t> getWorkGroupSize() const { return workgroupSize; }

 private:
  /// Current tile size configuration.
  SmallVector<int64_t, 4> tileSizes;

  /// Workgroup size to use.
  SmallVector<int64_t, 3> workgroupSize;

  /// Attribute for device constraints.
  spirv::ResourceLimitsAttr resourceLimits;
};
}  // namespace

LogicalResult TileSizeCalculator::setTileSizesBasedOnOps(
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
  };
  uint32_t opInfo = OpInfo::None;
  for (linalg::LinalgOp linalgOp : linalgOps) {
    Operation *op = linalgOp.getOperation();
    if (isa<linalg::ConvOp>(op)) opInfo |= OpInfo::Convolution;
    if (isa<linalg::MatmulOp>(op)) opInfo |= OpInfo::Matmul;
    if (isa<linalg::PoolingMaxOp>(op)) opInfo |= OpInfo::Pooling;
    if (isa<linalg::PoolingMinOp>(op)) opInfo |= OpInfo::Pooling;
    if (isa<linalg::PoolingSumOp>(op)) opInfo |= OpInfo::Pooling;
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
    // TODO(ravishankarm): This tiling is meant to enable promotion to workgroup
    // memory, but doesnt actually get us to a state where we can do this. The
    // promotion is possible only when the subviews created are constant
    // size. For now this doesnt really matter. Revisit this later.
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

/// Allocation callback for allocation workgroup local memory.
static Value allocateWorkgroupMemory(OpBuilder &b, SubViewOp subview,
                                     ArrayRef<Value> boundingSubViewSize,
                                     OperationFolder *folder) {
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape(boundingSubViewSize.size(),
                                ShapedType::kDynamicSize);
  return b.create<AllocOp>(
      subview.getLoc(),
      MemRefType::get(shape, subview.getType().getElementType(), {},
                      getWorkgroupMemorySpace()),
      boundingSubViewSize);
}

/// Deallocation callback for allocation workgroup local memory.
static LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  auto allocOp = buffer.getDefiningOp<AllocOp>();
  b.create<DeallocOp>(allocOp.getLoc(), buffer);
  return success();
}

/// Insert barrier after `op`.
static void insertBarrierAfter(OpBuilder &b, Location loc, Operation *op) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(op);
  b.create<spirv::ControlBarrierOp>(loc, spirv::Scope::Workgroup,
                                    spirv::Scope::Workgroup,
                                    spirv::MemorySemantics::AcquireRelease);
}

/// Function used as callback for copyin/copyout in promotion pattern used to
/// promote subviews to workgroup memory.
static LogicalResult copyToFromWorkgroupMemory(
    OpBuilder &b, Value src, Value dst, StringRef marker = PromotionMarker) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, marker);
  return success();
}

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, FunctionPass> {
  LinalgTileAndFusePass(ArrayRef<int64_t> workGroupSize = {},
                        bool useWorkgroupMem = false)
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {
    this->useWorkgroupMemory = useWorkgroupMem;
  }
  LinalgTileAndFusePass(const LinalgTileAndFusePass &pass) {}

  void runOnFunction() override;

  Option<bool> useWorkgroupMemory{
      *this, "use-workgroup-memory",
      llvm::cl::desc("Promote subviews to use workgroup memory"),
      llvm::cl::init(false)};

 private:
  SmallVector<int64_t, 3> workGroupSize;
};

/// Pattern for tiling operations. Updates the workgroup size in the surrounding
/// function operation if tiling succeeds.
template <typename OpTy>
struct TilingPattern : public linalg::LinalgTilingPattern<OpTy> {
  using Base = linalg::LinalgTilingPattern<OpTy>;
  TilingPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                ArrayRef<int64_t> workgroupSize,
                linalg::LinalgMarker marker = linalg::LinalgMarker(),
                PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit),
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

/// Pattern for tiling convolution and pooling operations. Currently is just a
/// way to not tile when the operation has padding.
template <typename OpTy>
struct TileConvPoolPattern : public TilingPattern<OpTy> {
  using Base = TilingPattern<OpTy>;
  using Base::TilingPattern;

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
};

/// Pattern to promote subviews to memory.
// TODO(ravishankarm): Generalize this for other operations.
struct PromoteSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteSubviewsPattern(MLIRContext *context,
                         linalg::LinalgPromotionOptions options,
                         linalg::LinalgMarker marker = linalg::LinalgMarker(),
                         PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!hasWorkItemMarker(op)) return failure();
    return linalg::LinalgPromotionPattern<linalg::MatmulOp>::matchAndRewrite(
        op, rewriter);
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

  TileSizeCalculator tileSizeCalculator(funcOp);
  if (workGroupSize.empty()) {
    // Get the tile sizes to use for the lowering.
    SmallVector<int64_t, 3> tileSizes;
    SmallVector<linalg::LinalgOp, 1> opsVec(linalgOps.begin(), linalgOps.end());
    if (failed(tileSizeCalculator.setTileSizesBasedOnOps(opsVec)))
      return signalPassFailure();
  } else {
    tileSizeCalculator.setTileSizesBasedOnWorkgroupSize(workGroupSize);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";
    llvm::dbgs() << "# workgroup sizes at start: [";
    interleaveComma(workGroupSize, llvm::dbgs());
    llvm::dbgs() << "]\ntile sizes: [";
    interleaveComma(tileSizeCalculator.getTileSizes(), llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  OwningRewritePatternList tilingPatterns;
  tilingPatterns.insert<TileConvPoolPattern<linalg::ConvOp>,
                        TilingPattern<linalg::MatmulOp>,
                        TileConvPoolPattern<linalg::PoolingMaxOp>,
                        TileConvPoolPattern<linalg::PoolingMinOp>,
                        TileConvPoolPattern<linalg::PoolingSumOp>>(
      context,
      linalg::LinalgTilingOptions()
          .setTileSizes(tileSizeCalculator.getTileSizes())
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
      tileSizeCalculator.getWorkGroupSize(),
      linalg::LinalgMarker(ArrayRef<Identifier>(),
                           Identifier::get(getWorkItemMarker(), context)));
  applyPatternsAndFoldGreedily(getOperation(), tilingPatterns);

  if (useWorkgroupMemory) {
    // The promotion patterns are put separate from the tiling patterns to make
    // sure that the allocated scratchspace memory is constant sizes which
    // requires some folding to trigger.
    OwningRewritePatternList promotionPatterns;
    promotionPatterns.insert<PromoteSubviewsPattern>(
        context,
        linalg::LinalgPromotionOptions()
            .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                          deallocateWorkgroupMemory)
            .setCopyInOutFns(
                [&](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                  return copyToFromWorkgroupMemory(b, src, dst);
                },
                [&](OpBuilder &b, Value src, Value dst) -> LogicalResult {
                  return copyToFromWorkgroupMemory(b, src, dst);
                }),
        linalg::LinalgMarker(Identifier::get(getWorkItemMarker(), context),
                             Identifier::get(PromotionMarker, context)));
    applyPatternsAndFoldGreedily(getOperation(), promotionPatterns);
  }

  // Add barrier after all linalg operations marked with workitem marker.
  OpBuilder builder(context);
  funcOp.walk([&builder](linalg::LinalgOp linalgOp) {
    if (hasMarker(linalgOp, PromotionMarker)) {
      setWorkItemMarker(linalgOp);
      insertBarrierAfter(builder, linalgOp.getLoc(), linalgOp);
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createLinalgTileAndFusePass(
    ArrayRef<int64_t> workGroupSize, bool useWorkgroupMemory) {
  return std::make_unique<LinalgTileAndFusePass>(workGroupSize,
                                                 useWorkgroupMemory);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
