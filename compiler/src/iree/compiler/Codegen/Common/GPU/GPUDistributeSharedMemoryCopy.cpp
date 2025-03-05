// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute-shared-memory-copy"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTESHAREDMEMORYCOPYPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
//====---------------------------------------------------------------------===//
// Pass to lower workgroup memory copy to distibuted
// transfer_read/transfer_write ops.
//====---------------------------------------------------------------------===//

/// Prints the given `funcOp` after a leading `step` comment header.
static void debugPrint(Operation *op, const char *step) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << step << " ---//\n";
    op->print(llvm::dbgs(), mlir::OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

// Markers for intermediate transformations.
static const llvm::StringRef kCopyToDistribute = "copy_to_distribute";
static const llvm::StringRef kCopyDistributed = "copy_distributed";

// For optimal performance we always want to copy 128 bits
static constexpr int copyVectorNumBits = 128;

/// Tiles copy to shared memory mapping. Copy to shared memory are not part of
/// the launch config but needs to be distributed on the workgroup picked by the
/// root op.
static LogicalResult tileCopyToWorkgroupMem(mlir::FunctionOpInterface funcOp,
                                            ArrayRef<int64_t> workgroupSize) {
  // Tile and distribute copy to workgroup memory.
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        // We tile to 4 as we want each thread to load 4 element in a cyclic
        // distribution.
        SmallVector<Value> tileSizesVal;
        MemRefType dstMemRefType =
            llvm::cast<MemRefType>(cast<linalg::GenericOp>(operation)
                                       .getDpsInitOperand(0)
                                       ->get()
                                       .getType());

        unsigned rank = dstMemRefType.getRank();
        // Return empty tile size for zero dim tensor.
        if (rank == 0)
          return tileSizesVal;
        int copyTileSize =
            copyVectorNumBits / dstMemRefType.getElementTypeBitWidth();
        for (unsigned i = 0; i < rank - 1; i++) {
          int64_t t = (rank - i) <= kNumGPUDims ? 1 : 0;
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), t));
        }
        tileSizesVal.push_back(builder.create<arith::ConstantIndexOp>(
            operation->getLoc(), copyTileSize));
        return tileSizesVal;
      };
  auto getCopyThreadProcInfoFn =
      [workgroupSize](OpBuilder &builder, Location loc,
                      ArrayRef<Range> parallelLoopRanges) {
        return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                        workgroupSize);
      };
  linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
  copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(wgCopyTileSizeFn)
          .setDistributionOptions(copyInvocationDistributionOptions);

  auto filter = LinalgTransformationFilter(
      {StringAttr::get(funcOp.getContext(), getCopyToWorkgroupMemoryMarker())},
      StringAttr::get(funcOp.getContext(), getVectorizeMarker()));
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

// Returns the vector size to use for the given genericOp considering its
// operand/result element types.
static int getBaseVectorSize(linalg::GenericOp genericOp) {
  assert(genericOp.getNumDpsInits() == 1);
  unsigned resultBW =
      llvm::cast<MemRefType>(genericOp.getDpsInitOperand(0)->get().getType())
          .getElementTypeBitWidth();
  // Check the operand element types. If we have some sub-byte types there, make
  // sure we at least read a full byte for the sub-byte-element operands.
  unsigned operandBW = std::numeric_limits<unsigned>::max();
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    unsigned b =
        IREE::Util::getTypeBitWidth(getElementTypeOrSelf(operand->get()));
    operandBW = std::min(operandBW, b);
  }
  int vectorSize = copyVectorNumBits / resultBW;
  if (operandBW < resultBW && operandBW < 8) {
    // Scale up to make sure we read at least a full byte for the
    // sub-byte-element operand.
    vectorSize *= 8 / operandBW;
  }
  return vectorSize;
}

/// Compute a tile size so that the numer of iteraton is equal to the flat
/// workgroup size.
static std::optional<SmallVector<int64_t>>
getTileToDistributableSize(linalg::GenericOp copyOp,
                           int64_t flatWorkgroupSize) {
  SmallVector<int64_t> shape = copyOp.getStaticLoopRanges();
  int targetVectorSize = getBaseVectorSize(copyOp);
  SmallVector<int64_t> unroll;
  assert(shape.back() % targetVectorSize == 0);
  int64_t threadsAvailable = flatWorkgroupSize;
  for (auto [index, dim] : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = index == 0 ? targetVectorSize : 1;
    int64_t numThreads = dim / numElementPerThread;
    numThreads = std::min(numThreads, threadsAvailable);
    unroll.push_back(numThreads * numElementPerThread);
    assert(threadsAvailable % numThreads == 0);
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1)
      break;
  }
  assert(threadsAvailable == 1);
  unroll.resize(shape.size(), 1);
  std::reverse(unroll.begin(), unroll.end());
  return unroll;
}

/// Tiles copies using serial loops into a shape that can be distributed onto
/// thread.
static LogicalResult tileToUnroll(mlir::FunctionOpInterface funcOp,
                                  int64_t flatWorkgroupSize) {
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [flatWorkgroupSize](OpBuilder &builder, Operation *operation) {
        SmallVector<Value> tileSizesVal;
        auto copyOp = dyn_cast<linalg::GenericOp>(operation);
        if (!copyOp)
          return tileSizesVal;
        std::optional<SmallVector<int64_t>> staticSize =
            getTileToDistributableSize(copyOp, flatWorkgroupSize);
        for (int64_t dim : *staticSize) {
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), dim));
        }
        return tileSizesVal;
      };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(wgCopyTileSizeFn);

  MLIRContext *context = funcOp.getContext();
  auto filter = LinalgTransformationFilter(
      {StringAttr::get(context, getCopyToWorkgroupMemoryMarker())},
      StringAttr::get(context, kCopyToDistribute));
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

/// Break up the flat id onto the static loop ranges.
SmallVector<linalg::ProcInfo> getIds(OpBuilder &b, Location loc,
                                     ArrayRef<Range> parallelLoopRanges,
                                     Value flatThreadId) {
  SmallVector<linalg::ProcInfo> infos;
  SmallVector<int64_t> delinSizes;
  for (Range r : parallelLoopRanges) {
    auto offset = dyn_cast<Attribute>(r.offset);
    auto stride = dyn_cast<Attribute>(r.stride);
    auto size = dyn_cast<Attribute>(r.size);
    assert(offset && stride && size);
    int64_t numThreadsDim = (llvm::cast<IntegerAttr>(size).getInt() -
                             llvm::cast<IntegerAttr>(offset).getInt()) /
                            llvm::cast<IntegerAttr>(stride).getInt();
    delinSizes.push_back(numThreadsDim);
  }
  ValueRange dims =
      b.create<affine::AffineDelinearizeIndexOp>(loc, flatThreadId, delinSizes)
          .getResults();

  for (auto [dimId, numThreadsDim] : llvm::zip_equal(dims, delinSizes)) {
    linalg::ProcInfo info;
    info.procId = dimId;
    info.nprocs = b.create<arith::ConstantIndexOp>(loc, numThreadsDim);
    info.distributionMethod =
        linalg::DistributionMethod::CyclicNumProcsEqNumIters;
    infos.push_back(info);
  }
  return infos;
}

/// Return the shape of copy op that can be vectorized to a
/// transfer_read/transfer_write of size `targetVectorSize`.
SmallVector<int64_t> getNativeDstShape(linalg::GenericOp copyOp) {
  int targetVectorSize = getBaseVectorSize(copyOp);
  SmallVector<int64_t> dstShape;
  for (int64_t dim : copyOp.getStaticLoopRanges()) {
    // Skip tiling of dimension of size 1 to simplify distribution.
    dstShape.push_back(dim == 1 ? 0 : 1);
  }
  dstShape.back() = targetVectorSize;
  return dstShape;
}

/// Distributes linalg copy onto threads based on the flat id.
static LogicalResult tileAndDistribute(mlir::FunctionOpInterface funcOp,
                                       Value flatThreadId) {
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        SmallVector<Value> tileSizesVal;
        auto copyOp = dyn_cast<linalg::GenericOp>(operation);
        if (!copyOp)
          return tileSizesVal;
        SmallVector<int64_t> staticSize = getNativeDstShape(copyOp);
        for (int64_t dim : staticSize) {
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), dim));
        }
        return tileSizesVal;
      };
  auto getCopyThreadProcInfoFn =
      [flatThreadId](OpBuilder &builder, Location loc,
                     ArrayRef<Range> parallelLoopRanges) {
        return getIds(builder, loc, parallelLoopRanges, flatThreadId);
      };
  linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
  copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizeComputationFunction(wgCopyTileSizeFn)
          .setDistributionOptions(copyInvocationDistributionOptions);

  auto filter = LinalgTransformationFilter(
      {StringAttr::get(funcOp.getContext(), kCopyToDistribute)},
      StringAttr::get(funcOp.getContext(), kCopyDistributed));
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

/// Vectorizes generic ops that have CopyToWorkgroupMemoryMarker or
// `kCopyDistributed` marker.
static void
vectorizeCopyToWorkgroupMemoryOps(mlir::FunctionOpInterface funcOp) {
  MLIRContext *context = funcOp.getContext();
  IRRewriter rewriter(context);
  auto filter = LinalgTransformationFilter(
      {StringAttr::get(context, getCopyToWorkgroupMemoryMarker()),
       StringAttr::get(context, kCopyDistributed)},
      std::nullopt);

  funcOp.walk([&](linalg::GenericOp op) {
    if (succeeded(filter.checkAndNotify(rewriter, op))) {
      (void)linalg::vectorize(rewriter, op);
    }
  });
}

/// Return a flattened Id Value by combining the 3D gpu thread IDs.
static Value createFlatId(mlir::FunctionOpInterface funcOp,
                          ArrayRef<int64_t> workgroupSize) {
  OpBuilder b(funcOp.getFunctionBody());
  Type indexType = b.getIndexType();
  Value threadX =
      b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType, gpu::Dimension::x);
  Value threadY =
      b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType, gpu::Dimension::y);
  Value threadZ =
      b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType, gpu::Dimension::z);
  Value flatThreadId = b.create<affine::AffineLinearizeIndexOp>(
      funcOp.getLoc(), ValueRange{threadZ, threadY, threadX},
      ArrayRef<int64_t>{workgroupSize[2], workgroupSize[1], workgroupSize[0]},
      /*disjoint=*/true);
  return flatThreadId;
}

/// Hoist allocations to the top of the loop if they have no dependencies.
template <typename AllocTy>
static void hoistAlloc(mlir::FunctionOpInterface funcOp) {
  SmallVector<AllocTy> allocs;
  funcOp.walk([&](AllocTy alloc) {
    if (alloc.getOperands().empty())
      allocs.push_back(alloc);
  });
  for (AllocTy alloc : allocs) {
    alloc->moveBefore(&(*funcOp.getBlocks().begin()),
                      funcOp.getBlocks().begin()->begin());
  }
}

/// We insert barriers conservatively, remove barriers that are obviously not
/// needed.
static void removeRedundantBarriers(mlir::FunctionOpInterface funcOp) {
  funcOp.walk([](linalg::GenericOp copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
      Operation *prevOp = copyOp->getPrevNode();
      SmallVector<Operation *> redundantBarriers;
      while (prevOp) {
        if (isa<gpu::BarrierOp>(prevOp))
          redundantBarriers.push_back(prevOp);
        else
          break;
        prevOp = prevOp->getPrevNode();
      }
      if (prevOp && hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
        for (Operation *op : redundantBarriers)
          op->erase();
      }
    }
  });
}

/// Return the number of iteration if it is static, otherwise returns 0.
static int64_t numIteration(scf::ForOp forOp) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lbCstOp || !ubCstOp || !stepCstOp || lbCstOp.value() < 0 ||
      ubCstOp.value() < 0 || stepCstOp.value() < 0)
    return 0;
  int64_t tripCount =
      llvm::divideCeil(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
  return tripCount;
}

/// Fully unroll all the static loops unless they are part of the ignore map.
static void
unrollSharedMemoryLoops(mlir::FunctionOpInterface funcOp,
                        const llvm::SmallDenseSet<scf::ForOp> &loopsToIgnore) {
  SmallVector<scf::ForOp> forOpsToUnroll;
  funcOp.walk([&](scf::ForOp forOp) {
    if (!loopsToIgnore.count(forOp))
      forOpsToUnroll.push_back(forOp);
  });
  for (scf::ForOp forOp : llvm::reverse(forOpsToUnroll)) {
    (void)loopUnrollByFactor(forOp, numIteration(forOp));
  }
}
} // namespace

LogicalResult gpuDistributeSharedMemoryCopy(mlir::FunctionOpInterface funcOp) {
  auto maybeWorkgroupSize = getWorkgroupSize(funcOp);
  if (!maybeWorkgroupSize) {
    return funcOp.emitOpError("failed to distribute shared memory copy since "
                              "workgroup size isnt set");
  }
  SmallVector<int64_t> workgroupSize = maybeWorkgroupSize.value();
  workgroupSize.resize(3, 1);
  MLIRContext *context = funcOp.getContext();
  SmallVector<linalg::GenericOp> copiesToWorkgroupMem;
  funcOp.walk([&](linalg::GenericOp copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker()))
      copiesToWorkgroupMem.push_back(copyOp);
  });
  if (copiesToWorkgroupMem.empty())
    return success();

  // Step 0. First clean up the IR.
  hoistAlloc<memref::AllocOp>(funcOp);
  hoistAlloc<memref::AllocaOp>(funcOp);
  removeRedundantBarriers(funcOp);

  int64_t flatWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  bool isAligned = llvm::all_of(
      copiesToWorkgroupMem, [flatWorkgroupSize](linalg::GenericOp copyOp) {
        MemRefType dstMemRefType = llvm::cast<MemRefType>(
            copyOp.getDpsInitOperand(0)->get().getType());
        auto shape = dstMemRefType.getShape();
        int targetVectorSize =
            copyVectorNumBits / dstMemRefType.getElementTypeBitWidth();
        return canPerformVectorAccessUsingAllThreads(shape, flatWorkgroupSize,
                                                     targetVectorSize);
      });
  debugPrint(funcOp, "After initial IR cleanup");

  if (isAligned) {
    // Ignore all the exisiting loop
    llvm::SmallDenseSet<scf::ForOp> loopsToIgnore;
    funcOp.walk([&](scf::ForOp loop) { loopsToIgnore.insert(loop); });

    // Step 1. tile copies to get to a shape that can be distributed to
    // 128bits per lane copies.
    if (failed(tileToUnroll(funcOp, flatWorkgroupSize))) {
      return failure();
    }
    debugPrint(funcOp, "After step 1: tiling");

    // Calculate a flat id that will then be broken down during distribution.
    Value flatId = createFlatId(funcOp, workgroupSize);
    // Step 2. Distribute the linalg op onto threads.
    if (failed(tileAndDistribute(funcOp, flatId))) {
      return failure();
    }
    debugPrint(funcOp, "After step 2: thread distribution");

    // Step 3. Vectorize the distributed copies.
    vectorizeCopyToWorkgroupMemoryOps(funcOp);
    debugPrint(funcOp, "After step 3: vectorization");

    // Step4. Finally unroll all the loop created
    unrollSharedMemoryLoops(funcOp, loopsToIgnore);
    debugPrint(funcOp, "After step 4: unrolling");
  } else {
    // Fall back to basic tiling for cases where workgroup memory size is not
    // well aligned on the number of threads.
    // TODO(thomasraoux): Handle this case with padding instead so that we get
    // good performance for more complex shapes.
    if (failed(tileCopyToWorkgroupMem(funcOp, workgroupSize))) {
      return failure();
    }
    debugPrint(funcOp, "After tiling for unaligned case");

    // Apply canonicalization patterns.
    RewritePatternSet threadTilingCanonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    populateAffineMinSCFCanonicalizationPattern(
        threadTilingCanonicalizationPatterns);
    if (failed(applyPatternsGreedily(
            funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
      return failure();
    }
  }

  return success();
}

namespace {
struct GPUDistributeSharedMemoryCopyPass final
    : impl::GPUDistributeSharedMemoryCopyPassBase<
          GPUDistributeSharedMemoryCopyPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (failed(gpuDistributeSharedMemoryCopy(funcOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
