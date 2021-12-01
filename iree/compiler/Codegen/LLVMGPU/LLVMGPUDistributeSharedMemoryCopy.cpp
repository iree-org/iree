// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

//====---------------------------------------------------------------------===//
// Pass to lower workgroup memory copy to distibuted
// transfer_read/transfer_write ops.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

/// Patterns for copy to shared memory mapping. Copy to shared memory are not
/// part of the launch config but needs to be distributed on the workgroup
/// picked by the root op.
static void populateTilingCopyToWorkgroupMemPatterns(
    OwningRewritePatternList &patterns, ArrayRef<int64_t> workgroupSize) {
  // Tile and distribute copy to workgroup memory.
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        const int64_t copyTileSize = 4;
        // We tile to 4 as we want each thread to load 4 element in a cyclic
        // distribution.
        SmallVector<Value, 4> tileSizesVal;
        unsigned rank =
            cast<linalg::CopyOp>(operation).getOutputBufferTypes()[0].getRank();
        for (unsigned i = 0; i < rank - 1; i++) {
          int64_t t = (rank - i) <= kNumGPUDims ? 1 : 0;
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), t));
        }
        tileSizesVal.push_back(builder.create<arith::ConstantIndexOp>(
            operation->getLoc(), copyTileSize));
        return tileSizesVal;
      };
  auto getCopyThreadProcInfoFn = [workgroupSize](
                                     OpBuilder &builder, Location loc,
                                     ArrayRef<Range> parallelLoopRanges) {
    return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                    workgroupSize);
  };
  linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
  copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;
  copyInvocationDistributionOptions.distributionMethod = {
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(wgCopyTileSizeFn)
          .setDistributionOptions(copyInvocationDistributionOptions);
  patterns.insert<linalg::LinalgTilingPattern<linalg::CopyOp>>(
      patterns.getContext(), tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getCopyToWorkgroupMemoryMarker(),
                           patterns.getContext())},
          Identifier::get(getVectorizeMarker(), patterns.getContext())));
}

static void populateVectorizationPatterns(RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::CopyOp>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(Identifier::get(
          getCopyToWorkgroupMemoryMarker(), patterns.getContext())));
}

// TODO(thomasraoux): Extend this to support smaller vector size as well.
static constexpr int targetVectorSize = 4;

/// Compute a vector size so that the numer of elements is equal to the flat
/// workgroup size.
static Optional<SmallVector<int64_t, 4>> getGPUNativeVectorSize(
    Operation *op, int64_t flatWorkgroupSize) {
  auto vt = dyn_cast<VectorTransferOpInterface>(op);
  if (!vt) return llvm::None;
  if (!vt.permutation_map().isMinorIdentity()) return llvm::None;
  ArrayRef<int64_t> shape = vt.getVectorType().getShape();
  SmallVector<int64_t, 4> unroll;
  assert(shape.back() % targetVectorSize == 0);
  int64_t threadsAvailable = flatWorkgroupSize;
  for (auto &dim : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = dim.index() == 0 ? targetVectorSize : 1;
    int64_t numThreads = dim.value() / numElementPerThread;
    numThreads = std::min(numThreads, threadsAvailable);
    unroll.push_back(numThreads * numElementPerThread);
    assert(threadsAvailable % numThreads == 0);
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1) break;
  }
  assert(threadsAvailable == 1);
  unroll.resize(shape.size(), 1);
  std::reverse(unroll.begin(), unroll.end());
  if (unroll == shape) return llvm::None;
  return unroll;
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                         int64_t flatWorkgroupSize) {
  auto getShape = [flatWorkgroupSize](Operation *op) {
    return getGPUNativeVectorSize(op, flatWorkgroupSize);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions().setNativeShapeFn(getShape));
}

/// Return a flattened Id Value by combining the 3D gpu thread IDs.
static Value createFlatId(FuncOp funcOp, ArrayRef<int64_t> workgroupSize) {
  OpBuilder b(funcOp.getBody());
  Type indexType = b.getIndexType();
  AffineExpr d0 = getAffineDimExpr(0, b.getContext());
  AffineExpr d1 = getAffineDimExpr(1, b.getContext());
  AffineExpr d2 = getAffineDimExpr(2, b.getContext());
  Value threadX = b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType,
                                            b.getStringAttr("x"));
  Value threadY = b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType,
                                            b.getStringAttr("y"));
  Value threadZ = b.create<gpu::ThreadIdOp>(funcOp.getLoc(), indexType,
                                            b.getStringAttr("z"));
  Value flatThreadId = makeComposedAffineApply(
      b, funcOp.getLoc(),
      d0 + workgroupSize[0] * d1 + (workgroupSize[0] * workgroupSize[1]) * d2,
      {threadX, threadY, threadZ});
  return flatThreadId;
}

/// Distribute a transfer read operations on the given thread ids.
static void distributeTransferRead(FuncOp funcOp, Value flatThreadId,
                                   int64_t flatWorkgroupSize) {
  funcOp.walk([&](vector::TransferReadOp readOp) {
    OpBuilder b(readOp);
    Value id = flatThreadId;
    SmallVector<int64_t, 2> multiplier;
    auto shape = readOp.getVectorType().getShape();
    SmallVector<Value> ids;
    SmallVector<AffineExpr> exprs;
    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    int64_t numThreads = flatWorkgroupSize;
    for (auto &dim : llvm::enumerate(llvm::reverse(shape))) {
      int64_t threads =
          dim.index() == 0 ? (dim.value() / targetVectorSize) : dim.value();
      // If we don't need to distribute the dimension, skip it.
      if (threads == 1) continue;
      exprs.push_back(getAffineDimExpr(shape.size() - dim.index() - 1,
                                       funcOp->getContext()));
      multiplier.push_back(threads);
      Value dimId = id;
      assert(numThreads % threads == 0);
      if (numThreads / threads > 1)
        dimId =
            makeComposedAffineApply(b, funcOp.getLoc(), d0 % threads, {dimId});
      ids.push_back(dimId);
      numThreads = numThreads / threads;
      id = makeComposedAffineApply(b, funcOp.getLoc(), d0.floorDiv(threads),
                                   {id});
      if (numThreads <= 1) break;
    }
    std::reverse(ids.begin(), ids.end());
    Optional<mlir::vector::DistributeOps> ops =
        vector::distributPointwiseVectorOp(
            b, readOp, ids, multiplier,
            AffineMap::get(shape.size(), 0, exprs, funcOp.getContext()));
    if (ops.hasValue()) {
      SmallPtrSet<Operation *, 1> extractOp({ops->extract, ops->insert});
      readOp.getResult().replaceAllUsesExcept(ops->insert.getResult(),
                                              extractOp);
    }
  });
}

namespace {

class LLVMGPUDistributeSharedMemoryCopyPass
    : public LLVMGPUDistributeSharedMemoryCopyBase<
          LLVMGPUDistributeSharedMemoryCopyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    auto entryPointOp = getEntryPoint(funcOp);
    if (!entryPointOp) return;
    auto workgroupSize = getWorkgroupSize(entryPointOp);
    workgroupSize.resize(3, 1);
    MLIRContext *context = &getContext();
    SmallVector<linalg::CopyOp> copiesToWorkgroupMem;
    funcOp.walk([&](linalg::CopyOp copyOp) {
      if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker()))
        copiesToWorkgroupMem.push_back(copyOp);
    });
    if (copiesToWorkgroupMem.empty()) return;
    int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    bool isAligned = llvm::all_of(
        copiesToWorkgroupMem, [flatWorkgroupSize](linalg::CopyOp copyOp) {
          auto shape = copyOp.output().getType().cast<MemRefType>().getShape();
          // Verify that each dimension of the shape can be distributed on the
          // threads
          int64_t threadsAvailable = flatWorkgroupSize;
          for (auto &dim : llvm::enumerate(llvm::reverse(shape))) {
            int64_t numElementPerThread =
                dim.index() == 0 ? targetVectorSize : 1;
            int64_t numThreads = dim.value() / numElementPerThread;
            if (numThreads == 0) return false;
            numThreads = std::min(numThreads, threadsAvailable);
            if (threadsAvailable % numThreads != 0) return false;
            threadsAvailable = threadsAvailable / numThreads;
            if (threadsAvailable == 1) break;
          }
          return threadsAvailable == 1;
        });
    if (isAligned) {
      // Step 1. Vectorize the shared memory copy.
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorizationPatterns)))) {
        return signalPassFailure();
      }

      // Step 2. Unroll transfer_read/transfer_write to a vector with the number
      // of element equal to `targetVectorSize * targetVectorSize`. The.
      // transfer op generated can. then be distributed to a single op of target
      // size.
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns, flatWorkgroupSize);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }
      // Step 3. Distribute the transfer ops onto the flat ids.
      Value flatId = createFlatId(funcOp, workgroupSize);
      distributeTransferRead(funcOp, flatId, flatWorkgroupSize);
      // Propagate vector distribution to the chain of ops.
      RewritePatternSet distributePatterns(context);
      vector::populatePropagateVectorDistributionPatterns(distributePatterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(distributePatterns)))) {
        return signalPassFailure();
      }
    } else {
      // Fall back to basic tiling for cases where workgroup memory size is not
      // well aligned on the number of threads.
      // TODO(thomasraoux): Handle this case with padding instead so that we get
      // good performance for more complex shapes.
      OwningRewritePatternList threadLevelTilingPatterns(context);
      populateTilingCopyToWorkgroupMemPatterns(threadLevelTilingPatterns,
                                               workgroupSize);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadLevelTilingPatterns)))) {
        return signalPassFailure();
      }
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createLLVMGPUDistributeSharedMemoryCopy() {
  return std::make_unique<LLVMGPUDistributeSharedMemoryCopyPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
