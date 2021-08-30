// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---- SPIRVCopyToWorkgroupMemoryPass.cpp ------------------------------===//
//
// This pass lowers linalg.copy for copying data to the workgroup memory.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <numeric>

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

template <typename GPUIdOp, typename GPUCountOp>
linalg::ProcInfo getLinearizedGPUProcessorIdAndCount(
    Location loc, ConversionPatternRewriter &rewriter) {
  SmallVector<linalg::ProcInfo, 3> procInfo =
      getGPUProcessorIdsAndCounts<GPUIdOp, GPUCountOp>(rewriter, loc,
                                                       kNumGPUDims);
  linalg::ProcInfo linearized;
  linearized.procId = procInfo[0].procId;
  linearized.nprocs = procInfo[0].nprocs;
  for (unsigned i = 0; i < kNumGPUDims - 1; ++i) {
    linearized.procId =
        rewriter.create<MulIOp>(loc, linearized.procId, procInfo[i + 1].nprocs);
    linearized.procId =
        rewriter.create<AddIOp>(loc, linearized.procId, procInfo[i + 1].procId);
    linearized.nprocs =
        rewriter.create<MulIOp>(loc, linearized.nprocs, procInfo[i + 1].nprocs);
  }
  return linearized;
}

/// Distributes scf.parallel to processors with the processors logically
/// arranged with same dimensionality as the number of loops, i.e. a
/// scf.parallel with 2 loops to a 2D grid of processors. `processorIDs` and
/// `numProcessors` must be of same size as the number of loops and are the
/// values to use for process ID and number of processors along each dimension
/// in the distributed code.
/// This method accounts for the case where the number of processors is not
/// enough to execute the entire iteration space with one iteration mapped to
/// each processor. So implements a cyclic distribution of iterations to
/// processors.
LogicalResult distributeCyclicallyToProcessors(
    ConversionPatternRewriter &rewriter, scf::ParallelOp pLoopOp,
    ArrayRef<linalg::ProcInfo> procInfo) {
  unsigned numLoops = pLoopOp.getNumLoops();
  assert(numLoops == procInfo.size() &&
         "expected as many ids as number of loops");
  SmallVector<LoopBounds, 2> forBounds;
  SmallVector<unsigned, 2> permutation;
  forBounds.reserve(numLoops);
  permutation.reserve(numLoops);
  Location loc = pLoopOp.getLoc();
  auto lbs = pLoopOp.lowerBound(), ubs = pLoopOp.upperBound(),
       steps = pLoopOp.step();
  for (unsigned i : llvm::seq<unsigned>(0, procInfo.size())) {
    Value mappedLb = rewriter.create<AddIOp>(
        loc, lbs[i],
        rewriter.create<MulIOp>(loc, steps[i], procInfo[i].procId));
    Value mappedStep =
        rewriter.create<MulIOp>(loc, steps[i], procInfo[i].nprocs);
    forBounds.push_back({mappedLb, ubs[i], mappedStep});
    permutation.push_back(i);
  }
  replacePLoopOp(rewriter, pLoopOp, /*newPLoopBounds=*/{}, forBounds,
                 permutation);
  return success();
}

/// Returns the number of bytes copied when loading to/storing from workgorup
/// memory. It is approximated to be the size of the underlying allocation being
/// copied into/from.
Optional<int64_t> getLinearizedCopySize(linalg::CopyOp copyOp) {
  Value src = copyOp.input();
  Value dst = copyOp.output();
  MemRefType srcType = src.getType().cast<MemRefType>();
  MemRefType dstType = dst.getType().cast<MemRefType>();

  Value workgroupMemoryView;
  MemRefType workgroupMemoryType;
  if (srcType.getMemorySpaceAsInt() == getWorkgroupMemorySpace()) {
    workgroupMemoryView = src;
    workgroupMemoryType = srcType;
  } else if (dstType.getMemorySpaceAsInt() == getWorkgroupMemorySpace()) {
    workgroupMemoryView = dst;
    workgroupMemoryType = dstType;
  } else {
    return {};
  }

  memref::SubViewOp workgroupMemorySubviewOp =
      dyn_cast_or_null<memref::SubViewOp>(workgroupMemoryView.getDefiningOp());
  if (!workgroupMemorySubviewOp) return {};
  memref::AllocOp allocOp = dyn_cast_or_null<memref::AllocOp>(
      workgroupMemorySubviewOp.source().getDefiningOp());
  if (!allocOp) return {};

  MemRefType allocOpType = allocOp.getType();
  if (!allocOpType.hasStaticShape()) return {};
  return allocOpType.getNumElements();
}

LogicalResult distributeCopyOp(linalg::CopyOp copyOp, scf::ParallelOp pLoopOp,
                               ConversionPatternRewriter &rewriter) {
  pLoopOp = collapseParallelLoops(rewriter, pLoopOp);
  if (!pLoopOp) return failure();

  Optional<int64_t> copyLength = getLinearizedCopySize(copyOp);
  linalg::ProcInfo idAndCount =
      getLinearizedGPUProcessorIdAndCount<gpu::ThreadIdOp, gpu::BlockDimOp>(
          copyOp.getLoc(), rewriter);
  auto workgroupSize =
      spirv::lookupLocalWorkGroupSize(copyOp).getValues<APInt>();
  int64_t linearizedWorkgroupSize = std::accumulate(
      workgroupSize.begin(), workgroupSize.end(), 1,
      [](int64_t total, APInt value) { return total * value.getSExtValue(); });

  if (copyLength.hasValue() && !workgroupSize.empty() &&
      copyLength.getValue() <= linearizedWorkgroupSize) {
    return distributeSingleIterationPerProcessor(rewriter, pLoopOp, idAndCount,
                                                 /*generateGuard=*/true);
  }
  return distributeCyclicallyToProcessors(rewriter, pLoopOp, idAndCount);
}

// Applies tiling followed to load/store optimized size then distribute on
// incovations.
LogicalResult tileAndDistributeCopy(linalg::CopyOp copyOp,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  linalg::LinalgTilingOptions options;
  // Tile to memory access of 128bits as those tend to be optimal on most GPUs.
  constexpr unsigned vecLoadBits = 128;
  unsigned elementBits =
      copyOp.getSource().getType().cast<MemRefType>().getElementTypeBitWidth();
  if (elementBits == 0 || vecLoadBits % elementBits != 0) return failure();
  unsigned numElement = vecLoadBits / elementBits;
  options.setTileSizes({1, numElement})
      .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);
  Optional<linalg::TiledLinalgOp> tiledOp =
      linalg::tileLinalgOp(rewriter, copyOp, options);
  if (!tiledOp) return failure();
  if (tiledOp->loops.empty()) return success();
  setMarker(tiledOp->op, getVectorizeMarker());
  auto pLoopOp = cast<scf::ParallelOp>(tiledOp->loops[0]);
  return distributeCopyOp(copyOp, pLoopOp, rewriter);
}

// Pattern to tile and distribute linalg::CopyOp.
struct TileAndDistributeCopyOp : public OpConversionPattern<linalg::CopyOp> {
  using OpConversionPattern<linalg::CopyOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      linalg::CopyOp linalgOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!hasMarker(linalgOp, getCopyToWorkgroupMemoryMarker())) {
      return failure();
    }
    if (failed(tileAndDistributeCopy(linalgOp, operands, rewriter))) {
      return failure();
    }

    // Insert a barrier if read or write shared memory.
    if (llvm::any_of(linalgOp.getOperands(), [](Value output) {
          return output.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                 getWorkgroupMemorySpace();
        })) {
      rewriter.create<spirv::ControlBarrierOp>(
          linalgOp.getLoc(), spirv::Scope::Workgroup, spirv::Scope::Workgroup,
          spirv::MemorySemantics::AcquireRelease);
    }
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

/// CopyOp that are loading to/storing from workgroup memory are special cased
/// to use all workitems to do a copy. This is done by linearizing the copy
/// operation.
// TODO(ravishankarm): This linearization is achieved through collapsing the
// generated parallel loops from a multi-dimensional copy. Such lowering results
// in mods/divs in the collapsed loop body. This can be removed by reshaping the
// copy to be a 1D copy. This seems to be hitting an error in reshape
// canonicalization. Investigate this further.
struct SerializeAndDistributeCopy : public OpConversionPattern<linalg::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::CopyOp copyOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!hasMarker(copyOp, {getCopyToWorkgroupMemoryMarker()}))
      return failure();

    Optional<linalg::LinalgLoops> loops =
        linalg::linalgOpToParallelLoops(rewriter, copyOp);
    if (!loops) return failure();
    if (!loops.getValue().empty()) {
      auto pLoopOp = cast<scf::ParallelOp>(loops.getValue()[0]);
      if (failed(distributeCopyOp(copyOp, pLoopOp, rewriter))) return failure();
    }

    // If the `copyOp` writes to workgroup memory insert barrier after the
    // op.
    if (llvm::any_of(copyOp.getOperands(), [](Value output) {
          MemRefType outputType = output.getType().dyn_cast<MemRefType>();
          return outputType &&
                 outputType.getMemorySpaceAsInt() == getWorkgroupMemorySpace();
        })) {
      rewriter.create<spirv::ControlBarrierOp>(
          copyOp.getLoc(), spirv::Scope::Workgroup, spirv::Scope::Workgroup,
          spirv::MemorySemantics::AcquireRelease);
    }

    rewriter.eraseOp(copyOp);
    return success();
  }
};

struct SPIRVCopyToWorkgroupMemoryPass
    : public SPIRVCopyToWorkgroupMemoryBase<SPIRVCopyToWorkgroupMemoryPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;

 private:
  void tileAndVectorizeLinalgCopy(FuncOp funcOp, MLIRContext *context);
  void lowerVectorOps(FuncOp funcOp, MLIRContext *context);
};

void SPIRVCopyToWorkgroupMemoryPass::tileAndVectorizeLinalgCopy(
    FuncOp funcOp, MLIRContext *context) {
  // 1. Tile linalg and distribute it on invocations.
  std::unique_ptr<ConversionTarget> target =
      std::make_unique<ConversionTarget>(*context);
  target->addDynamicallyLegalOp<linalg::CopyOp>([&](linalg::CopyOp copy) {
    return !(hasMarker(copy, getCopyToWorkgroupMemoryMarker()));
  });
  target->markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  OwningRewritePatternList patterns(&getContext());
  // TODO(antiagainst): Re-enable vectorizing workgroup memory copy once the
  // whole pipeline is in a better state.
  // patterns.add<TileAndDistributeCopyOp>(context);
  patterns.add<SerializeAndDistributeCopy>(context);
  if (failed(applyPartialConversion(funcOp, *target, std::move(patterns)))) {
    return signalPassFailure();
  }

  // 2. Canonicalize the IR generated by tiling.
  RewritePatternSet canonicalizePatterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  populateAffineMinCanonicalizationPattern(canonicalizePatterns);
  scf::populateSCFForLoopCanonicalizationPatterns(canonicalizePatterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(canonicalizePatterns));

  // 3. Vectorize the tiled linalg to be able to map it to load/store vector.
  OwningRewritePatternList vectorizationPatterns(&getContext());
  linalg::insertVectorizationPatterns<linalg::CopyOp>(
      vectorizationPatterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context), {}));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
}

void SPIRVCopyToWorkgroupMemoryPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  tileAndVectorizeLinalgCopy(funcOp, context);
}
}  // namespace

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//
std::unique_ptr<OperationPass<FuncOp>> createSPIRVCopyToWorkgroupMemoryPass() {
  return std::make_unique<SPIRVCopyToWorkgroupMemoryPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
