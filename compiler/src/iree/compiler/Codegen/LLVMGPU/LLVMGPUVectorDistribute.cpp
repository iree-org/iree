// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-distribute"

using LayoutDimension = mlir::iree_compiler::IREE::VectorExt::LayoutDimension;
using LayoutDimensionAttr =
    mlir::iree_compiler::IREE::VectorExt::LayoutDimensionAttr;
using VectorLayoutInterface =
    mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface;
using PerDimLayoutAttr = mlir::iree_compiler::IREE::VectorExt::PerDimLayoutAttr;
using LayoutAttr = mlir::iree_compiler::IREE::VectorExt::LayoutAttr;

namespace mlir::iree_compiler {

namespace {

// Vector layout option setter aimed at contractions. Currently this only sets
// anchors for two types of operations; vector.contract and vector.transfer_read
// from non-shared memory. The assumption in this case is that all IR input to
// this pass has a leaf rooted on a transfer_read or includes a contraction in
// the program slice, meaning all operations should receive layouts. Layout
// setting for other problems like reductions is TODO.
class ContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  ContractionVectorLayoutOptions(Operation *root, ArrayAttr types,
                                 ArrayRef<int64_t> workgroupSize,
                                 IREE::GPU::MMAScheduleAttr schedule,
                                 Value laneId)
      : VectorLayoutOptions(root), mmaTypes(types),
        workgroupSize(workgroupSize), schedule(schedule),
        patterns(root->getContext()) {
    populateGPUDistributionPatterns(patterns);
    populateGPUDistributionLayoutAttrPatterns(laneId, patterns);
    populateGPUDistributeNestedLayoutAttrPatterns(laneId, patterns);
  }

  void setAnchorOps(VectorLayoutAnalysis &analysis) override {
    MLIRContext *context = root->getContext();
    root->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case([&](vector::ContractionOp contract) {
            setContractionAnchor(context, analysis, contract);
          })
          .Case([&](vector::TransferReadOp transfer) {
            setTransferReadAnchor(context, analysis, transfer);
          });
    });
  }

  RewritePatternSet &getPatterns() { return patterns; }

private:
  // Sets an anchoring layout for the given contraction op. Looks for a
  // supported mma type from the cached list of mma types and populates the
  // necessary distribution pattern for those contractions.
  void setContractionAnchor(MLIRContext *context,
                            VectorLayoutAnalysis &analysis,
                            vector::ContractionOp contract) {
    // TODO: Add SIMT fallback.
    assert(schedule && "incompatible contraction op");

    auto layouts = schedule.getContractionLayout(contract);
    assert(layouts && "mma layout type must not be opaque");

    auto [aLayout, bLayout, cLayout] = *layouts;
    analysis.setAnchor(contract.getLhs(), aLayout);
    analysis.setAnchor(contract.getRhs(), bLayout);
    analysis.setAnchor(contract.getAcc(), cLayout);
    analysis.setAnchor(contract.getResult(), cLayout);
    contract->setAttr("iree.amdgpu.mfma", schedule.getIntrinsic());
    LLVM_DEBUG({
      llvm::dbgs() << "chosen a layout: " << aLayout << "\n";
      llvm::dbgs() << "chosen b layout: " << bLayout << "\n";
      llvm::dbgs() << "chosen c layout: " << cLayout << "\n";
      llvm::dbgs() << "anchor set on contract: " << contract << "\n";
    });

    if (isa<IREE::GPU::MFMAAttr>(schedule.getIntrinsic())) {
      if (!populatedMfma) {
        populateGPUDistributeNestedLayoutContractAMDGPUPatterns(patterns);
        populatedMfma = true;
      }
    } else {
      llvm_unreachable("Unsupported mma type");
    }
  }

  // Sets a layout anchor for reads from global memory.
  // The layout this generates is approximately the following:
  //
  // #layout = #iree_vector_ext.nested_layout<
  //    subgroups_per_workgroup = [1, ..., 1]
  //    batches_per_subgroup =    [<remaining undistributed elements>]
  //    outers_per_batch =        [1, ..., 1]
  //    threads_per_outer =       [<greedy from innermost memref dim>]
  //    elements_per_thread =     [1, ..., 128/element_bitwidth, ..., 1]
  //            innermost_memref_dimension ^^^^^^
  //
  // (All orders are the same)
  //    *_order = [<broadcasted_dims>, <transfer_permutation>]>
  //
  // So for the following transfer_read with 64 threads:
  //  vector.transfer_read ... : memref<16x256xf16>, vector<16x32xf16>
  //
  // We use the following layout:
  // #layout = #iree_vector_ext.nested_layout<
  //    subgroups_per_workgroup = [1, 1]
  //    batches_per_subgroup =    [1, 1]
  //    outers_per_batch =        [1, 1]
  //    threads_per_outer =       [16, 4]
  //    elements_per_thread =     [1, 8]
  //
  //    *_order = [0, 1]>
  void setTransferReadAnchor(MLIRContext *context,
                             VectorLayoutAnalysis &analysis,
                             vector::TransferReadOp transfer) {
    // TODO: Support masking.
    if (transfer.getMask()) {
      return;
    }
    // Shared memory loads are expected to take the layout of the contraction.
    auto sourceMemRefType =
        dyn_cast<MemRefType>(transfer.getSource().getType());
    if (!sourceMemRefType || hasSharedMemoryAddressSpace(sourceMemRefType)) {
      return;
    }

    int64_t bitWidth = IREE::Util::getTypeBitWidth(
        getElementTypeOrSelf(transfer.getVectorType()));
    if (!llvm::isPowerOf2_64(bitWidth) || bitWidth > 128) {
      return;
    }
    int64_t numElementsPerThread = 128 / bitWidth;
    int64_t flatNumElements =
        ShapedType::getNumElements(transfer.getVectorType().getShape());
    int64_t flatNumThreads = ShapedType::getNumElements(workgroupSize);
    if (flatNumElements % flatNumThreads != 0) {
      return;
    }
    numElementsPerThread =
        std::min(numElementsPerThread, flatNumElements / flatNumThreads);

    AffineMap transferMap = transfer.getPermutationMap();
    if (transferMap.getNumDims() == 0) {
      return;
    }

    // Select the innermost dim of the memref as the contiguous dim to load
    // from.
    int64_t transferRank = transfer.getVectorType().getRank();
    std::optional<unsigned> maybeDim = transferMap.getResultPosition(
        getAffineDimExpr(transferMap.getNumDims() - 1, context));
    int64_t distXDim = maybeDim ? *maybeDim : transferRank - 1;

    ArrayRef<int64_t> vectorShape = transfer.getVectorType().getShape();

    // Limit the maximum inner vector read width to the innermost contiguous
    // dimension. We could try to be clever and extend this to adjacent
    // dimensions in cases where the innermost read vector dimension is small,
    // but that requires comparing memref strides and is uncommon. For now
    // prioritize warp contiguity over 128-bit read granularity.
    numElementsPerThread =
        std::min(numElementsPerThread, vectorShape[distXDim]);

    llvm::SetVector<unsigned> vectorDimDistributionOrder;
    // Get the order in which to distribute vector dimensions to threads, going
    // from innermost to outermost memref dimension. It's important to note
    // that this heuristic only applies to matrix multiplication cases where
    // we are promoting the operands of a contraction to shared memory and we
    // have no producers fused with the matmul. In general there is no universal
    // way to set an anchoring layout for reads without doing an analysis of how
    // the read values are used.
    for (int i = transferMap.getNumDims() - 1; i >= 0; --i) {
      std::optional<unsigned> maybeDim =
          transferMap.getResultPosition(getAffineDimExpr(i, context));
      if (maybeDim) {
        vectorDimDistributionOrder.insert(*maybeDim);
      }
    }
    // Add all remaining (broadcasted) dimensions
    for (auto dim : llvm::seq(static_cast<int64_t>(0), transferRank)) {
      if (!vectorDimDistributionOrder.contains(dim))
        vectorDimDistributionOrder.insert(dim);
    }

    int64_t residualThreads = flatNumThreads;
    int64_t residualElements = numElementsPerThread;

    SmallVector<int64_t> order(vectorDimDistributionOrder.rbegin(),
                               vectorDimDistributionOrder.rend());

    // Distribute all threads in the workgroup to the "threads" dimension,
    // meaning subgroup counts is unit here, even though the read is being
    // distributed to multiple subgroups. This is in an attempt to do a
    // workgroup contiguous load.
    SmallVector<int64_t> subgroupCounts(transferRank, 1);
    SmallVector<int64_t> batchSizes(transferRank, 1);
    SmallVector<int64_t> outerSizes(transferRank, 1);
    SmallVector<int64_t> threadCounts(transferRank, 1);
    SmallVector<int64_t> elementSizes(transferRank, 1);

    for (auto dim : llvm::reverse(order)) {
      int64_t vectorSize = vectorShape[dim];
      // Set the element count for the innermost vector dimension.
      if (residualElements != 1) {
        elementSizes[dim] = residualElements;
        vectorSize /= residualElements;
        residualElements = 1;
      }

      assert((residualThreads % vectorSize == 0 ||
              vectorSize % residualThreads == 0) &&
             "dividing threads to incompatible vector");
      if (residualThreads <= vectorSize) {
        vectorSize /= residualThreads;
        threadCounts[dim] = residualThreads;
        residualThreads = 1;
      } else {
        residualThreads /= vectorSize;
        threadCounts[dim] = vectorSize;
        vectorSize = 1;
      }

      batchSizes[dim] = vectorSize;
    }

    // Note that the layout setting logic here necessarily uses all threads in
    // the workgroup to perform the read. As a result we can always directly
    // use the counts as the basis for computing the subgroup/thread indices.
    SmallVector<int64_t> subgroupBasis = subgroupCounts;
    SmallVector<int64_t> threadBasis = threadCounts;

    auto layout = IREE::VectorExt::NestedLayoutAttr::get(
        context, subgroupCounts, order, batchSizes, order, outerSizes, order,
        threadCounts, order, elementSizes, order, subgroupBasis, threadBasis);
    analysis.setAnchor(transfer.getResult(), layout);
  }

  ArrayAttr mmaTypes;
  SmallVector<int64_t, 3> workgroupSize;
  IREE::GPU::MMAScheduleAttr schedule;

  bool populatedMfma = false;
  RewritePatternSet patterns;
};

struct LLVMGPUVectorDistributePass
    : public LLVMGPUVectorDistributeBase<LLVMGPUVectorDistributePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    FailureOr<ArrayAttr> maybeSupportedTypes =
        getSupportedMmaTypes(llvm::cast<func::FuncOp>(func));
    // TODO: Support FMA fallback. Contractions always benefit from an anchoring
    // layout because they do implicit shuffles, or broadcast when loading data.
    if (failed(maybeSupportedTypes)) {
      func->emitError() << "Failed to collect the set of supported mma types "
                           "for vector distribution";
      return signalPassFailure();
    }

    std::optional<int64_t> maybeSubgroupSize = std::nullopt;
    if (func->hasAttr("subgroup_size")) {
      maybeSubgroupSize =
          llvm::cast<IntegerAttr>(func->getAttr("subgroup_size")).getInt();
    } else {
      maybeSubgroupSize = getSubgroupSize(func);
    }
    if (!maybeSubgroupSize) {
      func.emitError() << "subgroup size required for vector distribution";
      return signalPassFailure();
    }

    OpBuilder builder(func);
    builder.setInsertionPointToStart(&func.getFunctionBody().front());
    SmallVector<OpFoldResult> threadGrid = {
        builder.createOrFold<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::x),
        builder.createOrFold<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::y),
        builder.createOrFold<gpu::ThreadIdOp>(func.getLoc(),
                                              gpu::Dimension::z)};

    std::array<int64_t, 3> workgroupSize;
    if (func->hasAttr("subgroup_size")) {
      auto tmpSizes =
          llvm::cast<ArrayAttr>(func->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(tmpSizes)) {
        workgroupSize[i] = llvm::cast<IntegerAttr>(size).getInt();
      }
    } else {
      workgroupSize = getWorkgroupSize(func);
    }

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    auto scheduleAttr =
        func->getAttrOfType<IREE::GPU::MMAScheduleAttr>(scheduleAttrName);
    if (!scheduleAttr) {
      IREE::HAL::ExecutableExportOp entryPoint = *getEntryPoint(func);
      scheduleAttr = entryPoint->getAttrOfType<IREE::GPU::MMAScheduleAttr>(
          scheduleAttrName);
    }
    LLVM_DEBUG({
      if (scheduleAttr)
        llvm::dbgs() << "schedule: " << scheduleAttr << "\n";
    });

    AffineExpr x, y, z;
    bindSymbols(func.getContext(), x, y, z);
    // Construct the expression for linearizing the thread indices.
    AffineExpr linearId =
        x + workgroupSize[0] * y + workgroupSize[1] * workgroupSize[0] * z;
    AffineExpr laneId = linearId % *maybeSubgroupSize;

    // This all needs some kind of simplification; the arithmetic it produces
    // doest not get folded away as nicely as it could.
    AffineMap idMap = AffineMap::getMultiDimIdentityMap(2, func.getContext());

    // Clamp the thread indices to the workgroup sizes.
    OpFoldResult c0 =
        builder.createOrFold<arith::ConstantIndexOp>(func.getLoc(), 0);
    threadGrid[0] = affine::makeComposedFoldedAffineMax(
        builder, func.getLoc(), idMap, {threadGrid[0], c0});
    threadGrid[1] = affine::makeComposedFoldedAffineMax(
        builder, func.getLoc(), idMap, {threadGrid[1], c0});
    threadGrid[2] = affine::makeComposedFoldedAffineMax(
        builder, func.getLoc(), idMap, {threadGrid[2], c0});

    OpFoldResult dimX = builder.getIndexAttr(workgroupSize[0] - 1);
    OpFoldResult dimY = builder.getIndexAttr(workgroupSize[1] - 1);
    OpFoldResult dimZ = builder.getIndexAttr(workgroupSize[2] - 1);
    threadGrid[0] = affine::makeComposedFoldedAffineMin(
        builder, func.getLoc(), idMap, {threadGrid[0], dimX});
    threadGrid[1] = affine::makeComposedFoldedAffineMin(
        builder, func.getLoc(), idMap, {threadGrid[1], dimY});
    threadGrid[2] = affine::makeComposedFoldedAffineMin(
        builder, func.getLoc(), idMap, {threadGrid[2], dimZ});
    Value laneVal = affine::makeComposedAffineApply(builder, func.getLoc(),
                                                    laneId, threadGrid);

    ContractionVectorLayoutOptions options(
        func, *maybeSupportedTypes, workgroupSize, scheduleAttr, laneVal);
    // TODO: This should return failure when distribution fails for any op.
    distributeVectorOps(func, options.getPatterns(), options);
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUVectorDistribute() {
  return std::make_unique<LLVMGPUVectorDistributePass>();
}

} // namespace mlir::iree_compiler
