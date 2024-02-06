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
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using LayoutDimension = mlir::iree_compiler::IREE::VectorExt::LayoutDimension;
using LayoutDimensionAttr =
    mlir::iree_compiler::IREE::VectorExt::LayoutDimensionAttr;
using VectorLayoutInterface =
    mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface;
using PerDimLayoutAttr = mlir::iree_compiler::IREE::VectorExt::PerDimLayoutAttr;
using LayoutAttr = mlir::iree_compiler::IREE::VectorExt::LayoutAttr;

namespace mlir::iree_compiler {

namespace {

// Computes the per-dim layout from a list of dimension types and sizes,
// including an outer most inferred dimension. For example,
//
// sizes = [-1, x, y, z]
// problemSize = s
//
// sizes[0] = s / (x * y * z)
//
// Fails if s is not divisible by the list of known sizes.
FailureOr<PerDimLayoutAttr>
getPerDimLayout(MLIRContext *context, SmallVector<LayoutDimensionAttr> dimTypes,
                SmallVector<int64_t> sizes, int64_t problemSize) {
  assert(sizes.size() == dimTypes.size() && "dim types and sizes mismatch");
  if (sizes.front() != -1) {
    return failure();
  }
  int64_t residualElements = problemSize;
  for (int i = sizes.size() - 1, e = 1; i >= e; --i) {
    if (sizes[i] <= 0) {
      return failure();
    }
    if (residualElements % sizes[i] != 0) {
      return failure();
    }
    residualElements /= sizes[i];
  }
  sizes[0] = residualElements;
  return PerDimLayoutAttr::get(context, dimTypes, sizes);
}

// Vector layout option setter aimed at contractions. Currently this only sets
// anchors for two types of operations; vector.contract and vector.transfer_read
// from non-shared memory. The assumption in this case is that all IR input to
// this pass has a leaf rooted on a transfer_read or includes a contraction in
// the program slice, meaning all operations should receive layouts. Layout
// setting for other problems like reductions is TODO.
class ContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  ContractionVectorLayoutOptions(Operation *root, ArrayAttr types,
                                 ArrayRef<int64_t> workgroupSize, Value laneId)
      : VectorLayoutOptions(root), mmaTypes(types),
        workgroupSize(workgroupSize), patterns(root->getContext()) {
    populateGPUDistributionPatterns(patterns);
    populateGPUDistributionLayoutAttrPatterns(laneId, patterns);
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
    std::optional<IREE::GPU::MmaAttr> maybeMmaType =
        getCompatibleMmaAttr(mmaTypes, contract);
    // TODO: Add SIMT fallback.
    assert(maybeMmaType && "incompatible contraction op");

    auto mmaType = *maybeMmaType;
    auto maybeLayouts = mmaType.getContractionLayout(contract);
    assert(maybeMmaType && "mma layout type must not be opaque");

    auto [aLayout, bLayout, cLayout] = *maybeLayouts;
    analysis.setAnchor(contract.getLhs(), aLayout);
    analysis.setAnchor(contract.getRhs(), bLayout);
    analysis.setAnchor(contract.getAcc(), cLayout);
    analysis.setAnchor(contract.getResult(), cLayout);

    if (isa<IREE::GPU::MFMAAttr>(mmaType)) {
      if (!populatedMfma) {
        populateAMDGPUDistributionPatterns(patterns);
        populatedMfma = true;
      }
    } else {
      llvm_unreachable("Unsupported mma type");
    }
  }

  // Sets a layout anchor for reads from global memory.
  // NOTE: This is mostly ad-hoc trying to fit more general distribution of
  // transfers into the layout attributes that currently don't support more
  // than a few dimensions. This pattern needs to be reworked with the layout
  // attributes.
  //
  // The layout this generates is approximately the following:
  //
  // #row_layout = #iree_vector_ext.per_dim_layout<
  //   [BATCHX, LANEX,         VECTORX],
  //   [-1,     subgroup_size, 128 / element_type_bitwidth]>
  // #col_layout = #iree_vector_ext.per_dim_layout<
  //   [VECTORY, LANEY],
  //   [-1,      leftover threads from subgroup_size / LANEX]>
  // #layout = #iree_vector_ext.layout<#row_layout, #col_layout>
  //
  // Where -1 indicates assign all remaining elements along that dimension
  // to that dim type.
  void setTransferReadAnchor(MLIRContext *context,
                             VectorLayoutAnalysis &analysis,
                             vector::TransferReadOp transfer) {
    // TODO: Support masking.
    if (transfer.getMask()) {
      return;
    }
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

    // TODO: Support > 2-d transfers.
    int64_t transferRank = transfer.getVectorType().getRank();
    if (transferRank > 2) {
      return;
    }

    AffineMap transferMap = transfer.getPermutationMap();
    if (transferMap.getNumDims() == 0) {
      return;
    }

    // Select the inner most dim of the memref as the contiguous dim to load
    // from.
    std::optional<unsigned> maybeDim = transferMap.getResultPosition(
        getAffineDimExpr(transferMap.getNumDims() - 1, context));
    int64_t distXDim = maybeDim ? *maybeDim : transferRank - 1;

    ArrayRef<int64_t> vectorShape = transfer.getVectorType().getShape();

    // In most cases, BATCHX is expected to get a size of 1. For cases with
    // large linear loads or small thread counts this will have to distribute
    // residual elements to the batch dimension.
    SmallVector<LayoutDimensionAttr> xDimTypes = {
        LayoutDimensionAttr::get(context, LayoutDimension::BATCHX),
        LayoutDimensionAttr::get(context, LayoutDimension::LANEX),
        LayoutDimensionAttr::get(context, LayoutDimension::VECTORX)};
    // NOTE: this is cheating because subgroup size == workgroup size here.
    int64_t xThreads = vectorShape[distXDim] / numElementsPerThread;
    xThreads = std::min(flatNumThreads, xThreads);
    int64_t residualThreads = flatNumThreads / xThreads;
    SmallVector<int64_t> xDimSizes = {-1, xThreads, numElementsPerThread};

    // Here, we only distribute the second dimension, if present, along the
    // LANEY and VECTORY dimensions.
    SmallVector<SmallVector<LayoutDimensionAttr>> dimTypes = {
        {LayoutDimensionAttr::get(context, LayoutDimension::VECTORY),
         LayoutDimensionAttr::get(context, LayoutDimension::LANEY)}};
    SmallVector<SmallVector<int64_t>> dimSizes = {{-1, residualThreads}};

    SmallVector<PerDimLayoutAttr> perDimAttrs;
    int64_t idx = 0;
    // Walk the transfer op in reverse to match the preferred processing
    // order of the dimensions types.
    for (int i = transferRank - 1, e = 0; i >= e; --i) {
      int64_t problemSize = vectorShape[i];
      if (i == distXDim) {
        auto maybeLayout =
            getPerDimLayout(context, xDimTypes, xDimSizes, problemSize);
        if (failed(maybeLayout)) {
          return;
        }
        perDimAttrs.push_back(*maybeLayout);
        continue;
      }
      auto maybeLayout =
          getPerDimLayout(context, dimTypes[idx], dimSizes[idx], problemSize);
      idx++;
      if (failed(maybeLayout)) {
        return;
      }
      perDimAttrs.push_back(*maybeLayout);
    }
    SmallVector<PerDimLayoutAttr, 4> reversedLayout(llvm::reverse(perDimAttrs));

    auto layout = LayoutAttr::get(context, reversedLayout);
    analysis.setAnchor(transfer.getResult(), layout);
  }

  ArrayAttr mmaTypes;
  SmallVector<int64_t, 3> workgroupSize;

  bool populatedMfma = false;
  RewritePatternSet patterns;
};

struct LLVMGPUVectorDistributePass
    : public LLVMGPUVectorDistributeBase<LLVMGPUVectorDistributePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
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

    auto maybeSubgroupSize = getSubgroupSize(func);
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
    auto workgroupSize = getWorkgroupSize(func);
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

    OpFoldResult dimX = builder.createOrFold<arith::ConstantIndexOp>(
        func.getLoc(), workgroupSize[0] - 1);
    OpFoldResult dimY = builder.createOrFold<arith::ConstantIndexOp>(
        func.getLoc(), workgroupSize[1] - 1);
    OpFoldResult dimZ = builder.createOrFold<arith::ConstantIndexOp>(
        func.getLoc(), workgroupSize[2] - 1);
    threadGrid[0] = affine::makeComposedFoldedAffineMin(
        builder, func.getLoc(), idMap, {threadGrid[0], dimX});
    threadGrid[1] = affine::makeComposedFoldedAffineMin(
        builder, func.getLoc(), idMap, {threadGrid[1], dimY});
    threadGrid[2] = affine::makeComposedFoldedAffineMin(
        builder, func.getLoc(), idMap, {threadGrid[2], dimZ});
    Value laneVal = affine::makeComposedAffineApply(builder, func.getLoc(),
                                                    laneId, threadGrid);

    ContractionVectorLayoutOptions options(func, *maybeSupportedTypes,
                                           getWorkgroupSize(func), laneVal);
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
