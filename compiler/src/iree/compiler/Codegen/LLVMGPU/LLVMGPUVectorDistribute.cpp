// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
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

class ContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  ContractionVectorLayoutOptions(Operation *root, ArrayAttr types,
                                 ArrayRef<int64_t> workgroupSize,
                                 ArrayRef<Value> threadIds)
      : VectorLayoutOptions(root), mmaTypes(types),
        workgroupSize(workgroupSize), patterns(root->getContext()) {
    populateGPUDistributionPatterns(patterns);
    populateGPUDistributionLayoutAttrPatterns(threadIds, patterns);
  }

  void setAnchorOps(VectorLayoutAnalysis &analysis) override {
    MLIRContext *context = root->getContext();
    root->walk([&](Operation *op) {
      if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
        FailureOr<IREE::GPU::MmaAttr> maybeMmaType =
            getCompatibleMmaAttr(mmaTypes, contract);
        // TODO: Add SIMT fallback.
        assert(succeeded(maybeMmaType) && "incompatible contraction op");

        auto mmaType = *maybeMmaType;
        auto maybeLayouts = mmaType.getContractionLayout(contract);
        assert(succeeded(maybeMmaType) && "mma layout type must not be opaque");

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
      } else if (auto transfer = dyn_cast<vector::TransferReadOp>(op)) {
        // TODO: Support masking.
        if (transfer.getMask() || true) {
          return;
        }
        auto sourceMemRefType =
            dyn_cast<MemRefType>(transfer.getSource().getType());
        if (!sourceMemRefType ||
            hasSharedMemoryAddressSpace(sourceMemRefType)) {
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

        // TODO: Need delinearization of thread indices to support higher
        // dimensionalities.
        int64_t transferRank = transfer.getVectorType().getRank();
        if (transferRank > 4) {
          return;
        }

        AffineMap transferMap = transfer.getPermutationMap();
        if (transferMap.getNumDims() == 0) {
          return;
        }

        std::optional<unsigned> maybeDim = transferMap.getResultPosition(
            getAffineDimExpr(transferMap.getNumDims() - 1, context));
        int64_t distXDim = maybeDim ? *maybeDim : transferRank - 1;

        SmallVector<SmallVector<LayoutDimensionAttr>> dimTypes = {
            {LayoutDimensionAttr::get(context, LayoutDimension::VECTORY),
             LayoutDimensionAttr::get(context, LayoutDimension::LANEY)},
            {LayoutDimensionAttr::get(context, LayoutDimension::VECTORZ),
             LayoutDimensionAttr::get(context, LayoutDimension::LANEZ)},
            {LayoutDimensionAttr::get(context, LayoutDimension::BATCHY)}};
        SmallVector<SmallVector<int64_t>> dimSizes = {
            {-1, workgroupSize[1]}, {-1, workgroupSize[2]}, {-1}};

        ArrayRef<int64_t> vectorShape = transfer.getVectorType().getShape();

        SmallVector<PerDimLayoutAttr> perDimAttrs;
        int64_t idx = transferRank - 1;
        for (int i = transferRank - 1, e = 0; i >= e; --i) {
          int64_t problemSize = vectorShape[i];
          if (i == distXDim) {
            SmallVector<LayoutDimensionAttr> xDimTypes = {
                LayoutDimensionAttr::get(context, LayoutDimension::BATCHX),
                LayoutDimensionAttr::get(context, LayoutDimension::LANEX),
                LayoutDimensionAttr::get(context, LayoutDimension::VECTORX)};
            SmallVector<int64_t> xDimSizes = {-1, workgroupSize[0],
                                              numElementsPerThread};
            auto maybeLayout =
                getPerDimLayout(context, xDimTypes, xDimSizes, problemSize);
            if (failed(maybeLayout)) {
              return;
            }
            perDimAttrs.push_back(*maybeLayout);
            continue;
          }
          auto maybeLayout = getPerDimLayout(context, dimTypes[idx],
                                             dimSizes[idx], problemSize);
          idx--;
          if (failed(maybeLayout)) {
            return;
          }
          perDimAttrs.push_back(*maybeLayout);
        }

        analysis.setAnchor(transfer.getResult(),
                           LayoutAttr::get(context, perDimAttrs));
      }
    });
  }

  RewritePatternSet &getPatterns() { return patterns; }

private:
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
    // TODO: Support SIMT distribution fallback. Contractions always benefit
    // from an anchoring layout because they do implicit shuffles, or broadcast
    // when loading data.
    if (failed(maybeSupportedTypes)) {
      func->emitError() << "Failed to collect the set of supported mma types "
                           "for vector distribution";
      return signalPassFailure();
    }

    OpBuilder builder(func);
    builder.setInsertionPointToStart(&func.getFunctionBody().front());
    SmallVector<Value> threadGrid = {
        builder.create<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::x),
        builder.create<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::y),
        builder.create<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::z),
    };

    ContractionVectorLayoutOptions options(func, *maybeSupportedTypes,
                                           getWorkgroupSize(func), threadGrid);
    distributeVectorOps(func, options.getPatterns(), options);
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUVectorDistribute() {
  return std::make_unique<LLVMGPUVectorDistributePass>();
}

} // namespace mlir::iree_compiler
