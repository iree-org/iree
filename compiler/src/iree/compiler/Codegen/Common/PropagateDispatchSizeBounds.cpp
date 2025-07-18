// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATEDISPATCHSIZEBOUNDSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Fold statically-known block/grid dimensions to constants, since that's
// stronger than just imposing an upper bound on their size.
static void foldConstantBounds(
    FunctionOpInterface funcOp,
    const std::optional<SmallVector<int64_t>> &staticWorkgroupSizes,
    ArrayRef<int64_t> staticWorkgroupCounts) {
  IRRewriter rewriter(funcOp->getContext());
  auto rewriteToConstant = [&](Operation *op, int64_t constant) {
    rewriter.setInsertionPoint(op);
    Type constType = op->getResult(0).getType();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, constType, IntegerAttr::get(constType, constant));
  };
  funcOp->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](gpu::BlockDimOp blockDimOp) {
          int32_t dim = static_cast<int32_t>(blockDimOp.getDimension());
          if (staticWorkgroupSizes.has_value() &&
              staticWorkgroupSizes->size() > dim) {
            rewriteToConstant(blockDimOp, (*staticWorkgroupSizes)[dim]);
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupSizeOp wgSizeOp) {
          size_t dim = wgSizeOp.getDimension().getZExtValue();
          if (staticWorkgroupSizes.has_value() &&
              staticWorkgroupSizes->size() > dim) {
            rewriteToConstant(wgSizeOp, (*staticWorkgroupSizes)[dim]);
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupCountOp wgCountOp) {
          size_t dim = wgCountOp.getDimension().getZExtValue();
          int64_t size = staticWorkgroupCounts.size() > dim
                             ? staticWorkgroupCounts[dim]
                             : ShapedType::kDynamic;
          if (ShapedType::isStatic(size)) {
            rewriteToConstant(wgCountOp, size);
          }
        })
        .Default([](Operation *) {});
  });
}

static void applyBounds(FunctionOpInterface funcOp,
                        ArrayRef<std::optional<int64_t>> workgroupSizes,
                        ArrayRef<std::optional<int64_t>> workgroupCounts,
                        std::optional<uint64_t> subgroupSize) {
  Builder b(funcOp->getContext());
  funcOp->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](gpu::LaneIdOp laneIdOp) {
          if (subgroupSize) {
            laneIdOp.setUpperBoundAttr(b.getIndexAttr(*subgroupSize));
          }
        })
        .Case([&](gpu::ThreadIdOp tidOp) {
          std::optional<int64_t> bound =
              workgroupSizes[static_cast<uint32_t>(tidOp.getDimension())];
          if (bound) {
            tidOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](gpu::BlockDimOp blockDimOp) {
          std::optional<int64_t> bound =
              workgroupSizes[static_cast<int32_t>(blockDimOp.getDimension())];
          if (bound) {
            blockDimOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupSizeOp wgSizeOp) {
          std::optional<int64_t> bound =
              workgroupSizes[wgSizeOp.getDimension().getZExtValue()];
          if (bound) {
            wgSizeOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupIDOp wgIdOp) {
          std::optional<int64_t> bound =
              workgroupCounts[wgIdOp.getDimension().getZExtValue()];
          if (bound) {
            wgIdOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupCountOp wgCountOp) {
          std::optional<int64_t> bound =
              workgroupCounts[wgCountOp.getDimension().getZExtValue()];
          if (bound) {
            wgCountOp.setUpperBoundAttr(b.getIndexAttr(*bound));
          }
        })
        .Default([](Operation *) {});
  });
}

struct PropagateDispatchSizeBoundsPass final
    : impl::PropagateDispatchSizeBoundsPassBase<
          PropagateDispatchSizeBoundsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<std::optional<int64_t>, 3> workgroupSizes(3, std::nullopt);
    SmallVector<std::optional<int64_t>, 3> workgroupCounts(3, std::nullopt);

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (target) {
      ArrayRef<int32_t> targetWorkgroupSizes =
          target.getWgp().getMaxWorkgroupSizes().asArrayRef();
      ArrayRef<int32_t> targetWorkgroupCounts =
          target.getWgp().getMaxWorkgroupCounts().asArrayRef();
      llvm::transform(targetWorkgroupSizes, workgroupSizes.begin(),
                      [](int32_t x) { return std::optional<int64_t>{x}; });
      llvm::transform(targetWorkgroupCounts, workgroupCounts.begin(),
                      [](int32_t x) { return std::optional<int64_t>{x}; });
    }

    std::optional<SmallVector<int64_t>> staticWorkgroupSize =
        getWorkgroupSize(funcOp);

    std::optional<uint64_t> subgroupSize = getGPUSubgroupSize(funcOp);

    // Late in codegen, we've reconciled the workgroup size onto the export op.
    if (std::optional<IREE::HAL::ExecutableExportOp> exportOp =
            getEntryPoint(funcOp)) {
      if (std::optional<ArrayAttr> exportWorkgroupSize =
              exportOp->getWorkgroupSize()) {
        staticWorkgroupSize =
            llvm::map_to_vector(exportWorkgroupSize->getAsRange<IntegerAttr>(),
                                [](IntegerAttr a) { return a.getInt(); });
      }

      if (std::optional<uint64_t> exportSubgroupSize =
              exportOp->getSubgroupSizeAsUInt()) {
        subgroupSize = exportSubgroupSize;
      }
    }

    if (staticWorkgroupSize) {
      // Target info with no workgroup sizes gives a 0-length array, hence no
      // zip_equal.
      for (auto [size, staticSize] :
           llvm::zip(workgroupSizes, *staticWorkgroupSize)) {
        size = staticSize;
      }
    }
    SmallVector<int64_t> staticWorkgroupCounts = getStaticNumWorkgroups(funcOp);
    assert(staticWorkgroupCounts.size() <= 3 &&
           "workgroup counts are 3D at most");
    for (auto [count, staticCount] :
         llvm::zip(workgroupCounts, staticWorkgroupCounts)) {
      if (staticCount != ShapedType::kDynamic) {
        count = staticCount;
      }
    }

    foldConstantBounds(funcOp, staticWorkgroupSize, staticWorkgroupCounts);
    applyBounds(funcOp, workgroupSizes, workgroupCounts, subgroupSize);
  }
};
} // namespace

} // namespace mlir::iree_compiler
