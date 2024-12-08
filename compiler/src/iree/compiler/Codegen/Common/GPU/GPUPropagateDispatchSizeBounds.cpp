// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROPAGATEDISPATCHSIZEBOUNDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

static void applyBounds(FunctionOpInterface funcOp,
                        ArrayRef<int32_t> workgroupSizes,
                        ArrayRef<int32_t> workgroupCounts) {
  Builder b(funcOp->getContext());
  funcOp->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](gpu::ThreadIdOp tidOp) {
          tidOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupSizes[static_cast<uint32_t>(tidOp.getDimension())]));
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupSizeOp wgSizeOp) {
          wgSizeOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupSizes[wgSizeOp.getDimension().getZExtValue()]));
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupIDOp wgIdOp) {
          wgIdOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupCounts[wgIdOp.getDimension().getZExtValue()]));
        })
        .Case([&](IREE::HAL::InterfaceWorkgroupCountOp wgCountOp) {
          wgCountOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupCounts[wgCountOp.getDimension().getZExtValue()]));
        })
        .Default([](Operation *) {});
  });
}

struct GPUPropagateDispatchSizeBoundsPass final
    : impl::GPUPropagateDispatchSizeBoundsPassBase<
          GPUPropagateDispatchSizeBoundsPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target) {
      funcOp.emitWarning("no known target attribute late in GPU codegen");
      return;
    }
    SmallVector<int32_t, 3> workgroupSizes(
        target.getWgp().getMaxWorkgroupSizes().asArrayRef());
    SmallVector<int32_t, 3> workgroupCounts(
        target.getWgp().getMaxWorkgroupCounts().asArrayRef());

    std::optional<SmallVector<int64_t>> staticWorkgroupSize =
        getWorkgroupSize(funcOp);

    // Late in codegen, we've reconciled the workgroup size onto the export op.
    if (std::optional<IREE::HAL::ExecutableExportOp> exportOp =
            getEntryPoint(funcOp)) {
      if (std::optional<ArrayAttr> exportWorkgroupSize =
              exportOp->getWorkgroupSize()) {
        staticWorkgroupSize =
            llvm::map_to_vector(exportWorkgroupSize->getAsRange<IntegerAttr>(),
                                [](IntegerAttr a) { return a.getInt(); });
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

    applyBounds(funcOp, workgroupSizes, workgroupCounts);
  }
};
} // namespace

} // namespace mlir::iree_compiler
