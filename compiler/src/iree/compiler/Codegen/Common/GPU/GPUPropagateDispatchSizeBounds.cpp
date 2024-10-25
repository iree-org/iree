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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-propagate-dispatch-size-bounds"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROPAGATEDISPATCHSIZEBOUNDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

static void applyBounds(FunctionOpInterface funcOp,
                        ArrayRef<int32_t> workgroupSizes,
                        ArrayRef<int32_t> workgroupCounts) {
  Builder b(funcOp->getContext());
  funcOp->walk([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<gpu::ThreadIdOp>([&](auto tidOp) {
          tidOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupSizes[static_cast<uint32_t>(tidOp.getDimension())]));
        })
        .Case<IREE::HAL::InterfaceWorkgroupSizeOp>([&](auto wgSizeOp) {
          wgSizeOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupSizes[wgSizeOp.getDimension().getZExtValue()]));
        })
        .Case<IREE::HAL::InterfaceWorkgroupIDOp>([&](auto wgIdOp) {
          wgIdOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupCounts[wgIdOp.getDimension().getZExtValue()]));
        })
        .Case<IREE::HAL::InterfaceWorkgroupCountOp>([&](auto wgCountOp) {
          wgCountOp.setUpperBoundAttr(b.getIndexAttr(
              workgroupCounts[wgCountOp.getDimension().getZExtValue()]));
        })
        .Default([](Operation *op) { std::ignore = op; });
  });
}

struct GPUPropagateDispatchSizeBoundsPass final
    : impl::GPUPropagateDispatchSizeBoundsPassBase<
          GPUPropagateDispatchSizeBoundsPass> {
  using GPUPropagateDispatchSizeBoundsPassBase::
      GPUPropagateDispatchSizeBoundsPassBase;
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
    if (staticWorkgroupSize) {
      // Target info with no workgroup sizes gives a 0-length array.
      for (auto [size, staticSize] :
           llvm::zip(workgroupSizes, *staticWorkgroupSize)) {
        size = staticSize;
      }
    }
    SmallVector<int64_t> staticWorkgroupCounts = getStaticNumWorkgroups(funcOp);
    if (staticWorkgroupCounts.size() > 3) {
      funcOp.emitWarning("more than 3 workgroup count dimensions");
      return;
    }
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
