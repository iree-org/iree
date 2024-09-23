// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLANNOTATEKERNELFORTRANSLATIONPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {
// Extracts the amdgpu chipset version from the chip architecture in the
// executable target attribute.
static FailureOr<amdgpu::Chipset>
getChipsetVersion(IREE::HAL::ExecutableTargetAttr targetAttr) {
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(targetAttr);
  assert(gpuTarget);
  return amdgpu::Chipset::parse(gpuTarget.getArch());
}

// Set attributes on `funcOp` in order to use upstream's translation of
// ROCDL dialect attributes to LLVM. Primarily this is `rocdl.kernel`
// (sets the calling convention and workgroup size uniformity) but this will
// also set both forms of workgroup size metadata from `exportOp` (if it is set)
// and will set the waves_per_eq flag where relevant. Finally, it will mark
// kernel arguments `inreg` to enable argument preloading on supported
// architectures.
static LogicalResult
annotateKernelForTranslation(LLVM::LLVMFuncOp funcOp,
                             IREE::HAL::ExecutableVariantOp variantOp,
                             IREE::HAL::ExecutableExportOp exportOp) {
  OpBuilder builder(funcOp);
  auto *rocdlDialect =
      funcOp.getContext()->getLoadedDialect<ROCDL::ROCDLDialect>();
  assert(rocdlDialect && "ROCDL dialect not loaded");
  UnitAttr unitAttr = builder.getUnitAttr();
  rocdlDialect->getKernelAttrHelper().setAttr(funcOp, unitAttr);
  std::optional<ArrayAttr> workgroupSizeAttr = exportOp.getWorkgroupSize();
  if (workgroupSizeAttr && workgroupSizeAttr->size() <= 3) {
    std::array<int32_t, 3> wgSizes;
    int32_t flatWgSize = 1;
    for (auto [value, attr] : llvm::zip_equal(
             wgSizes, workgroupSizeAttr->getAsRange<IntegerAttr>())) {
      value = attr.getInt();
      flatWgSize *= value;
    }
    rocdlDialect->getReqdWorkGroupSizeAttrHelper().setAttr(
        funcOp, builder.getDenseI32ArrayAttr(wgSizes));
    rocdlDialect->getFlatWorkGroupSizeAttrHelper().setAttr(
        funcOp,
        builder.getStringAttr(Twine(flatWgSize) + "," + Twine(flatWgSize)));
  }

  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
  if (std::optional<IntegerAttr> attr =
          getConfigIntegerAttr(targetAttr, "waves_per_eu")) {
    rocdlDialect->getWavesPerEuAttrHelper().setAttr(funcOp, *attr);
  }

  // Kernel argument preloading is only supported on gfx940 and newer targets
  // from the CDNA family. This is enabled using the `inreg` function argument
  // attribute.
  FailureOr<amdgpu::Chipset> chipset = getChipsetVersion(targetAttr);
  if (failed(chipset))
    return variantOp.emitError() << "failed to parse amdgpu chipset";

  if (chipset->majorVersion != 9 || *chipset < amdgpu::Chipset(9, 4, 0))
    return success();

  auto inRegAttrName =
      builder.getStringAttr(LLVM::LLVMDialect::getInRegAttrName());
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
    funcOp.setArgAttr(i, inRegAttrName, unitAttr);

  return success();
}

/// Lowers an IREE hal.executable.variant operation using a suitable pass
/// pipeline.
struct ROCDLAnnotateKernelForTranslationPass final
    : impl::ROCDLAnnotateKernelForTranslationPassBase<
          ROCDLAnnotateKernelForTranslationPass> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp funcOp = getOperation();
    StringRef funcName = funcOp.getName();

    auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    if (!variantOp) {
      funcOp.emitError() << "cannot find parent hal.executable.variant op";
      return signalPassFailure();
    }

    IREE::HAL::ExecutableExportOp exportOp;
    // Try to find the matching executable export op.
    for (IREE::HAL::ExecutableExportOp candidate : variantOp.getExportOps()) {
      if (candidate.getSymName() == funcName) {
        exportOp = candidate;
        break;
      }
    }

    // Un-exported functions are library functions or otherwise not kernels, so
    // don't need these annotations.
    if (!exportOp)
      return;

    if (failed(annotateKernelForTranslation(funcOp, variantOp, exportOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
