// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
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
// Maps IREE denormal fp math mode to LLVM denormal mode kind for the
// denormal_fpenv attribute. Only PreserveSign and PositiveZero are valid;
// None should not be passed.
static LLVM::DenormalModeKind
toLLVMDenormalModeKind(IREE::Codegen::DenormalFpMath mode) {
  switch (mode) {
  case IREE::Codegen::DenormalFpMath::PreserveSign:
    return LLVM::DenormalModeKind::PreserveSign;
  case IREE::Codegen::DenormalFpMath::PositiveZero:
    return LLVM::DenormalModeKind::PositiveZero;
  default:
    return LLVM::DenormalModeKind::IEEE;
  }
}

// Sets denormal_fpenv on the function for float (f32) only: default mode
// remains IEEE, float mode is set to the given kind.
static void setDenormalFpenvForF32(LLVM::LLVMFuncOp funcOp,
                                   LLVM::DenormalModeKind floatMode) {
  MLIRContext *ctx = funcOp.getContext();
  auto attr = LLVM::DenormalFPEnvAttr::get(
      ctx, LLVM::DenormalModeKind::IEEE, LLVM::DenormalModeKind::IEEE,
      floatMode, floatMode);
  funcOp.setDenormalFpenvAttr(attr);
}

// Extracts the amdgpu chipset version from the chip architecture in the
// executable target attribute.
static FailureOr<amdgpu::Chipset>
getChipsetVersion(MLIRContext *context,
                  IREE::HAL::ExecutableTargetAttr targetAttr) {
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(context, targetAttr);
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
  if (IntegerAttr attr =
          getConfigWavesPerEuAttr(targetAttr.getConfiguration())) {
    rocdlDialect->getWavesPerEuAttrHelper().setAttr(funcOp, attr);
  }
  if (IREE::Codegen::DenormalFpMathAttr attr =
          getConfigDenormalFpMathF32Attr(targetAttr.getConfiguration());
      attr && attr.getValue() != IREE::Codegen::DenormalFpMath::None) {
    setDenormalFpenvForF32(funcOp, toLLVMDenormalModeKind(attr.getValue()));
  }

  // Check if the `denormal_fp_math_f32` dictionary is set and process it.
  auto denormalFp32 = cast_or_null<IREE::Codegen::DenormalFpMathAttr>(
      funcOp->getDiscardableAttr(
          IREE::Codegen::DenormalFpMathAttr::getFP32DictKeyName()));
  if (denormalFp32) {
    if (denormalFp32.getValue() != IREE::Codegen::DenormalFpMath::None) {
      setDenormalFpenvForF32(funcOp,
                             toLLVMDenormalModeKind(denormalFp32.getValue()));
    }

    // Discard the attribute.
    funcOp->removeDiscardableAttr(
        IREE::Codegen::DenormalFpMathAttr::getFP32DictKeyName());
  }

  // Kernel argument preloading is only supported on gfx942 and newer targets
  // from the CDNA family. This is enabled using the `inreg` function argument
  // attribute.
  FailureOr<amdgpu::Chipset> chipset =
      getChipsetVersion(builder.getContext(), targetAttr);
  if (failed(chipset)) {
    return variantOp.emitError() << "failed to parse amdgpu chipset";
  }

  if (chipset->majorVersion != 9 || *chipset < amdgpu::Chipset(9, 4, 0)) {
    return success();
  }

  auto inRegAttrName =
      builder.getStringAttr(LLVM::LLVMDialect::getInRegAttrName());
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    funcOp.setArgAttr(i, inRegAttrName, unitAttr);
  }

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
    if (!exportOp) {
      return;
    }

    if (failed(annotateKernelForTranslation(funcOp, variantOp, exportOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
