// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
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
  auto attr = LLVM::DenormalFPEnvAttr::get(ctx, LLVM::DenormalModeKind::IEEE,
                                           LLVM::DenormalModeKind::IEEE,
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
                             IREE::Codegen::DispatchConfigOp configOp) {
  OpBuilder builder(funcOp);
  auto *rocdlDialect =
      funcOp.getContext()->getLoadedDialect<ROCDL::ROCDLDialect>();
  assert(rocdlDialect && "ROCDL dialect not loaded");
  UnitAttr unitAttr = builder.getUnitAttr();
  rocdlDialect->getKernelAttrHelper().setAttr(funcOp, unitAttr);
  std::optional<ArrayRef<int64_t>> wgSize = configOp.getWorkgroupSize();
  if (wgSize && wgSize->size() <= 3) {
    std::array<int32_t, 3> wgSizes;
    int32_t flatWgSize = 1;
    for (auto [value, dim] : llvm::zip_equal(wgSizes, *wgSize)) {
      value = static_cast<int32_t>(dim);
      flatWgSize *= value;
    }
    rocdlDialect->getReqdWorkGroupSizeAttrHelper().setAttr(
        funcOp, builder.getDenseI32ArrayAttr(wgSizes));
    rocdlDialect->getFlatWorkGroupSizeAttrHelper().setAttr(
        funcOp,
        builder.getStringAttr(Twine(flatWgSize) + "," + Twine(flatWgSize)));
  }

  IREE::HAL::ExecutableTargetAttr targetAttr =
      IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
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
    return funcOp.emitError() << "failed to parse amdgpu chipset";
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

struct ROCDLAnnotateKernelForTranslationPass final
    : impl::ROCDLAnnotateKernelForTranslationPassBase<
          ROCDLAnnotateKernelForTranslationPass> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp funcOp = getOperation();
    StringRef funcName = funcOp.getName();

    // Find the matching dispatch_config op in the parent module.
    IREE::Codegen::DispatchConfigOp configOp;
    if (auto moduleOp = funcOp->getParentOfType<ModuleOp>()) {
      for (auto candidate :
           moduleOp.getOps<IREE::Codegen::DispatchConfigOp>()) {
        if (candidate.getFunctionRef() == funcName) {
          configOp = candidate;
          break;
        }
      }
    }

    // Functions without a dispatch_config are library functions or otherwise
    // not kernels, so don't need these annotations.
    if (!configOp) {
      return;
    }

    if (failed(annotateKernelForTranslation(funcOp, configOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
