// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-llvmgpu-convolution-to-igemm"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONVOLUTIONTOIGEMMPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Function for setting lowering configurations on contractions resulting from
/// the IGEMM transformation. This currently uses the TileAndFuse pipeline, and
/// tries to target MMA intrinsics.
static LogicalResult llvmgpuConfigFn(linalg::GenericOp genericOp,
                                     IREE::LinalgExt::Im2colOp im2colOp) {
  auto funcOp = genericOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp) {
    return genericOp.emitError("cannot find parent funcOp");
  }
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target) {
    return funcOp.emitError("missing GPU target in parent funcOp");
  }
  if (failed(IREE::GPU::setMatmulLoweringConfig(target, funcOp, genericOp))) {
    return IREE::GPU::setTileAndFuseLoweringConfig(target, funcOp, genericOp);
  }
  return success();
}

static bool llvmgpuControlFn(Operation *op) {
  // Do not convert anything that already has a lowering configuration.
  if (getLoweringConfig(op)) {
    return false;
  }
  return true;
}

struct LLVMGPUConvolutionToIGEMMPass final
    : impl::LLVMGPUConvolutionToIGEMMPassBase<LLVMGPUConvolutionToIGEMMPass> {
  using impl::LLVMGPUConvolutionToIGEMMPassBase<
      LLVMGPUConvolutionToIGEMMPass>::LLVMGPUConvolutionToIGEMMPassBase;

  void runOnOperation() override;
};

void LLVMGPUConvolutionToIGEMMPass::runOnOperation() {
  if (failed(convertToIGEMMAndSetConfig(getOperation(), llvmgpuConfigFn,
                                        llvmgpuControlFn))) {
    return signalPassFailure();
  }
}

} // namespace
} // namespace mlir::iree_compiler
