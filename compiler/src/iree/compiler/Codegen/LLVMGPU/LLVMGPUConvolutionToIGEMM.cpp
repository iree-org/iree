// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
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

/// Get the equivalent contraction problem from a conv_2d op. The N, H, and W
/// dimensions become M dimensions; P * Q * C becomes K; and F becomes N.
template <typename ConvOpTy>
static GPUMatmulShapeType getConvProblem(ConvOpTy convOp) {
  static_assert(llvm::is_one_of<ConvOpTy, linalg::Conv2DNchwFchwOp,
                                linalg::Conv2DNhwcHwcfOp>::value,
                "expected nchw conv or nhwc conv op");
  const bool isNchw = std::is_same_v<ConvOpTy, linalg::Conv2DNchwFchwOp>;
  auto inputType = cast<RankedTensorType>(convOp.getOperandTypes()[0]);
  auto accType = cast<RankedTensorType>(convOp.getResultTypes()[0]);
  ArrayRef<int64_t> accShape = accType.getShape();
  auto filterType = cast<RankedTensorType>(convOp.getOperandTypes()[1]);
  ArrayRef<int64_t> filterShape = filterType.getShape();
  // Result dims
  const int64_t NDim = 0;
  const int64_t HDim = isNchw ? 2 : 1;
  const int64_t WDim = isNchw ? 3 : 2;
  // Filter dims
  const int64_t CDim = isNchw ? 1 : 2;
  const int64_t PDim = isNchw ? 2 : 0;
  const int64_t QDim = isNchw ? 3 : 1;
  const int64_t FDim = isNchw ? 0 : 3;
  SmallVector<int64_t> mSizes, nSizes, kSizes;
  mSizes.append({accShape[NDim], accShape[HDim], accShape[WDim]});
  kSizes.append({filterShape[CDim] * filterShape[PDim] * filterShape[QDim]});
  nSizes.append({filterShape[FDim]});
  auto filterDynamicSizes = [](SmallVector<int64_t> sizes) {
    SmallVector<int64_t> filteredSizes;
    for (auto size : sizes)
      if (!ShapedType::isDynamic(size))
        filteredSizes.push_back(size);
    return filteredSizes;
  };
  Type lhsElemType = inputType.getElementType();
  Type rhsElemType = filterType.getElementType();
  Type accElemType = accType.getElementType();
  GPUMatmulShapeType problem{filterDynamicSizes(mSizes),
                             filterDynamicSizes(nSizes),
                             filterDynamicSizes(kSizes),
                             lhsElemType,
                             rhsElemType,
                             accElemType};
  return problem;
}

/// Control function for using IGEMM. For now, don't use IGEMM for any conv
/// that would not be able to target MMA intrinsics.
/// TODO(Max191): Remove the restriction for hitting MMA intrinsics once the
/// TileAndFuse pipeline can target intrinsics for unaligned cases.
static bool llvmgpuControlFn(Operation *op) {
  // Do not convert anything that already has a lowering configuration.
  if (getLoweringConfig(op)) {
    return false;
  }
  auto convNchw = dyn_cast<linalg::Conv2DNchwFchwOp>(op);
  auto convNhwc = dyn_cast<linalg::Conv2DNhwcHwcfOp>(op);
  if (!convNchw && !convNhwc) {
    return false;
  }
  GPUMatmulShapeType problem =
      convNchw ? getConvProblem(convNchw) : getConvProblem(convNhwc);
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  if (!funcOp)
    return false;
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target)
    return false;
  std::optional<GPUMMASchedule> schedule = getMmaScheduleFromProblemAndTarget(
      target, problem, /*transposedLhs=*/static_cast<bool>(convNchw),
      /*transposedRhs=*/static_cast<bool>(convNchw));
  return schedule.has_value();
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
