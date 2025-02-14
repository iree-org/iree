// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "iree-codegen-amdgpu-configure-buffer-instructions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPUCONFIGUREBUFFERINSTRUCTIONSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static llvm::cl::opt<bool> clAMDGPUEnableBufferInstructions(
    "iree-codegen-amdgpu-enable-buffer-instructions",
    llvm::cl::desc("Use buffer instructions (by using buffer fat pointers) "
                   "where possible on AMD targets"),
    llvm::cl::init(true));

/// Determine if `arg` is an arithmetic function of constants or HAL constant
/// loads, which is a conservative approximatino for workgroup-uniformity that
/// can be made more extensive if needed.
static bool isDefinitelyWorkgroupUniform(Value arg) {
  SmallVector<Value> worklist;
  worklist.push_back(arg);
  while (!worklist.empty()) {
    Value thisVal = worklist.pop_back_val();
    Operation *thisOp = thisVal.getDefiningOp();
    if (!thisOp)
      continue; // block arguments
    if (isa<IREE::HAL::InterfaceConstantLoadOp>(thisOp))
      continue;
    if (matchPattern(thisOp, m_Constant()))
      continue;
    if (auto assumeOp = dyn_cast<IREE::Util::AssumeIntOp>(thisOp)) {
      worklist.push_back(
          assumeOp.getOperand(cast<OpResult>(thisVal).getResultNumber()));
      continue;
    }
    if (isa<arith::ArithDialect>(thisOp->getDialect())) {
    }
    { llvm::append_range(worklist, thisOp->getOperands()); }
    return false;
  }
  return true;
}

/// Return the maximum value that has been `util.assume.int`'d about this value
/// if there is one.
/// TODO: it'd be nice to be able to run the IntRangeAnalysis just up to the
/// value in question, but we don't have that, so we approximate it.
static std::optional<int64_t> getDynamicSizeMax(Value size) {
  while (
      isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
                      arith::ExtUIOp>(size.getDefiningOp())) {
    size = size.getDefiningOp()->getOperand(0);
  }
  // Special case for constants that're still dynamic.
  APInt constVal;
  if (matchPattern(size, m_ConstantInt(&constVal))) {
    return constVal.getZExtValue();
  }
  auto assumeOp = size.getDefiningOp<IREE::Util::AssumeIntOp>();
  if (!assumeOp)
    return std::nullopt;
  std::optional<int64_t> maybeMax =
      assumeOp.getUnionedUnsignedRange(cast<OpResult>(size).getResultNumber())
          .second;
  return maybeMax;
}

static std::optional<int64_t>
getSpannedBytes(IREE::HAL::InterfaceBindingSubspanOp binding) {
  int64_t maxNumElems = 1;
  auto resultTy = dyn_cast<ShapedType>(binding.getType());
  if (!resultTy || !resultTy.hasRank())
    return std::nullopt;
  for (Value dynArg : binding.getResultDynamicDims(0)) {
    std::optional<int64_t> dimMax = getDynamicSizeMax(dynArg);
    if (!dimMax)
      return std::nullopt;
    maxNumElems *= (*dimMax);
  }
  for (int64_t dim : resultTy.getShape()) {
    if (ShapedType::isDynamic(dim))
      continue;
    maxNumElems *= dim;
  }
  return maxNumElems * IREE::Util::getTypeBitWidth(resultTy.getElementType()) /
         8;
}

namespace {

struct AMDGPUConfigureBufferInstructionsPass final
    : impl::AMDGPUConfigureBufferInstructionsPassBase<
          AMDGPUConfigureBufferInstructionsPass> {
  void runOnOperation() override {
    if (!clAMDGPUEnableBufferInstructions)
      return;
    FunctionOpInterface func = getOperation();
    auto *gpuDialect =
        getContext().getLoadedDialect<IREE::GPU::IREEGPUDialect>();
    auto annotationHelper =
        gpuDialect->getUseAmdgpuBufferInstructionsAttrHelper();
    auto unitAttr = UnitAttr::get(&getContext());
    func.walk([&](IREE::HAL::InterfaceBindingSubspanOp binding) {
      if (!isDefinitelyWorkgroupUniform(binding.getByteOffset())) {
        LDBG("Binding offset " << binding.getByteOffset()
                               << " not known workgroup-uniform\n");
        return;
      }
      std::optional<int64_t> maxBytes = getSpannedBytes(binding);
      if (!maxBytes) {
        LDBG("Couldn't bound binding size for " << binding);
        return;
      }
      if (*maxBytes >=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        LDBG("Size of " << binding << " too large (" << *maxBytes << " bytes)");
        return;
      }
      annotationHelper.setAttr(binding, unitAttr);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler
