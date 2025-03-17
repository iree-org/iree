// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>
#include <numeric>

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUVERIFYVECTORSIZELEGALITYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
struct LLVMCPUVerifyVectorSizeLegalityPass
    : impl::LLVMCPUVerifyVectorSizeLegalityPassBase<
          LLVMCPUVerifyVectorSizeLegalityPass> {
  using impl::LLVMCPUVerifyVectorSizeLegalityPassBase<
      LLVMCPUVerifyVectorSizeLegalityPass>::
      LLVMCPUVerifyVectorSizeLegalityPassBase;

  void runOnOperation() override;
};
} // namespace

static int64_t getTotalSizeInBytes(Type elemType, ArrayRef<int64_t> shape) {
  // We can't query bitwidth for some types (e.g., index type). For those
  // cases, we assume that they are 64 bits because most of modern systems are
  // 64-bit.
  int64_t elemBitWidth = 64;
  if (elemType.isIntOrFloat()) {
    elemBitWidth = elemType.getIntOrFloatBitWidth();
  }
  int64_t size = std::accumulate(shape.begin(), shape.end(), elemBitWidth,
                                 std::multiplies<int64_t>{});
  constexpr int64_t kBitsInByte = 8;
  size = llvm::divideCeil(size, kBitsInByte);
  return size;
}

void LLVMCPUVerifyVectorSizeLegalityPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  int64_t maxVectorSizeInBytes =
      getMaxVectorSizeForLargeVectorCheck(targetAttr);

  auto checkFn = [&](Type t) {
    auto vectorType = dyn_cast<VectorType>(t);
    if (!vectorType) {
      return false;
    }
    int64_t size =
        getTotalSizeInBytes(vectorType.getElementType(), vectorType.getShape());
    return size >= maxVectorSizeInBytes;
  };
  auto isLargeVectorContract = [&](vector::ContractionOp op) {
    SmallVector<int64_t> iterationBounds;
    op.getIterationBounds(iterationBounds);
    int64_t size = getTotalSizeInBytes(getElementTypeOrSelf(op.getAccType()),
                                       iterationBounds);
    return size >= maxVectorSizeInBytes;
  };

  SmallVector<Operation *> invalidOps;
  funcOp.walk([&](Operation *op) {
    auto contractOp = dyn_cast<vector::ContractionOp>(op);
    if (contractOp && isLargeVectorContract(contractOp)) {
      invalidOps.push_back(op);
      return;
    }

    SmallVector<Type> types(op->getOperandTypes());
    llvm::append_range(types, op->getResultTypes());
    if (llvm::any_of(types, checkFn)) {
      invalidOps.push_back(op);
    }
  });
  if (invalidOps.empty()) {
    return;
  }

  // Error fall-through. Attach all reported issues as notes.
  InFlightDiagnostic errorDiag =
      emitError(funcOp.getLoc())
      << "One or more operations with large vector sizes ("
      << maxVectorSizeInBytes << " bytes) were found:\n";
  for (Operation *op : invalidOps) {
    errorDiag.attachNote(op->getLoc()) << "  " << *op << "\n";
  }

  signalPassFailure();
}

} // namespace mlir::iree_compiler
