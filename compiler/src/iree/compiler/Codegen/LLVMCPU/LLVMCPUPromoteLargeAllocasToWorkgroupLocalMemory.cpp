// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Moves oversized static stack storage into HAL workgroup-local scratch memory.
//
// This pass only promotes from the entry block of the function.
// This is enough, since alloca operations must be hoisted to the entry block
// (cf LLVMCPUCheckIRBeforeLLVMConversion).

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUPROMOTELARGEALLOCASTOWORKGROUPLOCALMEMORYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

static bool canPromote(memref::AllocaOp op) {
  MemRefType type = op.getType();
  if (!type.hasStaticShape() || type.getMemorySpace() ||
      !op.getSymbolOperands().empty()) {
    return false;
  }
  SmallVector<int64_t> strides;
  int64_t offset;

  // Check the same constraints as the LLVMCPUAssignWorkgroupLocalMemory pass
  if (failed(type.getStridesAndOffset(strides, offset)) || offset < 0 ||
      llvm::any_of(strides, [](int64_t stride) { return stride < 0; })) {
    return false;
  }
  return !DataLayout::closest(op)
              .getTypeSize(type.getElementType())
              .isScalable();
}

struct LLVMCPUPromoteLargeAllocasToWorkgroupLocalMemoryPass final
    : impl::LLVMCPUPromoteLargeAllocasToWorkgroupLocalMemoryPassBase<
          LLVMCPUPromoteLargeAllocasToWorkgroupLocalMemoryPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    if (funcOp.getFunctionBody().empty() || !getDispatchConfigOp(funcOp)) {
      return;
    }

    struct Candidate {
      memref::AllocaOp op;
      int64_t size;
    };
    SmallVector<Candidate> candidates;
    int64_t total = 0;
    WalkResult result = funcOp.walk([&](memref::AllocaOp op) {
      // We need to only walk the entry block of the function (where allocas
      // are)
      if (op->getBlock() != &funcOp.getFunctionBody().front()) {
        return WalkResult::interrupt();
      }
      FailureOr<int64_t> size =
          getStackAllocationSize(op, /*emitDiagnostics=*/false);
      if (failed(size) || llvm::AddOverflow(total, *size, total)) {
        return WalkResult::interrupt();
      }
      if (canPromote(op)) {
        candidates.push_back({op, *size});
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return;
    }

    int64_t limit = getStackAllocationLimit(funcOp);
    llvm::stable_sort(candidates,
                      [](const Candidate &lhs, const Candidate &rhs) {
                        return lhs.size > rhs.size;
                      });
    IRRewriter rewriter(funcOp.getContext());
    for (const Candidate &candidate : candidates) {
      if (total <= limit) {
        break;
      }
      memref::AllocaOp op = candidate.op;
      MemRefType originalType = op.getType();
      auto localType = MemRefType::get(
          originalType.getShape(), originalType.getElementType(),
          originalType.getLayout(),
          IREE::Codegen::WorkgroupLocalMemoryAttr::get(funcOp.getContext()));
      rewriter.setInsertionPoint(op);
      Value alloc = memref::AllocOp::create(rewriter, op.getLoc(), localType);
      rewriter.replaceOpWithNewOp<memref::MemorySpaceCastOp>(op, originalType,
                                                             alloc);
      total -= candidate.size;
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
