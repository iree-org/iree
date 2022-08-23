// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"

#define DEBUG_TYPE "iree-llvmgpu-distribute"

namespace mlir {
namespace iree_compiler {

/// Clean up barriers if we have no shared memory allocations we expect to not
/// need any barriers and remove them.
static void cleanUpBarriers(func::FuncOp funcOp) {
  bool hasAlloc = false;
  SmallVector<Operation*> barriers;
  funcOp.walk([&](Operation* op) {
    if (isa<memref::AllocOp>(op))
      hasAlloc = true;
    else if (isa<gpu::BarrierOp>(op))
      barriers.push_back(op);
  });
  if (!hasAlloc) {
    for (Operation* op : barriers) op->erase();
  }
}

namespace {
struct LLVMGPUDistributePass
    : public LLVMGPUDistributeBase<LLVMGPUDistributePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    auto workgroupSize = llvm::to_vector(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().getValue(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    SmallVector<scf::ForeachThreadOp> foreachOps;
    funcOp.walk([&](scf::ForeachThreadOp foreachOp) {
      foreachOps.push_back(foreachOp);
    });
    for (scf::ForeachThreadOp op : foreachOps) {
      IRRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      if (failed(rewriteForeachThreadToGpu(op, workgroupSize, rewriter))) {
        return signalPassFailure();
      }
    }

    // Workaround, since we conservatively insert barrier ops, remove them if
    // they are obviously not needed.
    // TODO(thomasraoux): Improve barrier placement.
    cleanUpBarriers(funcOp);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUDistribute() {
  return std::make_unique<LLVMGPUDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
