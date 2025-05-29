// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmcpu-forall-to-for"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUFORALLTOFORPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

LogicalResult forallToForLoopIterArgs(RewriterBase &rewriter,
                                      scf::ForallOp forallOp) {

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);
  SmallVector<Value> iterArgs = forallOp.getOutputs();

  bool unsupportedOperation = false;
  auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                       ValueRange args) -> scf::ValueVector {
    // Inline `scf.forall` body excluding terminating `scf.forall.in_parallel`
    // op.
    IRMapping map;
    map.map(forallOp.getInductionVars(), ivs);
    map.map(forallOp.getRegionOutArgs(), args);
    for (auto &op : forallOp.getBody()->without_terminator()) {
      builder.clone(op, map);
    }

    // Process the `scf.forall.in_parallel` terminator,
    auto terminator = forallOp.getTerminator();
    SmallVector<Value> yieldedValues;

    for (auto &yieldOp : terminator.getYieldingOps()) {
      // Convert tensor.parallel_insert_slice to tensor.insert_slice
      if (auto parallelInsert =
              dyn_cast<tensor::ParallelInsertSliceOp>(&yieldOp)) {

        auto insertSlice = builder.create<tensor::InsertSliceOp>(
            parallelInsert.getLoc(),
            map.lookupOrDefault(parallelInsert.getSource()),
            map.lookupOrDefault(parallelInsert.getDest()),
            llvm::map_to_vector(
                parallelInsert.getOffsets(),
                [&](Value v) -> Value { return map.lookupOrDefault(v); }),
            llvm::map_to_vector(
                parallelInsert.getSizes(),
                [&](Value v) -> Value { return map.lookupOrDefault(v); }),
            llvm::map_to_vector(
                parallelInsert.getStrides(),
                [&](Value v) -> Value { return map.lookupOrDefault(v); }),
            parallelInsert.getStaticOffsets(), parallelInsert.getStaticSizes(),
            parallelInsert.getStaticStrides());
        yieldedValues.push_back(insertSlice.getResult());
      } else {
        forallOp.emitError("unsupported operation in in_parallel region");
        unsupportedOperation = true;
        return args;
      }
    }

    return yieldedValues;
  };

  scf::LoopNest loopNest = scf::buildLoopNest(rewriter, forallOp->getLoc(), lbs,
                                              ubs, steps, iterArgs, buildBody);

  if (unsupportedOperation) {
    return failure();
  }

  rewriter.replaceOp(forallOp, loopNest.results);

  return success();
}

struct LLVMCPUForallToForPass
    : impl::LLVMCPUForallToForPassBase<LLVMCPUForallToForPass> {
  using impl::LLVMCPUForallToForPassBase<
      LLVMCPUForallToForPass>::LLVMCPUForallToForPassBase;
  void runOnOperation() override {
    auto funcOp = getOperation();

    IRRewriter rewriter(funcOp->getContext());

    funcOp->walk([&](scf::ForallOp forallOp) {
      // Forall ops with workgroup mappings `#iree_codegen.workgroup_mapping<y>`
      // are for workgroup distribution, we only want to convert inner loops
      // produced by tiling to `scf.for`.
      if (!forallOp.getMapping()) {
        if (failed(forallToForLoopIterArgs(rewriter, forallOp))) {
          signalPassFailure();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
