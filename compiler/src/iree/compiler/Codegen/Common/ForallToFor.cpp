// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
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

#define DEBUG_TYPE "iree-codegen-forall-to-for"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FORALLTOFORPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

LogicalResult forallToForLoop(RewriterBase &rewriter, scf::ForallOp forallOp) {
  rewriter.setInsertionPoint(forallOp);

  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);
  SmallVector<Value> iterArgs = forallOp.getOutputs();

  bool unsupportedOperation = false;
  auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs,
                       ValueRange args) -> scf::ValueVector {
    // Inline `scf.forall` body excluding terminating
    // `scf.forall.in_parallel` op.
    IRMapping map;
    map.map(forallOp.getInductionVars(), ivs);
    map.map(forallOp.getRegionOutArgs(), args);
    for (auto &op : forallOp.getBody()->without_terminator()) {
      builder.clone(op, map);
    }

    // Convert + inline the contents of `scf.forall.in_parallel` terminator.
    auto terminator = forallOp.getTerminator();
    SmallVector<Value> yieldedValues;
    for (auto &yieldOp : terminator.getYieldingOps()) {
      // Convert tensor.parallel_insert_slice to tensor.insert_slice
      if (auto parallelInsert =
              dyn_cast<tensor::ParallelInsertSliceOp>(&yieldOp)) {


auto source = map.lookupOrDefault(parallelInsert.getSource());
auto dest = map.lookupOrDefault(parallelInsert.getDest());
auto offsets = llvm::map_to_vector(parallelInsert.getOffsets(),
    [&](Value v) { return map.lookupOrDefault(v); });
auto sizes = llvm::map_to_vector(parallelInsert.getSizes(),
    [&](Value v) { return map.lookupOrDefault(v); });
auto strides = llvm::map_to_vector(parallelInsert.getStrides(),
    [&](Value v) { return map.lookupOrDefault(v); });
auto insertSlice = builder.create<tensor::InsertSliceOp>(
     parallelInsert.getLoc(), source, dest, offsets, sizes, strides,
    parallelInsert.getStaticOffsets(),
    parallelInsert.getStaticSizes(),
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

struct ForallToForPass : impl::ForallToForPassBase<ForallToForPass> {
  using impl::ForallToForPassBase<ForallToForPass>::ForallToForPassBase;
  void runOnOperation() override {
    auto funcOp = getOperation();

    IRRewriter rewriter(funcOp->getContext());

    // Find `scf.forall` ops we want to convert in innermost to outermost order.
    SmallVector<scf::ForallOp> forallOps;
    funcOp->walk<WalkOrder::PostOrder>([&](scf::ForallOp forallOp) {
      // Forall ops with workgroup mappings `#iree_codegen.workgroup_mapping<y>`
      // are for workgroup distribution, we only want to convert inner loops
      // produced by tiling to `scf.for`.
      if (!forallOp.getMapping()) {
        forallOps.push_back(forallOp);
      }
    });

    // Convert `scf.forall` -> `scf.for`.
    for (auto forallOp : forallOps) {
      if (failed(iree_compiler::forallToForLoop(rewriter, forallOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
