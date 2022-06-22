// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TransformDialectLLVMGPUExtensions.h"

#include "iree-dialects/Transforms/Functional.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::TransformDialectLLVMGPUExtensions::
    TransformDialectLLVMGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectLLVMGPUExtension(
    DialectRegistry &registry) {
  registry
      .addExtensions<transform_dialect::TransformDialectLLVMGPUExtensions>();
}

// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?

static FailureOr<SmallVector<int64_t>> joinStaticRanges(
    ArrayRef<scf::ForeachThreadOp> ops) {
  SmallVector<int64_t> joinedStaticRange = {1, 1, 1};
  for (auto op : ops) {
    for (auto en : llvm::enumerate(op.getNumThreads())) {
      auto maybeInt = getConstantIntValue(en.value());
      if (!maybeInt) {
        op->emitError(
            "scf.foreach_thread with non-constant workgroup size cannot be "
            "lowered into the translation_info attr");
        return failure();
      }
      joinedStaticRange[en.index()] =
          std::max(joinedStaticRange[en.index()], *maybeInt);
    }
  }
  return joinedStaticRange;
}

//===---------------------------------------------------------------------===//
// Patterns for ForeachThreadToGpu rewrite.
//===---------------------------------------------------------------------===//

struct ForeachThreadToGpuRewriter
    : public OpRewritePattern<scf::ForeachThreadOp> {
  ForeachThreadToGpuRewriter(ArrayRef<int64_t> globalWorkgroupSizes,
                             MLIRContext *context)
      : OpRewritePattern<scf::ForeachThreadOp>(context),
        globalWorkgroupSizes(globalWorkgroupSizes.begin(),
                             globalWorkgroupSizes.end()) {}

  FailureOr<SmallVector<Value>> returningMatchAndRewrite(
      scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(foreachThreadOp, rewriter);
  }

  SmallVector<int64_t> globalWorkgroupSizes;
};

FailureOr<SmallVector<Value>>
ForeachThreadToGpuRewriter::returningMatchAndRewrite(
    scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) const {
  if (foreachThreadOp.getNumResults() > 0)
    return foreachThreadOp->emitError(
        "only bufferized scf.foreach_thread lowers to gpu.thread");
  if (foreachThreadOp.getNumThreads().size() > 3)
    return foreachThreadOp->emitError(
        "scf.foreach_thread with rank > 3 does not lower to gpu.thread");

  auto maybeWorkgroupSizes = joinStaticRanges({foreachThreadOp});
  assert(succeeded(maybeWorkgroupSizes) && "unexpected dynamic workgroup size");

  auto workgroupSizes = *maybeWorkgroupSizes;

  // Step 1. Create the gpu.thread ops
  Location loc = foreachThreadOp.getLoc();
  IndexType indexType = rewriter.getIndexType();
  SmallVector<Value> threadCount = foreachThreadOp.getNumThreads();
  SmallVector<gpu::Dimension, 3> gpuDims{gpu::Dimension::x, gpu::Dimension::y,
                                         gpu::Dimension::z};
  SmallVector<Value, 3> threadOps;
  threadCount.resize(3, rewriter.create<arith::ConstantIndexOp>(loc, 1));
  for (int64_t idx : llvm::seq<int64_t>(0, threadCount.size())) {
    threadOps.push_back(
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpuDims[idx]));
  }

  // Step 2. Maybe create conditionals to predicate the region.
  Value predicate;
  for (auto it : llvm::zip(threadOps, workgroupSizes, globalWorkgroupSizes)) {
    auto threadId = std::get<0>(it);
    auto workgroupSize = std::get<1>(it);
    auto globalWorkgroupSize = std::get<2>(it);
    assert(workgroupSize <= globalWorkgroupSize && "workgroup size overflow");
    if (workgroupSize == globalWorkgroupSize) continue;
    Value tmpPredicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadId,
        rewriter.create<arith::ConstantIndexOp>(loc, workgroupSize));
    predicate =
        predicate ? rewriter.create<arith::AndIOp>(loc, predicate, tmpPredicate)
                  : tmpPredicate;
  }

  // Step 3. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 3.a. If predicated, move at the beginning.
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 3.a. Otherwise, move inline just before foreachThreadOp.
    targetBlock = foreachThreadOp->getBlock();
    insertionPoint = Block::iterator(foreachThreadOp);
  }
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 4. RAUW thread indices to thread ops.
  for (auto it : llvm::zip(foreachThreadOp.getThreadIndices(), threadOps))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  // Step 5. syncthreads.
  rewriter.create<gpu::BarrierOp>(loc);

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return threadCount;
}

//===---------------------------------------------------------------------===//
// IREE-specific LLVMGPU transformations.
//===---------------------------------------------------------------------===//

// TODO: if the number of threads was wired like the workgroup_count, we could
// reuse most of the code and not require a static number of threads.
// TODO: synchronizations for imperfectly nested stuff.
DiagnosedSilenceableFailure
transform_dialect::ForeachThreadToGpuAndTranslationInfo::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (!isa<HAL::ExecutableVariantOp>(state.getTopLevel())) {
    emitOpError() << "applies only to HAL::ExecutableVariantOp targets";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  DenseMap<func::FuncOp, SmallVector<scf::ForeachThreadOp>>
      foreachThreadsPerFunc;
  DenseMap<scf::ForeachThreadOp, func::FuncOp> foreachThreadToFunc;
  WalkResult walkResult = state.getTopLevel()->walk<WalkOrder::PostOrder>(
      [&](scf::ForeachThreadOp op) {
        if (op->getParentOfType<scf::ForeachThreadOp>())
          return WalkResult::interrupt();

        auto funcOp = op->getParentOfType<func::FuncOp>();
        foreachThreadToFunc.insert(std::make_pair(op, funcOp));
        auto it = foreachThreadsPerFunc.find(funcOp);
        if (it == foreachThreadsPerFunc.end())
          foreachThreadsPerFunc.insert(
              std::make_pair(funcOp, SmallVector<scf::ForeachThreadOp>{op}));
        else
          it->second.push_back(op);
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();

  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps;
  state.getTopLevel()->walk(
      [&](IREE::HAL::ExecutableExportOp op) { exportOps[op.sym_name()] = op; });

  walkResult =
      state.getTopLevel()->walk<WalkOrder::PostOrder>([&](func::FuncOp funcOp) {
        auto maybeJoinedRanges =
            joinStaticRanges(foreachThreadsPerFunc.lookup(funcOp));
        if (failed(maybeJoinedRanges)) return WalkResult::interrupt();
        for (auto foreachThreadOp : foreachThreadsPerFunc.lookup(funcOp)) {
          auto workgroupSize = functional::applyReturningPatternAt(
              ForeachThreadToGpuRewriter(*maybeJoinedRanges, getContext()),
              foreachThreadOp);
          if (failed(workgroupSize)) return WalkResult::interrupt();
        }
        auto exportOp = exportOps.lookup(funcOp.getName());
        OpBuilder b(exportOp);
        auto newAttr = b.getIndexArrayAttr(*maybeJoinedRanges);
        exportOp->setAttr(exportOp.workgroup_sizeAttrName(), newAttr);
        return WalkResult::advance();
      });

  return walkResult.wasInterrupted()
             ? DiagnosedSilenceableFailure::definiteFailure()
             : DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
