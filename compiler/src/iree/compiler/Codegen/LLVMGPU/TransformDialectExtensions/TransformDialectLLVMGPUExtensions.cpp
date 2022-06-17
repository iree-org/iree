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

//===---------------------------------------------------------------------===//
// Patterns for ForeachThreadToGpu rewrite.
//===---------------------------------------------------------------------===//

struct ForeachThreadToGpuRewriter
    : public OpRewritePattern<scf::ForeachThreadOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<SmallVector<Value>> returningMatchAndRewrite(
      scf::ForeachThreadOp foreachThreadOp, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(scf::ForeachThreadOp foreachThreadOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(foreachThreadOp, rewriter);
  }
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

  // Step 1. Create the gpu.thread ops
  Location loc = foreachThreadOp.getLoc();
  IndexType indexType = rewriter.getIndexType();
  SmallVector<Value> threadCount = foreachThreadOp.getNumThreads();
  SmallVector<gpu::Dimension, 3> gpuDims{gpu::Dimension::x, gpu::Dimension::y,
                                         gpu::Dimension::z};
  SmallVector<Value> threadOps;
  threadOps.reserve(3);
  for (int64_t idx : llvm::seq<int64_t>(0, threadCount.size())) {
    threadOps.push_back(
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpuDims[idx]));
  }
  threadCount.resize(3, rewriter.create<arith::ConstantIndexOp>(loc, 1));

  // Step 2. Move the body of foreachThreadOp after the op.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *block = foreachThreadOp->getBlock();
  block->getOperations().splice(
      Block::iterator(foreachThreadOp),
      foreachThreadOp.getRegion().front().getOperations());

  // Step 3. RAUW thread indices to thread ops.
  for (auto it : llvm::zip(foreachThreadOp.getThreadIndices(), threadOps))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

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
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps;
  state.getTopLevel()->walk(
      [&](IREE::HAL::ExecutableExportOp op) { exportOps[op.sym_name()] = op; });

  if (state.getTopLevel()
          ->walk<WalkOrder::PostOrder>([&](scf::ForeachThreadOp op) {
            auto funcOp = op->getParentOfType<func::FuncOp>();
            auto exportOp = exportOps.lookup(funcOp.getName());
            // Step 1. Apply the rewrite.
            auto workgroupSize = functional::applyReturningPatternAt(
                ForeachThreadToGpuRewriter(getContext()), op);
            if (failed(workgroupSize)) return WalkResult::interrupt();

            // Step 2. Ensure the workgroupSize is static and extract it.
            SmallVector<int64_t, 3> staticThreadCount;
            for (auto en : llvm::enumerate(*workgroupSize)) {
              auto maybeInt = getConstantIntValue(en.value());
              if (!maybeInt) {
                op->emitError("scf.foreach_thread rank #")
                    << en.index() << " is not a constant and cannot be lowered "
                    << "into the translation_info attr";
                return WalkResult::interrupt();
              }
              staticThreadCount.push_back(*maybeInt);
            }

            // Step 3. Check and fill the attribute that requires
            OpBuilder b(exportOp);
            auto maybeExistingAttr =
                exportOp->getAttr(exportOp.workgroup_sizeAttrName());
            auto newAttr = b.getIndexArrayAttr(staticThreadCount);
            if (maybeExistingAttr && maybeExistingAttr != newAttr) {
              exportOp->emitError("multiple mismatching workgroup_size ")
                  << "attributes found: " << maybeExistingAttr << " and "
                  << newAttr;
              return WalkResult::interrupt();
            }
            exportOp->setAttr(exportOp.workgroup_sizeAttrName(), newAttr);
            return WalkResult::advance();
          })
          .wasInterrupted())
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/LLVMGPU/TransformDialectExtensions/TransformDialectLLVMGPUExtensionsOps.cpp.inc"
