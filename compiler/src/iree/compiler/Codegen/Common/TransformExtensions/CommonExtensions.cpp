// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommonExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::CommonExtensions::CommonExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectCommonExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::CommonExtensions>();
}

//===---------------------------------------------------------------------===//
// ApplyPatternsOp
//===---------------------------------------------------------------------===//

static void addRankReducingPatterns(RewritePatternSet &patterns) {
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  linalg::populateFoldUnitExtentDimsPatterns(patterns);
}

static void addAllRegisteredCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
}

DiagnosedSilenceableFailure transform_dialect::ApplyPatternsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError(
        "applies only to isolated-from-above targets because it needs to apply "
        "patterns greedily");
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  }
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  if (getCanonicalization()) addAllRegisteredCanonicalizationPatterns(patterns);
  if (getRankReducing()) addRankReducingPatterns(patterns);

  TrackingListener listener(state);
  GreedyRewriteConfig config;
  LogicalResult result = applyPatternsAndFoldGreedily(
      target, std::move(patterns), config, &listener);
  LogicalResult listenerResult = listener.checkErrorState();
  if (failed(result) || failed(listenerResult))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.assign({target});
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// IREEBufferizeOp
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
// TODO: Maybe bufferize should have a separate cpu and a gpu version. This is
// unclear though: what happens on heterogeneous HW ?
//===---------------------------------------------------------------------===//

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  return success();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static FailureOr<Value> gpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  MemRefType allocType = MemRefType::get(memRefType.getShape(),
                                         memRefType.getElementType(), {}, 3);
  return builder
      .create<memref::AllocOp>(loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult gpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  builder.create<memref::DeallocOp>(loc, allocation);
  return success();
}

static LogicalResult gpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

DiagnosedSilenceableFailure transform_dialect::IREEBufferizeOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  PassManager pm(getContext());
  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  if (getTargetGpu()) {
    allocationFn = gpuComprehensiveBufferizeAllocationFn;
    deallocationFn = gpuComprehensiveBufferizeDeallocationFn;
    memcpyFn = gpuComprehensiveBufferizeCopyFn;
  }
  mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
      pm, allocationFn, deallocationFn, memcpyFn);
  WalkResult res = state.getTopLevel()->walk([&](ModuleOp moduleOp) {
    if (failed(pm.run(moduleOp))) {
      getOperation()->emitError()
          << "failed to bufferize ModuleOp:\n"
          << *(moduleOp.getOperation()) << "\nunder top-level:\n"
          << *state.getTopLevel();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return DiagnosedSilenceableFailure(failure(res.wasInterrupted()));
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
