// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensions.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE {

transform_dialect::IREEGPUExtensions::IREEGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensionsOps.cpp.inc"
      >();
}

//===---------------------------------------------------------------------===//
// ApplyDropMultiMmaOpUnitDims
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyDropMultiMmaOpUnitDims::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPUDropUnitDimsPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ApplyLowerMultiMmaOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyLowerMultiMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPULowerMultiMmaPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ApplyLowerBarrierRegionPatternsOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyLowerBarrierRegionPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  GPU::populateIREEGPULowerBarrierRegionPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ApplyLowerValueBarrierOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyLowerValueBarrierOp::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPULowerValueBarrierPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ApplyUnrollMultiMmaOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyUnrollMultiMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  GPU::populateIREEGPUVectorUnrollPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ApplyVectorizeIREEGPUOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyVectorizeIREEGPUOp::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPUVectorizationPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ConvertToMultiMmaOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::ConvertToMultiMmaOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  auto multiMmaOp =
      GPU::convertContractionToMultiMma(rewriter, target, getIntrinsicKind());
  if (failed(multiMmaOp)) {
    return mlir::emitDefiniteFailure(target, "conversion to multi_mma failed");
  }
  results.push_back(*multiMmaOp);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ConvertToMultiMmaOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// DistributeMultiMmaOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::DistributeMultiMmaOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(target);
  if (!mmaOp) {
    return mlir::emitDefiniteFailure(target, "target is not a multi_mma op");
  }
  rewriter.setInsertionPoint(mmaOp);
  auto maybeForall = IREE::GPU::distributeMultiMmaOp(rewriter, mmaOp);
  if (failed(maybeForall)) {
    return mlir::emitDefiniteFailure(mmaOp, "multi_mma distribution failed");
  }
  results.push_back(*maybeForall);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::DistributeMultiMmaOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// ForallToLanesOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::ForallToLanesOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  IREE::GPU::mapLaneForalls(rewriter, target, /*insertBarrier=*/true);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ForallToLanesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// FuseForallOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::FuseForallOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  auto producers = state.getPayloadOps(getProducer());
  auto consumers = state.getPayloadOps(getConsumer());

  int64_t numProducers = llvm::range_size(producers);
  int64_t numConsumers = llvm::range_size(consumers);
  if (numProducers != 1 || numConsumers != 1) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "More than one producer or consumer");
  }

  auto producer = dyn_cast<scf::ForallOp>(*producers.begin());
  auto consumer = dyn_cast<scf::ForallOp>(*consumers.begin());
  if (!producer || !consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non-forall producer or consumer");
  }

  tensor::ExtractSliceOp sliceConsumer;
  Operation *currProducer = producer;

  SmallVector<Operation *> consumerChain;
  while (currProducer->hasOneUse()) {
    Operation *nextConsumer = *currProducer->user_begin();
    if (auto maybeSlice = dyn_cast<tensor::ExtractSliceOp>(nextConsumer)) {
      sliceConsumer = maybeSlice;
      consumerChain.push_back(sliceConsumer);
      break;
    }
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(nextConsumer)) {
      consumerChain.push_back(nextConsumer);
      currProducer = nextConsumer;
      continue;
    }
  }

  if (!sliceConsumer || sliceConsumer->getParentOp() != consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "producer loop not consumed by single "
                                     "extracted slice from the consumer loop");
  }

  if (failed(GPU::fuseForallIntoConsumer(rewriter, producer, consumer,
                                         consumerChain))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "failed to fuse forall ops");
  }

  results.set(getOperation()->getOpResult(0), {consumer});
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::FuseForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getProducerMutable(), effects);
  transform::consumesHandle(getConsumerMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// FuseCollapseShapeIntoForallOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::FuseCollapseShapeIntoForallOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto producers = state.getPayloadOps(getProducer());
  auto consumers = state.getPayloadOps(getConsumer());

  int64_t numProducers = llvm::range_size(producers);
  int64_t numConsumers = llvm::range_size(consumers);
  if (numProducers != 1 || numConsumers != 1) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "More than one producer or consumer");
  }

  auto producer = dyn_cast<scf::ForallOp>(*producers.begin());
  if (!producer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non-forall producer");
  }
  auto consumer = dyn_cast<tensor::CollapseShapeOp>(*consumers.begin());
  if (!consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non-collapse_shape consumer");
  }

  FailureOr<scf::ForallOp> fusedForallOp =
      GPU::fuseCollapseShapeIntoProducerForall(rewriter, producer, consumer);
  if (failed(fusedForallOp)) {
    return mlir::emitSilenceableFailure(state.getTopLevel(),
                                        "failed to fuse collapse_shape op");
  }

  results.set(getOperation()->getOpResult(0), {fusedForallOp.value()});
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::FuseCollapseShapeIntoForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getProducerMutable(), effects);
  transform::consumesHandle(getConsumerMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// FuseExtractSliceIntoForallOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::FuseExtractSliceIntoForallOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto producers = state.getPayloadOps(getProducer());
  auto consumers = state.getPayloadOps(getConsumer());

  int64_t numProducers = llvm::range_size(producers);
  int64_t numConsumers = llvm::range_size(consumers);
  if (numProducers != 1 || numConsumers != 1) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "More than one producer or consumer");
  }

  auto producer = dyn_cast<scf::ForallOp>(*producers.begin());
  if (!producer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non-forall producer");
  }
  auto consumer = dyn_cast<tensor::ExtractSliceOp>(*consumers.begin());
  if (!consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non-extract_slice consumer");
  }

  FailureOr<scf::ForallOp> fusedForallOp =
      GPU::fuseExtractSliceIntoProducerForall(rewriter, producer, consumer);
  if (failed(fusedForallOp)) {
    return mlir::emitSilenceableFailure(*this,
                                        "failed to fuse extract_slice op");
  }

  results.set(getOperation()->getOpResult(0), {fusedForallOp.value()});
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::FuseExtractSliceIntoForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getProducerMutable(), effects);
  transform::consumesHandle(getConsumerMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

} // namespace mlir::iree_compiler::IREE

void mlir::iree_compiler::registerTransformDialectIREEGPUExtension(
    DialectRegistry &registry) {
  registry.addExtensions<
      mlir::iree_compiler::IREE::transform_dialect::IREEGPUExtensions>();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensionsOps.cpp.inc"
