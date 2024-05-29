// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensions.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
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
// ApplyLowerValueBarrierOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyLowerValueBarrierOp::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPULowerValueBarrierPatterns(patterns);
}

//===---------------------------------------------------------------------===//
// ApplyUnrollMultiMmaOp
//===---------------------------------------------------------------------===//

static bool isReductionIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::reduction;
}
static bool isParallelIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::parallel;
}

/// Pick an unrolling order that reuses the LHS register.
static std::optional<SmallVector<int64_t>>
gpuMultiMmaUnrollOrder(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : mmaOp.getIndexingMapsArray()[0].getResults()) {
    dims.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>> getMultiMmaUnitShape(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> targetOuterShape(mmaOp.getIteratorTypes().size(), 1);
  return targetOuterShape;
}

void transform_dialect::ApplyUnrollMultiMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  GPU::populateIREEGPUVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getMultiMmaUnitShape)
                    .setUnrollTraversalOrderFn(gpuMultiMmaUnrollOrder));
}

//===---------------------------------------------------------------------===//
// ApplyVectorizeMultiMmaOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyVectorizeMultiMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPUVectorizationPatterns(patterns);
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

  if (!producer->hasOneUse()) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "non-single use producer");
  }

  auto sliceConsumer =
      dyn_cast<tensor::ExtractSliceOp>(*producer->user_begin());
  if (!sliceConsumer || sliceConsumer->getParentOp() != consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "producer loop sole consumer is not an "
                                     "extracted slice from the consumer loop");
  }

  if (failed(GPU::fuseForallIntoSlice(rewriter, producer, consumer,
                                      sliceConsumer))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "failed to fuse forall ops");
  }

  results.set(getOperation()->getOpResult(0), {consumer});
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::FuseForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getProducer(), effects);
  transform::consumesHandle(getConsumer(), effects);
  transform::producesHandle(getResult(), effects);
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
