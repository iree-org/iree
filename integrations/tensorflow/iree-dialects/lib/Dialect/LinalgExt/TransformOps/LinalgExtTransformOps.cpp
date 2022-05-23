// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree-dialects/Transforms/Functional.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

LinalgExt::LinalgExtTransformOpsExtension::LinalgExtTransformOpsExtension() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.cpp.inc"
      >();
}

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

/// Extracts a vector of int64_t from an array attribute. Asserts if the
/// attribute contains values other than integers.
static SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getSExtValue());
  return result;
}

//===---------------------------------------------------------------------===//
// FuseProducersOp
//===---------------------------------------------------------------------===//

LogicalResult
LinalgExt::FuseProducersOp::apply(transform::TransformResults &transformResults,
                                  transform::TransformState &state) {
  SmallVector<int64_t> operandsToFuse = extractI64Array(getOperandsToFuse());
  LinalgExt::LinalgExtFusionPattern pattern(getContext(), operandsToFuse);
  size_t numProducers = operandsToFuse.size();

  SmallVector<Operation *> transformedOps;
  SmallVector<SmallVector<Operation *>> fusedOps(numProducers);
  for (Operation *target : state.getPayloadOps(getTarget())) {
    // Apply the pattern.
    FailureOr<LinalgExt::FusionResult> result =
        functional::applyReturningPatternAt(pattern,
                                            cast<linalg::LinalgOp>(target));
    if (failed(result))
      return failure();

    // Update the fused operations.
    transformedOps.push_back(result->consumerOp);
    for (size_t i = 0; i < numProducers; ++i)
      fusedOps[i].push_back(result->fusedOps[i]);
  }

  transformResults.set(getTransformed().cast<OpResult>(), transformedOps);
  for (size_t i = 0; i < numProducers; ++i)
    transformResults.set(getFusedOps()[i], fusedOps[i]);
  return success();
}

LogicalResult LinalgExt::FuseProducersOp::verify() {
  SmallVector<int64_t> operandsToFuse = extractI64Array(getOperandsToFuse());
  llvm::SmallDenseSet<int64_t> operandsSet;
  for (int64_t operandToFuse : operandsToFuse) {
    if (operandToFuse < 0) {
      return emitOpError() << "expects positive operand numbers, found "
                           << operandToFuse;
    }
    if (operandsSet.count(operandToFuse) != 0) {
      return emitOpError() << "expects unique operand numbers, found "
                           << operandToFuse << " multiple times";
    }
    operandsSet.insert(operandToFuse);
  }
  return success();
}

ParseResult LinalgExt::FuseProducersOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  OpAsmParser::UnresolvedOperand targetOperand;
  SMLoc opLoc;
  if (parser.getCurrentLocation(&opLoc))
    return failure();
  if (parser.parseOperand(targetOperand))
    return parser.emitError(opLoc, "expected `target` operand");
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  StringRef operandsToFuseAttrName("operands_to_fuse");
  Attribute operandsToFuseAttr = result.attributes.get(operandsToFuseAttrName);
  if (!operandsToFuseAttr) {
    return parser.emitError(opLoc, llvm::formatv("expected `{0}` attribute",
                                                 operandsToFuseAttrName));
  }
  auto operandsToFuseArrayAttr = operandsToFuseAttr.dyn_cast<ArrayAttr>();
  if (!operandsToFuseArrayAttr) {
    return parser.emitError(opLoc,
                            llvm::formatv("`{0}` attribute must be an array",
                                          operandsToFuseAttrName));
  }
  Type pdlOpType = parser.getBuilder().getType<pdl::OperationType>();
  size_t numProducers = operandsToFuseArrayAttr.size();
  result.addTypes(SmallVector<Type>(numProducers + 1, pdlOpType));
  if (parser.resolveOperand(targetOperand, pdlOpType, result.operands))
    return failure();
  return success();
}

void LinalgExt::FuseProducersOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult
LinalgExt::TileToLinalgExtTileOp::apply(transform::TransformResults &results,
                                        transform::TransformState &state) {
  linalg::LinalgTilingOptions tilingOptions;
  SmallVector<int64_t> tileSizes = extractI64Array(getSizes());
  if (!tileSizes.empty())
    tilingOptions.setTileSizes(tileSizes);

  LinalgExt::LinalgExtTilingPattern pattern(this->getContext(), tilingOptions);
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  auto tilingInterfaceOp = dyn_cast<TilingInterface>(targets.front());
  if (!tilingInterfaceOp) {
    targets.front()->emitError("Cannot tile op: Not a TilingInterface");
    return failure();
  }

  FailureOr<iree_compiler::IREE::LinalgExt::TilingResult> result =
      functional::applyReturningPatternAt(pattern, tilingInterfaceOp);
  if (failed(result))
    return failure();
  results.set(getTiledOp().cast<OpResult>(), result->tiledOp);
  results.set(getTileOp().cast<OpResult>(), result->tileOp.getOperation());
  return success();
}

LogicalResult
LinalgExt::FuseIntoContainingOp::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {

  ArrayRef<Operation *> producerOps = state.getPayloadOps(getProducerOp());
  ArrayRef<Operation *> containingOps = state.getPayloadOps(getContainingOp());
  for (auto it : llvm::zip(producerOps, containingOps)) {
    auto producerOp = dyn_cast<linalg::LinalgOp>(std::get<0>(it));
    Operation *containingOp = std::get<1>(it);
    if (!producerOp) {
      std::get<0>(it)->emitError("Cannot fuse op: Not a LinalgOp");
      return failure();
    }
    LinalgExt::LinalgExtFusionInContainingOpPattern pattern(this->getContext(),
                                                            containingOp);
    if (failed(functional::applyReturningPatternAt(pattern, producerOp)))
      return failure();
  }
  return success();
}

FailureOr<scf::ForOp> LinalgExt::RewriteLinalgExtTileToScfForOp::applyToOne(
    LinalgExt::TileOp target) {
  LinalgExt::TileOpToSCFRewriter pattern(this->getContext());
  auto functionalRewrite =
      [&](LinalgExt::TileOp op,
          PatternRewriter &rewriter) -> FailureOr<scf::ForOp> {
    auto result = pattern.returningMatchAndRewrite(op, rewriter);
    if (failed(result))
      return failure();
    return result;
  };
  return functional::applyAt(target, functionalRewrite);
}

FailureOr<LinalgExt::InParallelOp>
LinalgExt::RewriteLinalgExtTileToInParallelOp::applyToOne(
    LinalgExt::TileOp target) {
  LinalgExt::TileOpToInParallelRewriter pattern(this->getContext());
  auto functionalRewrite =
      [&](LinalgExt::TileOp op,
          PatternRewriter &rewriter) -> FailureOr<LinalgExt::InParallelOp> {
    auto result = pattern.returningMatchAndRewrite(op, rewriter);
    if (failed(result))
      return failure();
    return result;
  };
  return functional::applyAt(target, functionalRewrite);
}

FailureOr<Operation *>
LinalgExt::RewriteLinalgExtInParallelToAsyncOp::applyToOne(
    LinalgExt::InParallelOp target) {
  LinalgExt::InParallelOpToAsyncRewriter pattern(this->getContext());
  auto functionalRewrite =
      [&](LinalgExt::InParallelOp op,
          PatternRewriter &rewriter) -> FailureOr<Operation *> {
    auto result = pattern.returningMatchAndRewrite(op, rewriter);
    if (failed(result))
      return failure();
    return result;
  };
  return functional::applyAt(target, functionalRewrite);
}

LogicalResult LinalgExt::RewriteLinalgExtInParallelToHALOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  LinalgExt::InParallelOpToHALRewriter pattern(this->getContext());
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  return functional::applyReturningPatternAt(
      pattern, cast<LinalgExt::InParallelOp>(targets.front()));
}

FailureOr<scf::ForOp>
LinalgExt::RewriteLinalgExtInParallelToScfForOp::applyToOne(
    LinalgExt::InParallelOp target) {
  LinalgExt::InParallelOpToScfForRewriter pattern(this->getContext());
  auto functionalRewrite =
      [&](LinalgExt::InParallelOp op,
          PatternRewriter &rewriter) -> FailureOr<scf::ForOp> {
    auto result = pattern.returningMatchAndRewrite(op, rewriter);
    if (failed(result))
      return failure();
    return result;
  };
  return functional::applyAt(target, functionalRewrite);
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.cpp.inc"
