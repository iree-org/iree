// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"

namespace mlir::iree_compiler {

TypedAttr convertAttributeToStream(TypedAttr attr) {
  if (!attr)
    return {};
  if (auto parameterAttr = dyn_cast<IREE::Flow::NamedParameterAttr>(attr)) {
    return IREE::Stream::NamedParameterAttr::get(
        attr.getContext(), parameterAttr.getType(), parameterAttr.getScope(),
        parameterAttr.getKey(), parameterAttr.getConfig());
  }
  return attr;
}

IREE::Stream::AffinityAttr
tryLookupGlobalAffinity(Operation *op,
                        IREE::Stream::AffinityAnalysis *affinityAnalysis) {
  return affinityAnalysis->lookupGlobalAffinity(op);
}

IREE::Stream::AffinityAttr
tryLookupExecutionAffinity(Operation *op,
                           IREE::Stream::AffinityAnalysis *affinityAnalysis) {
  assert(llvm::isa<IREE::Stream::AffinityOpInterface>(op) &&
         "must be an affinity op");
  return affinityAnalysis->lookupExecutionAffinity(op);
}

IREE::Stream::AffinityAttr
tryLookupResultAffinity(Value value,
                        IREE::Stream::AffinityAnalysis *affinityAnalysis) {
  return affinityAnalysis->lookupResourceAffinity(value);
}

static std::pair<Value, Value>
resolveTensorOperand(Location loc, Value convertedOperand, OpBuilder &builder) {
  auto operandType = convertedOperand.getType();
  if (llvm::isa<IREE::Stream::ResourceType>(operandType)) {
    // Prior to https://reviews.llvm.org/D111620 this is the path we'd take;
    // the tensor operands would be remapped into their new resource types.
    // This is still possible during rewriting if we ourselves produce a new
    // resource type, but the automatic materialization will go down the
    // unrealized_conversion_cast path below.
    return std::make_pair(convertedOperand,
                          builder.createOrFold<IREE::Stream::ResourceSizeOp>(
                              loc, builder.getIndexType(), convertedOperand));
  } else if (auto castOp =
                 convertedOperand
                     .getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    // We only have a single tensor type conversion and it expands to (resource,
    // size) so that's all we look for here.
    assert(castOp.getNumOperands() == 2 && "expected (resource, size)");
    return std::make_pair(castOp.getOperand(0), castOp.getOperand(1));
  }
  assert(false &&
         "unexpected operand; expected either a IREE::Stream::ResourceType or "
         "the result of a mlir::UnrealizedConversionCastOp");
  return std::make_pair(Value{}, Value{});
}

void expandResourceOperand(Location loc, Value operand,
                           SmallVectorImpl<Value> &newOperands,
                           OpBuilder &builder) {
  if (llvm::isa<TensorType>(operand.getType())) {
    auto [resource, resourceSize] = resolveTensorOperand(loc, operand, builder);
    newOperands.push_back(resource);
    newOperands.push_back(resourceSize);
  } else if (llvm::isa<IREE::Stream::ResourceType>(operand.getType())) {
    newOperands.push_back(operand);
    newOperands.push_back(
        builder.createOrFold<IREE::Stream::ResourceSizeOp>(loc, operand));
  } else {
    newOperands.push_back(operand);
  }
}

SmallVector<Value> expandResourceOperands(Location loc, ValueRange operands,
                                          ConversionPatternRewriter &rewriter) {
  SmallVector<Value> expandedOperands;
  expandedOperands.reserve(operands.size());
  for (auto operand : operands) {
    expandResourceOperand(loc, operand, expandedOperands, rewriter);
  }
  return expandedOperands;
}

ConvertedTensor resolveTensorOperand(
    Location loc, Value originalOperand, Value convertedOperand,
    IREE::Stream::AffinityAnalysis *affinityAnalysis, OpBuilder &builder) {
  auto [resource, resourceSize] =
      resolveTensorOperand(loc, convertedOperand, builder);
  auto affinityAttr = affinityAnalysis->lookupResourceAffinity(originalOperand);
  return {affinityAttr, resource, resourceSize};
}

ConvertedTensor transferTensorOperand(
    Location loc, Value originalOperand, Value convertedOperand,
    IREE::Stream::AffinityAttr requiredAffinityAttr,
    IREE::Stream::AffinityAnalysis *affinityAnalysis, OpBuilder &builder) {
  auto [resource, resourceSize] =
      resolveTensorOperand(loc, convertedOperand, builder);
  auto affinityAttr = affinityAnalysis->lookupResourceAffinity(originalOperand);
  if (affinityAttr != requiredAffinityAttr) {
    resource = builder.create<IREE::Stream::AsyncTransferOp>(
        loc, resource.getType(), resource, resourceSize, resourceSize,
        affinityAttr, requiredAffinityAttr);
  }
  return {requiredAffinityAttr, resource, resourceSize};
}

} // namespace mlir::iree_compiler
