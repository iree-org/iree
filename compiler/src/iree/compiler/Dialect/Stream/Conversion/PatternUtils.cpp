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
resolveTensorOperands(Location loc, ValueRange convertedOperand,
                      OpBuilder &builder) {
  if (convertedOperand.size() == 2) {
    return {convertedOperand[0], convertedOperand[1]};
  }

  auto operandType = convertedOperand.front().getType();
  if (llvm::isa<IREE::Stream::ResourceType>(operandType)) {
    // Prior to https://reviews.llvm.org/D111620 this is the path we'd take;
    // the tensor operands would be remapped into their new resource types.
    // This is still possible during rewriting if we ourselves produce a new
    // resource type, but the automatic materialization will go down the
    // unrealized_conversion_cast path below.
    return std::make_pair(
        convertedOperand.front(),
        builder.createOrFold<IREE::Stream::ResourceSizeOp>(
            loc, builder.getIndexType(), convertedOperand.front()));
  }
  assert(0 &&
         "unexpected operand; expected either a IREE::Stream::ResourceType or "
         "the result of a mlir::UnrealizedConversionCastOp");
  return std::make_pair(Value{}, Value{});
}

ConvertedTensor resolveTensorOperands(
    Location loc, Value originalOperand, ValueRange convertedOperand,
    IREE::Stream::AffinityAnalysis *affinityAnalysis, OpBuilder &builder) {
  auto [resource, resourceSize] =
      resolveTensorOperands(loc, convertedOperand, builder);
  auto affinityAttr = affinityAnalysis->lookupResourceAffinity(originalOperand);
  return {affinityAttr, resource, resourceSize};
}

ConvertedTensor transferTensorOperands(
    Location loc, Value originalOperand, ValueRange convertedOperand,
    IREE::Stream::AffinityAttr requiredAffinityAttr,
    IREE::Stream::AffinityAnalysis *affinityAnalysis, OpBuilder &builder) {
  auto [resource, resourceSize] =
      resolveTensorOperands(loc, convertedOperand, builder);
  auto affinityAttr = affinityAnalysis->lookupResourceAffinity(originalOperand);
  if (affinityAttr != requiredAffinityAttr) {
    resource = builder.create<IREE::Stream::AsyncTransferOp>(
        loc, resource.getType(), resource, resourceSize, resourceSize,
        affinityAttr, requiredAffinityAttr);
  }
  return {requiredAffinityAttr, resource, resourceSize};
}

void replaceOpWithMultiple(Operation *op,
                           ArrayRef<SmallVector<Value>> replacements,
                           ConversionPatternRewriter &rewriter) {
  auto r = llvm::map_to_vector(
      replacements, [](ArrayRef<Value> v) -> ValueRange { return v; });
  rewriter.replaceOpWithMultiple(op, r);
}

void replaceOpWithMultiple(Operation *op, ValueRange resources,
                           ValueRange sizes,
                           ConversionPatternRewriter &rewriter) {
  SmallVector<SmallVector<Value>> replacements = llvm::map_to_vector(
      llvm::zip_equal(resources, sizes), [](auto it) -> SmallVector<Value> {
        if (std::get<1>(it)) {
          return {std::get<0>(it), std::get<1>(it)};
        }
        return {std::get<0>(it)};
      });
  replaceOpWithMultiple(op, replacements, rewriter);
}

} // namespace mlir::iree_compiler
