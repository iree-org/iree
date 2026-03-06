// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/DistributionPatterns.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VectorExt {

using VectorValue = TypedValue<VectorType>;

constexpr StringLiteral kVectorLayoutFetcherStorageAttrName =
    "__vector_layout_fetcher_storage";

constexpr StringLiteral kVectorLayoutRedistributeAttrName =
    "__vector_layout_redistribute";

/// Set signature for the operation based on the analysis. Returns failure if
/// an operation contains vectors that cannot be distributed i.e. they have no
/// layout.
LogicalResult
setOpSignature(Operation *op,
               const llvm::MapVector<Value, VectorLayoutInterface> &layouts,
               const VectorLayoutOptions &options) {
  SmallVector<Attribute> operands;
  SmallVector<Attribute> results;

  for (Value operand : op->getOperands()) {
    if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
      if (auto layout = layouts.lookup(vectorOperand)) {
        operands.push_back(layout);
        continue;
      }
      if (auto layout = options.getDefaultLayout(vectorOperand.getType())) {
        operands.push_back(layout);
        continue;
      }
      return failure();
    }
    operands.push_back(UnitAttr::get(op->getContext()));
  }

  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<VectorValue>(result)) {
      if (auto layout = layouts.lookup(vectorResult)) {
        results.push_back(layout);
        continue;
      }
      if (auto layout = options.getDefaultLayout(vectorResult.getType())) {
        results.push_back(layout);
        continue;
      }
      return failure();
    }
    results.push_back(UnitAttr::get(op->getContext()));
  }

  ArrayAttr operandsAttr = ArrayAttr::get(op->getContext(), operands);
  ArrayAttr resultsAttr = ArrayAttr::get(op->getContext(), results);
  Attribute signature[] = {operandsAttr, resultsAttr};
  op->setAttr(kVectorLayoutFetcherStorageAttrName,
              ArrayAttr::get(op->getContext(), signature));
  return success();
}

bool hasOpSignature(Operation *op) {
  return op->hasAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
}

void removeOpSignature(Operation *op) {
  op->removeDiscardableAttr(kVectorLayoutFetcherStorageAttrName);
}

bool isMarkedForRedistribution(Operation *op) {
  return op->hasAttr(kVectorLayoutRedistributeAttrName) && hasOpSignature(op);
}

void clearRedistributionMark(Operation *op) {
  op->removeAttr(kVectorLayoutRedistributeAttrName);
}

DistributionSignature getOpSignature(Operation *op) {
  ArrayAttr signatureAttr =
      op->getAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
  assert(signatureAttr && "Op should have a signature attribute.");
  assert(signatureAttr.size() == 2 && "Malformed signature attribute.");

  ArrayAttr operandsAttr = dyn_cast<ArrayAttr>(signatureAttr[0]);
  ArrayAttr resultsAttr = dyn_cast<ArrayAttr>(signatureAttr[1]);
  assert(operandsAttr && resultsAttr && "Malformed signature attribute.");
  assert(operandsAttr.size() == op->getNumOperands() &&
         "Malformed signature attribute.");
  assert(resultsAttr.size() == op->getNumResults() &&
         "Malformed signature attribute.");

  DistributionSignature signature;

  auto addLayoutToSignature([&](Value value, Attribute layout) {
    // Ignore null attributes.
    if (isa<UnitAttr>(layout)) {
      assert(!isa<VectorValue>(value) &&
             "Malformed signature attribute: unit attribute for vector value.");
      return;
    }

    assert(isa<VectorValue>(value) &&
           "Malformed signature attribute: non-unit attribute for non-vector "
           "value.");
    auto vector = cast<VectorValue>(value);

    auto vectorLayout = cast<VectorLayoutInterface>(layout);
    signature[vector] = vectorLayout;
  });

  for (auto [value, layout] :
       llvm::zip_equal(op->getOperands(), operandsAttr)) {
    addLayoutToSignature(value, layout);
  }
  for (auto [value, layout] : llvm::zip_equal(op->getResults(), resultsAttr)) {
    addLayoutToSignature(value, layout);
  }

  return signature;
}

VectorValue
DistributionPattern::getDistributed(RewriterBase &rewriter, VectorValue value,
                                    VectorLayoutInterface layout) const {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    return cast<VectorValue>(toSIMD.getInput());
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = layout.getDistributedShape();
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = IREE::VectorExt::ToSIMTOp::create(rewriter, value.getLoc(),
                                                  distributedType, value);
  return toSIMT.getResult();
}

SmallVector<Value> DistributionPattern::getOpDistributedReplacements(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  SmallVector<Value> replacements;
  for (auto [opResult, replacement] :
       llvm::zip_equal(op->getOpResults(), values)) {
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<VectorValue>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      Value toSIMD = IREE::VectorExt::ToSIMDOp::create(
          rewriter, oldResult.getLoc(), oldResult.getType(), replacement);
      // Add to replacements.
      replacement = toSIMD;
    }
    replacements.push_back(replacement);
  }
  return replacements;
}

void DistributionPattern::replaceOpWithDistributedValues(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements =
      getOpDistributedReplacements(rewriter, op, values);
  rewriter.replaceOp(op, replacements);
}

std::optional<DistributionSignature>
DistributionPattern::getOpSignature(Operation *op) const {
  if (!IREE::VectorExt::hasOpSignature(op)) {
    return std::nullopt;
  }
  return IREE::VectorExt::getOpSignature(op);
}

void DistributionPattern::setSignatureForRedistribution(
    RewriterBase &rewriter, Operation *op,
    ArrayRef<VectorLayoutInterface> inputLayouts,
    ArrayRef<VectorLayoutInterface> outputLayouts) const {
  auto unitAttr = UnitAttr::get(rewriter.getContext());
  auto inputAttrs = SmallVector<Attribute>(op->getNumOperands(), unitAttr);
  auto outputAttrs = SmallVector<Attribute>(op->getNumResults(), unitAttr);

  auto isVectorType = [](Value x) { return isa<VectorType>(x.getType()); };
  assert(llvm::count_if(op->getOperands(), isVectorType) ==
         inputLayouts.size());
  int64_t currVectorInput = 0;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (isVectorType(operand)) {
      inputAttrs[idx] = inputLayouts[currVectorInput];
      ++currVectorInput;
    }
  }

  assert(llvm::count_if(op->getResults(), isVectorType) ==
         outputLayouts.size());
  int64_t currVectorOutput = 0;
  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (isVectorType(result)) {
      outputAttrs[idx] = outputLayouts[currVectorOutput];
      ++currVectorOutput;
    }
  }

  auto inputArrayAttr = ArrayAttr::get(rewriter.getContext(), inputAttrs);
  auto outputArrayAttr = ArrayAttr::get(rewriter.getContext(), outputAttrs);

  Attribute signature[] = {inputArrayAttr, outputArrayAttr};
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr(kVectorLayoutFetcherStorageAttrName,
                ArrayAttr::get(rewriter.getContext(), signature));
    op->setAttr(kVectorLayoutRedistributeAttrName, unitAttr);
  });
}

LogicalResult
DistributionPattern::replaceParentMask(PatternRewriter &rewriter,
                                       vector::MaskOp maskOp) const {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(maskOp);
  std::optional<DistributionSignature> signatureMask = getOpSignature(maskOp);
  if (!signatureMask.has_value()) {
    return rewriter.notifyMatchFailure(maskOp, "mask should have a signature.");
  }
  SmallVector<Value> returns = maskOp.getBody()->getTerminator()->getOperands();
  for (auto [idx, ret] : llvm::enumerate(returns)) {
    if (VectorValue vectorRet = dyn_cast<VectorValue>(ret)) {
      VectorValue maskRet = cast<VectorValue>(maskOp.getResult(idx));
      VectorLayoutInterface layout =
          dyn_cast<NestedLayoutAttr>(signatureMask.value()[maskRet]);
      if (!layout) {
        return rewriter.notifyMatchFailure(maskOp,
                                           "layout must be NestedLayoutAttr");
      }
      ret = getDistributed(rewriter, vectorRet, layout);
    }
  }
  rewriter.eraseOp(maskOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(maskOp.getBody(), maskOp);
  replaceOpWithDistributedValues(rewriter, maskOp, returns);
  return success();
}

} // namespace mlir::iree_compiler::IREE::VectorExt
