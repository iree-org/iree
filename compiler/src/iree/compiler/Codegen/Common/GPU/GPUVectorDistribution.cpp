// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

using VectorValue = TypedValue<VectorType>;

constexpr StringLiteral kVectorLayoutFetcherStorageAttrName =
    "__vector_layout_fetcher_storage";

static void setOpSignature(Operation *op, VectorLayoutAnalysis &analysis) {
  SmallVector<Attribute> operands;
  SmallVector<Attribute> results;

  for (Value operand : op->getOperands()) {
    if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
      operands.push_back(
          analysis.getLayout<VectorLayoutInterface>(vectorOperand));
      continue;
    }
    operands.push_back(VectorLayoutInterface());
  }

  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<VectorValue>(result)) {
      results.push_back(
          analysis.getLayout<VectorLayoutInterface>(vectorResult));
      continue;
    }
    results.push_back(VectorLayoutInterface());
  }

  ArrayAttr operandsAttr = ArrayAttr::get(op->getContext(), operands);
  ArrayAttr resultsAttr = ArrayAttr::get(op->getContext(), results);
  Attribute signature[] = {operandsAttr, resultsAttr};
  op->setAttr(kVectorLayoutFetcherStorageAttrName,
              ArrayAttr::get(op->getContext(), signature));
}

static bool hasOpSignature(Operation *op) {
  return op->hasAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
}

static DistributionSignature getOpSignature(Operation *op) {
  ArrayAttr signatureAttr =
      op->getAttrOfType<ArrayAttr>(kVectorLayoutFetcherStorageAttrName);
  assert(signatureAttr && "Op should have a signature attribute.");
  assert(signatureAttr.size() == 2 && "Malformed signature attribute.");

  ArrayAttr operandsAttr = dyn_cast<ArrayAttr>(signatureAttr[0]);
  ArrayAttr resultsAttr = dyn_cast<ArrayAttr>(signatureAttr[1]);
  assert(operandsAttr && resultsAttr && "Malformed signature attribute.");

  DistributionSignature signature;
  for (Attribute operandAttr : operandsAttr) {
    // Ignore null attributes.
    if (!operandAttr) {
      signature.operands.push_back(VectorLayoutInterface());
      continue;
    }

    auto operandLayout = cast<VectorLayoutInterface>(operandAttr);
    assert(operandLayout && "Malformed signature attribute.");
    signature.operands.push_back(operandLayout);
  }

  for (Attribute resultAttr : resultsAttr) {
    // Ignore null attributes.
    if (!resultAttr) {
      signature.results.push_back(VectorLayoutInterface());
      continue;
    }

    VectorLayoutInterface resultLayout =
        cast<VectorLayoutInterface>(resultAttr);
    assert(resultLayout && "Malformed signature attribute.");
    signature.results.push_back(resultLayout);
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
  SmallVector<int64_t> distributedShape =
      layout.getDistributedShape(value.getType());
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void DistributionPattern::replaceOpWithDistributedValues(
    RewriterBase &rewriter, Operation *op, ValueRange values) const {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (auto [opResult, replacement] :
       llvm::zip_equal(op->getOpResults(), values)) {
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<VectorValue>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      Value toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Add to replacements.
      replacement = toSIMD;
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

std::optional<DistributionSignature>
DistributionPattern::getOpSignature(Operation *op) const {
  if (!hasOpSignature(op)) {
    return std::nullopt;
  }
  return ::mlir::iree_compiler::getOpSignature(op);
}

static void
debugPrintUniqueOperationNames(SmallVectorImpl<Operation *> &worklist) {
  DenseSet<StringRef> uniqueNames;
  for (Operation *op : worklist) {
    uniqueNames.insert(op->getName().getStringRef());
  }

  for (StringRef name : uniqueNames) {
    llvm::dbgs().indent(2) << "* " << name << "\n";
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

/// A rewriter for the pattern rewriting driver.
struct VectorDistributionRewriter : PatternRewriter {
  VectorDistributionRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

static void applyVectorDistribution(Operation *root,
                                    const FrozenRewritePatternSet &patterns) {

  SmallVector<Operation *> worklist;

  VectorDistributionRewriter rewriter(root->getContext());
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // Collect all the operations to be distributed.
  LLVM_DEBUG(llvm::dbgs() << "Collecting operations to be distributed\n");
  root->walk([&](Operation *op) {
    if (hasOpSignature(op)) {
      worklist.push_back(op);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Operations to be distributed:\n");
  LLVM_DEBUG(debugPrintUniqueOperationNames(worklist));

  // Note that the pattern application here never runs on a newly created
  // operation. It always runs on an existing operation. This ensures that no
  // invalidated state of the analysis is ever used.
  for (Operation *op : worklist) {
    LLVM_DEBUG(llvm::dbgs() << "Distributing: ");
    LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    if (failed(applicator.matchAndRewrite(op, rewriter))) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << ": Failed to distribute operation:\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs().indent(2)
               << ": Successfully distributed operation:\n");
  }
}

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  auto values = llvm::to_vector_of<Value>(op->getOperands());
  llvm::append_range(values, op->getResults());

  // First check if any of them are vector values.
  if (llvm::none_of(values, [](Value value) -> bool {
        return isa<VectorValue>(value);
      })) {
    return false;
  }

  // Check if all operands and results of this operation have a layout.
  return llvm::all_of(values, [&](Value value) -> bool {
    auto vectorValue = dyn_cast<VectorValue>(value);
    return !vectorValue || analysis.getLayout<Attribute>(vectorValue);
  });
}

void distributeVectorOps(Operation *root,
                         RewritePatternSet &distributionPatterns,
                         VectorLayoutOptions &options) {
  // Run the analysis and determine the layouts.
  LLVM_DEBUG(llvm::dbgs() << "Running Layout Analysis\n");
  VectorLayoutAnalysis analysis(root);
  options.setAnchorOps(analysis);
  if (failed(analysis.run()))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Succeded\n");
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  // Go to each operation, and set its distribution signature.
  LLVM_DEBUG(
      llvm::dbgs() << "Setting distribution signatures for operations\n");
  root->walk([&](Operation *op) {
    if (canDistribute(op, analysis)) {
      setOpSignature(op, analysis);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Distribution signatures set\n");
  LLVM_DEBUG(root->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n\n");

  FrozenRewritePatternSet frozenPatterns(std::move(distributionPatterns));
  return applyVectorDistribution(root, frozenPatterns);
}

} // namespace mlir::iree_compiler
