// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include <numeric>
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    operands.push_back(UnitAttr::get(op->getContext()));
  }

  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<VectorValue>(result)) {
      results.push_back(
          analysis.getLayout<VectorLayoutInterface>(vectorResult));
      continue;
    }
    results.push_back(UnitAttr::get(op->getContext()));
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
    assert(vectorLayout && "Malformed signature attribute.");
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
  if (llvm::none_of(values, llvm::IsaPred<VectorValue>))
    return false;

  // Check if all operands and results of this operation have a layout.
  return llvm::all_of(values, [&analysis](Value value) {
    auto vectorValue = dyn_cast<VectorValue>(value);
    return !vectorValue || analysis.getLayout<Attribute>(vectorValue);
  });
}

// When there exist a layout conflict, we'll try to write back to shared memory
// and read back to register with correct layout.
struct DecomposeLayoutConflictResolutions final
    : OpDistributionPattern<IREE::VectorExt::LayoutConflictResolutionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult
  matchAndRewrite(IREE::VectorExt::LayoutConflictResolutionOp resolutionOp,
                  DistributionSignature &signature,
                  PatternRewriter &rewriter) const override {
    auto loc = resolutionOp.getLoc();
    VectorValue vector = resolutionOp.getInput();
    VectorValue result = resolutionOp.getOutput();
    LayoutAttr currentLayout = dyn_cast<LayoutAttr>(signature[vector]);
    if (!currentLayout)
      return failure();
    LayoutAttr targetLayout = dyn_cast<LayoutAttr>(signature[result]);
    if (!targetLayout)
      return failure();

    SmallVector<int64_t> currentVecShape = currentLayout.getDistributedShape();
    SmallVector<int64_t> targetVecShape = targetLayout.getDistributedShape();
    if (currentVecShape.size() != targetVecShape.size())
      return failure();

    // If the conditions suffice, we can skip the trip to shared memory
    // and just use the default/more efficient layout conflict resolution
    // distribution.
    auto numElements = [](ArrayRef<int64_t> vector) {
      return std::accumulate(vector.begin(), vector.end(), 1,
                             std::multiplies<int64_t>());
    };
    if (numElements(currentVecShape) == numElements(targetVecShape) &&
        !currentLayout.hasLaneConflictWith(targetLayout))
      return failure();

    // Compute Warp and Subgroup Related information
    auto funcOp = resolutionOp->getParentOfType<func::FuncOp>();
    if (!funcOp)
      return failure();
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!workgroupSize.has_value() || !subgroupSize.has_value()) {
      return failure();
    }
    int64_t flatThreadSize = ShapedType::getNumElements(workgroupSize.value());
    if (flatThreadSize % subgroupSize.value() != 0)
      return failure();
    int64_t numSubgroups = flatThreadSize / subgroupSize.value();

    // Define shapes and types needed to be roundtripped to shared-memory.
    auto resolutionType =
        llvm::dyn_cast_or_null<VectorType>(resolutionOp.getResult().getType());
    if (!resolutionType)
      return failure();
    if (!resolutionType.hasStaticShape())
      return failure();
    auto paddedShape = SmallVector<int64_t>(resolutionType.getShape());
    int64_t vectorRank = resolutionType.getRank();
    paddedShape[vectorRank - 1] *= numSubgroups;

    // Modification for subgroup offseting. Stack subgroup data on the fastest
    // dimension. auto subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc);
    AffineExpr d0, d1, d2, s0;
    bindDims(rewriter.getContext(), d0, d1, d2);
    bindSymbols(rewriter.getContext(), s0);
    auto indexType = rewriter.getIndexType();
    Value threadX =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::x);
    Value threadY =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::y);
    Value threadZ =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpu::Dimension::z);
    Value flatThreadId = affine::makeComposedAffineApply(
        rewriter, loc,
        (d0 + workgroupSize.value()[0] * d1 +
         (workgroupSize.value()[0] * workgroupSize.value()[1]) * d2),
        {threadX, threadY, threadZ});
    Value subgroupOffset = affine::makeComposedAffineApply(
        rewriter, loc,
        s0.floorDiv(subgroupSize.value()) *
            resolutionType.getShape()[vectorRank - 1],
        {flatThreadId});

    // Create shared memory to store the intermediate from src layout.
    auto workgroupMemoryAddressSpace = Attribute(gpu::AddressSpaceAttr::get(
        rewriter.getContext(), gpu::AddressSpace::Workgroup));
    MemRefType allocType =
        MemRefType::get(paddedShape, resolutionType.getElementType(),
                        AffineMap(), workgroupMemoryAddressSpace);
    auto alloc = rewriter.create<memref::AllocOp>(loc, allocType);

    SmallVector<OpFoldResult> offsets(vectorRank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(vectorRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> shapes = llvm::to_vector(
        llvm::map_range(resolutionType.getShape(), [&](int64_t dim) {
          return OpFoldResult(rewriter.getIndexAttr(dim));
        }));
    offsets[vectorRank - 1] = subgroupOffset;
    auto subview = rewriter.create<memref::SubViewOp>(loc, alloc, offsets,
                                                      shapes, strides);

    // Creating write/trip to shared memory using src layout.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(resolutionType.getRank(), c0);
    SmallVector<bool> inBounds(vectorRank, true);
    auto write = rewriter.create<vector::TransferWriteOp>(loc, vector, subview,
                                                          indices, inBounds);
    // Insert gpu.barrier
    rewriter.create<gpu::BarrierOp>(write.getLoc());

    // Creating read from shared memory using dst layout.
    // Read with offset starting from the warpIdx * OG fastest dim.
    indices[vectorRank - 1] = subgroupOffset;
    auto read = rewriter.create<vector::TransferReadOp>(loc, resolutionType,
                                                        alloc, indices);

    // Set layouts to read and write.
    auto unitAttr = UnitAttr::get(rewriter.getContext());
    auto writeAttrs = SmallVector<Attribute>(write->getNumOperands(), unitAttr);
    writeAttrs[0] =
        currentLayout; // 1st operand is src which requires currentLayout.
    ArrayAttr writeOperandsAttr =
        ArrayAttr::get(rewriter.getContext(), writeAttrs);
    ArrayAttr writeResultsAttr = ArrayAttr::get(rewriter.getContext(), {});
    Attribute writeSignature[] = {writeOperandsAttr, writeResultsAttr};
    write->setAttr(kVectorLayoutFetcherStorageAttrName,
                   ArrayAttr::get(rewriter.getContext(), writeSignature));

    ArrayAttr readOperandsAttr = ArrayAttr::get(
        rewriter.getContext(),
        SmallVector<Attribute>(read->getNumOperands(), unitAttr));
    ArrayAttr readResultsAttr =
        ArrayAttr::get(rewriter.getContext(), {targetLayout});
    Attribute readSignature[] = {readOperandsAttr, readResultsAttr};
    read->setAttr(kVectorLayoutFetcherStorageAttrName,
                  ArrayAttr::get(rewriter.getContext(), readSignature));

    rewriter.replaceOp(resolutionOp, read.getResult());
    return success();
  }
};

LogicalResult distributeVectorOps(Operation *root,
                                  RewritePatternSet &distributionPatterns,
                                  VectorLayoutOptions &options) {
  // Run the analysis and determine the layouts.
  LLVM_DEBUG(llvm::dbgs() << "Running Layout Analysis\n");
  VectorLayoutAnalysis analysis(root);
  if (failed(options.setAnchorOps(analysis)))
    return failure();
  if (failed(analysis.run()))
    return failure();
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

  {
    RewritePatternSet preprocessPatterns(root->getContext());
    preprocessPatterns.add<DecomposeLayoutConflictResolutions>(
        preprocessPatterns.getContext());
    FrozenRewritePatternSet frozenPreprocessPatterns(
        std::move(preprocessPatterns));
    applyVectorDistribution(root, frozenPreprocessPatterns);

    LLVM_DEBUG(llvm::dbgs() << "After Decomposition of Layout Conflicts\n");
    LLVM_DEBUG(root->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n\n");
  }

  FrozenRewritePatternSet frozenPatterns(std::move(distributionPatterns));
  applyVectorDistribution(root, frozenPatterns);

  RewritePatternSet patterns(root->getContext());
  IREE::VectorExt::ToSIMDOp::getCanonicalizationPatterns(patterns,
                                                         root->getContext());
  IREE::VectorExt::ToSIMTOp::getCanonicalizationPatterns(patterns,
                                                         root->getContext());
  if (failed(applyPatternsAndFoldGreedily(root, std::move(patterns)))) {
    return failure();
  }

  if (options.verifyConversion()) {
    WalkResult hasConversionOp = root->walk([](Operation *op) {
      if (isa<IREE::VectorExt::ToSIMDOp, IREE::VectorExt::ToSIMTOp>(op)) {
        for (auto user : op->getUsers()) {
          if (!isa<IREE::VectorExt::ToSIMDOp, IREE::VectorExt::ToSIMTOp>(
                  user)) {
            LLVM_DEBUG({
              llvm::dbgs() << "Found live cast op: " << *op << "\n";
              llvm::dbgs() << "With live user: " << *user << "\n";
            });
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (hasConversionOp.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

} // namespace mlir::iree_compiler
