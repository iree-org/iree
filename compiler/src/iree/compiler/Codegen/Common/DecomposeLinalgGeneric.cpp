// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct DecomposeParallelGeneric
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  struct BodyValueInfo {
    // The generic operand that this body value maps to.
    Value outerValue;
    // The indexing map associated with this body value.
    AffineMap indexingMap;
  };
  struct OpInfo {
    linalg::LinalgOp op;

    // The iteration domain, either as statically known values or dynamically
    // known symbols. This will equal the number of loops and the number of
    // dimensions in each indexing map.
    SmallVector<OpFoldResult> domain;

    // All dimensions of the domain, using ShapedType::kDynamicDim to signify
    // a dynamic dimension.
    SmallVector<int64_t> domainTypeDims;

    // Values for the subset of dynamic dims in domainTypeDims;
    SmallVector<Value> domainTypeDynamicDims;

    // Map of body SSA values to indexing map/outer value. If the body value
    // is not associated with a generic operand (i.e. it is captured), it will
    // not be in this map.
    llvm::DenseMap<Value, BodyValueInfo> bodyValueInfo;

    // Iterator types as a vector for easier construction of ops.
    SmallVector<StringRef> iteratorTypes;

    OpInfo(linalg::LinalgOp &op) : op(op) {
      for (Attribute attr : op.getIteratorTypes()) {
        iteratorTypes.push_back(attr.cast<StringAttr>().getValue());
      }
    }

    Value createDomainInitTensor(Type elementType, OpBuilder &builder) {
      return builder.create<linalg::InitTensorOp>(
          op.getLoc(), domainTypeDynamicDims, domainTypeDims, elementType);
    }
  };

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa<linalg::GenericOp>(op)) {
      return failure();
    }
    // Don't match unless if multiple ops to decompose.
    if (op.getBlock()->getOperations().size() <= 2) {
      return rewriter.notifyMatchFailure(op, "<= 2 body ops");
    }
    // Only operate on tensor ops that return results.
    if (!op.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(op, "not tensor based");
    }
    // Only operate on parallel generics.
    if (op.getNumParallelLoops() != op.getNumLoops()) {
      return rewriter.notifyMatchFailure(op, "not parallel");
    }

    OpInfo info{op};

    // While we unroll operations, we populate an intermediate that covers
    // the entire iteration domain. Here we compute that domain as a combination
    // of static and dynamic values.
    info.domainTypeDims = op.getStaticLoopRanges();
    for (auto it : llvm::zip(info.domainTypeDims,
                             op.createLoopRanges(rewriter, op.getLoc()))) {
      auto staticSize = std::get<0>(it);
      auto dynamicSize = std::get<1>(it).size;
      if (staticSize == ShapedType::kDynamicSize) {
        // Add an SSA value for dynamic.
        info.domain.push_back(dynamicSize);
        info.domainTypeDynamicDims.push_back(dynamicSize);
      } else {
        // Add an attribute for static.
        info.domain.push_back(rewriter.getIndexAttr(staticSize));
      }
    }

    // Populate the bodyValueMap with initial values.
    for (OpOperand *operand : op.getInputAndOutputOperands()) {
      Value blockValue = op.getTiedBlockArgument(operand);
      AffineMap indexingMap = op.getTiedIndexingMap(operand);
      info.bodyValueInfo.insert(std::make_pair(
          blockValue, BodyValueInfo{operand->get(), indexingMap}));
    }
    assert(info.bodyValueInfo.size() == op.getNumInputsAndOutputs() &&
           "mismatched ins/outs to block argument mapping");

    Operation *createdGenericOp = nullptr;
    for (Operation &childOp : op.getBlock()->getOperations()) {
      createdGenericOp = emitIntermediateGeneric(childOp, info, rewriter);
    }

    // The last generic replaces the match.
    rewriter.replaceOp(op, createdGenericOp->getResults());
    return success();
  }

  Operation *emitIntermediateGeneric(Operation &childOp, OpInfo &info,
                                     PatternRewriter &rewriter) const {
    // Map operands that are inputs to the overall op.
    SmallVector<Value> newIns;
    SmallVector<AffineMap> newIndexingMaps;
    SmallVector<std::pair<Value, unsigned>> operandToInsMap;
    for (Value operand : childOp.getOperands()) {
      auto foundIt = info.bodyValueInfo.find(operand);
      if (foundIt != info.bodyValueInfo.end()) {
        operandToInsMap.push_back({operand, newIns.size()});
        newIns.push_back(foundIt->second.outerValue);
        newIndexingMaps.push_back(foundIt->second.indexingMap);
      }
    }

    // Prepare outs / result types.
    bool isYield = false;
    SmallVector<Type> resultTypes;
    SmallVector<Value> newOuts;
    SmallVector<AffineMap> newResultIndexingMaps;
    if (llvm::isa<linalg::YieldOp>(childOp)) {
      // Final yield generic sets up results from the original op.
      isYield = true;
      for (Type t : info.op->getResultTypes()) {
        resultTypes.push_back(t);
      }
      for (OpOperand *out : info.op.getOutputOperands()) {
        newOuts.push_back(out->get());
        AffineMap indexingMap = info.op.getTiedIndexingMap(out);
        newIndexingMaps.push_back(indexingMap);
        newResultIndexingMaps.push_back(indexingMap);
      }
    } else {
      // Materialize scalar op results as a temporary.
      for (auto elementType : childOp.getResultTypes()) {
        Value initValue = info.createDomainInitTensor(elementType, rewriter);
        resultTypes.push_back(initValue.getType());
        newOuts.push_back(initValue);
        AffineMap indexingMap =
            rewriter.getMultiDimIdentityMap(info.domain.size());
        newIndexingMaps.push_back(indexingMap);
        newResultIndexingMaps.push_back(indexingMap);
      }
    }

    // Create new linalg op.
    auto newLinalgOp = rewriter.create<linalg::GenericOp>(
        info.op.getLoc(), resultTypes, newIns, newOuts, newIndexingMaps,
        info.iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          BlockAndValueMapping mapper;
          for (std::pair<Value, unsigned> operandToIns : operandToInsMap) {
            mapper.map(operandToIns.first, blockArgs[operandToIns.second]);
          }
          Operation *newChild = rewriter.insert(childOp.clone(mapper));
          if (!isYield) {
            nestedBuilder.create<linalg::YieldOp>(info.op.getLoc(),
                                                  newChild->getResults());
          }
        });

    // Associate the original child results with the newly created outer
    // values. Does nothing for a yield (no results).
    for (unsigned i = 0; i < childOp.getNumResults(); ++i) {
      info.bodyValueInfo.insert(std::make_pair(
          childOp.getResult(i),
          BodyValueInfo{newLinalgOp->getResult(i), newResultIndexingMaps[i]}));
    }

    return newLinalgOp;
  }
};

class DecomposeLinalgGenericPass
    : public DecomposeLinalgGenericBase<DecomposeLinalgGenericPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<DecomposeParallelGeneric>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createDecomposeLinalgGenericPass() {
  return std::make_unique<DecomposeLinalgGenericPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
