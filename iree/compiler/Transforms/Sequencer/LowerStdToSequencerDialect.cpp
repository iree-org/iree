// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/IR/Dialect.h"
#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/Sequencer/HLDialect.h"
#include "iree/compiler/IR/Sequencer/HLOps.h"
#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/Utils/MemRefUtils.h"
#include "iree/compiler/Utils/OpCreationUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

template <typename T>
class SequencerConversionPattern : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
};

struct CallOpLowering : public SequencerConversionPattern<CallOp> {
  using SequencerConversionPattern::SequencerConversionPattern;

  PatternMatchResult matchAndRewrite(
      CallOp callOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 4> resultTypes(callOp.getResultTypes());
    rewriter.replaceOpWithNewOp<IREESeq::HL::CallOp>(callOp, callOp.getCallee(),
                                                     resultTypes, operands);

    return matchSuccess();
  }
};

struct CallIndirectOpLowering
    : public SequencerConversionPattern<CallIndirectOp> {
  using SequencerConversionPattern::SequencerConversionPattern;

  PatternMatchResult matchAndRewrite(
      CallIndirectOp callOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::HL::CallIndirectOp>(
        callOp, callOp.getCallee(), operands);
    return matchSuccess();
  }
};

struct ReturnOpLowering : public SequencerConversionPattern<ReturnOp> {
  using SequencerConversionPattern::SequencerConversionPattern;

  PatternMatchResult matchAndRewrite(
      ReturnOp returnOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value *, 4> newOperands;
    newOperands.reserve(operands.size());
    for (auto *operand : operands) {
      newOperands.push_back(wrapAsMemRef(operand, returnOp, rewriter));
    }
    rewriter.replaceOpWithNewOp<IREESeq::HL::ReturnOp>(returnOp, newOperands);
    return matchSuccess();
  }
};

struct BranchOpLowering : public SequencerConversionPattern<BranchOp> {
  using SequencerConversionPattern::SequencerConversionPattern;
  PatternMatchResult matchAndRewrite(
      BranchOp branchOp, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::HL::BranchOp>(
        branchOp, destinations[0], operands[0]);
    return this->matchSuccess();
  }
};

struct CondBranchOpLowering : public SequencerConversionPattern<CondBranchOp> {
  using SequencerConversionPattern::SequencerConversionPattern;
  PatternMatchResult matchAndRewrite(
      CondBranchOp condBranchOp, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *condValue =
        loadAccessValue(condBranchOp.getLoc(), properOperands[0], rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::HL::CondBranchOp>(
        condBranchOp, condValue,
        destinations[IREESeq::HL::CondBranchOp::trueIndex],
        operands[IREESeq::HL::CondBranchOp::trueIndex],
        destinations[IREESeq::HL::CondBranchOp::falseIndex],
        operands[IREESeq::HL::CondBranchOp::falseIndex]);
    return this->matchSuccess();
  }
};

struct AllocOpLowering : public SequencerConversionPattern<AllocOp> {
  using SequencerConversionPattern::SequencerConversionPattern;
  PatternMatchResult matchAndRewrite(
      AllocOp allocOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): replace with length computation.
    rewriter.replaceOpWithNewOp<IREESeq::HL::AllocHeapOp>(
        allocOp, allocOp.getType(), operands);
    return matchSuccess();
  }
};

struct DeallocOpLowering : public SequencerConversionPattern<DeallocOp> {
  using SequencerConversionPattern::SequencerConversionPattern;
  PatternMatchResult matchAndRewrite(
      DeallocOp deallocOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::HL::DiscardOp>(deallocOp, operands[0]);
    return matchSuccess();
  }
};

struct LoadOpLowering : public SequencerConversionPattern<LoadOp> {
  using SequencerConversionPattern::SequencerConversionPattern;
  PatternMatchResult matchAndRewrite(
      LoadOp loadOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (loadOp.getMemRefType().getRank() != 0) {
      loadOp.emitError() << "Cannot lower load of non-scalar";
      return matchFailure();
    }
    ArrayRef<Value *> dimPieces;
    auto dst = rewriter.create<AllocOp>(loadOp.getLoc(), loadOp.getMemRefType(),
                                        dimPieces);
    auto emptyArrayMemref = createArrayConstant(rewriter, loadOp.getLoc(), {});
    rewriter.create<IREESeq::HL::CopyOp>(loadOp.getLoc(), loadOp.getMemRef(),
                                         /*srcIndices=*/emptyArrayMemref, dst,
                                         /*dstIndices=*/emptyArrayMemref,
                                         /*lengths=*/emptyArrayMemref);

    rewriter.replaceOpWithNewOp<IREE::MemRefToScalarOp>(loadOp, dst);

    return matchSuccess();
  }
};

struct StoreOpLowering : public SequencerConversionPattern<StoreOp> {
  using SequencerConversionPattern::SequencerConversionPattern;
  PatternMatchResult matchAndRewrite(
      StoreOp storeOp, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (storeOp.getMemRefType().getRank() != 0) {
      storeOp.emitError() << "Cannot lower store of non-scalar";
      return matchFailure();
    }

    auto src = rewriter.create<IREE::ScalarToMemRefOp>(
        storeOp.getLoc(), storeOp.getValueToStore());

    auto emptyArrayMemref = createArrayConstant(rewriter, storeOp.getLoc(), {});
    rewriter.replaceOpWithNewOp<IREESeq::HL::CopyOp>(
        storeOp, src, /*srcIndices=*/emptyArrayMemref, storeOp.getMemRef(),
        /*dstIndices=*/emptyArrayMemref, /*lengths=*/emptyArrayMemref);

    return matchSuccess();
  }
};

}  // namespace

void populateLowerStdToSequencerPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context) {
  patterns.insert<
      // Control flow.
      CallOpLowering, CallIndirectOpLowering, ReturnOpLowering,
      BranchOpLowering, CondBranchOpLowering,
      // Memory management.
      AllocOpLowering, DeallocOpLowering, LoadOpLowering, StoreOpLowering>(
      context);
}

}  // namespace iree_compiler
}  // namespace mlir
