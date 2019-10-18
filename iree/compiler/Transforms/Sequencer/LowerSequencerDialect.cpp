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
#include "iree/compiler/IR/Sequencer/LLDialect.h"
#include "iree/compiler/IR/Sequencer/LLOps.h"
#include "iree/compiler/IR/StructureOps.h"
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Returns an integer scalar memref containing the offset specified by |indices|
// within |type|.
Value *computeOffset(Location loc, Value *reference, Value *indices,
                     OpBuilder &builder) {
  auto referenceType = reference->getType().cast<ShapedType>();
  auto *shapeMemRef = builder
                          .create<IREESeq::LL::AllocHeapOp>(
                              loc,
                              MemRefType::get({referenceType.getRank()},
                                              builder.getIntegerType(32)),
                              ArrayRef<Value *>{})
                          .getResult();
  builder.create<IREESeq::LL::ShapeOp>(loc, reference, shapeMemRef);
  auto *resultMemRef =
      builder
          .create<IREESeq::LL::AllocHeapOp>(
              loc, MemRefType::get({}, builder.getIntegerType(32)),
              ArrayRef<Value *>{})
          .getResult();
  auto elementSizeAttr = builder.getIntegerAttr(
      builder.getIntegerType(8), referenceType.getElementTypeBitWidth() / 8);
  builder.create<IREESeq::LL::ComputeOffsetOp>(
      loc, shapeMemRef, elementSizeAttr, indices, resultMemRef);
  return resultMemRef;
}

// Returns a tuple of (offset, length) integer scalar memrefs with the range
// specified by |indices| and |lengths| within |type|.
std::pair<Value *, Value *> computeRange(Location loc, Value *reference,
                                         Value *indices, Value *lengths,
                                         OpBuilder &builder) {
  auto referenceType = reference->getType().cast<ShapedType>();
  auto *shapeMemRef = builder
                          .create<IREESeq::LL::AllocHeapOp>(
                              loc,
                              MemRefType::get({referenceType.getRank()},
                                              builder.getIntegerType(32)),
                              ArrayRef<Value *>{})
                          .getResult();
  builder.create<IREESeq::LL::ShapeOp>(loc, reference, shapeMemRef);
  auto *offsetMemRef =
      builder
          .create<IREESeq::LL::AllocHeapOp>(
              loc, MemRefType::get({}, builder.getIntegerType(32)),
              ArrayRef<Value *>{})
          .getResult();
  auto *lengthMemRef =
      builder
          .create<IREESeq::LL::AllocHeapOp>(
              loc, MemRefType::get({}, builder.getIntegerType(32)),
              ArrayRef<Value *>{})
          .getResult();
  auto elementSizeAttr = builder.getIntegerAttr(
      builder.getIntegerType(8), referenceType.getElementTypeBitWidth() / 8);
  builder.create<IREESeq::LL::ComputeRangeOp>(loc, shapeMemRef, elementSizeAttr,
                                              indices, lengths, offsetMemRef,
                                              lengthMemRef);
  return {offsetMemRef, lengthMemRef};
}

struct LowerSliceOpPattern : public OpRewritePattern<IREESeq::HL::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREESeq::HL::SliceOp op,
                                     PatternRewriter &rewriter) const {
    auto range = computeRange(op.getLoc(), op.src(), op.indices(), op.lengths(),
                              rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::LL::DynamicSliceOp>(
        op, ArrayRef<Type>{op.getResult()->getType()},
        ArrayRef<Value *>{op.src(), range.first, range.second}, op.getAttrs());
    return matchSuccess();
  }
};

struct LowerShapeOpPattern : public OpRewritePattern<IREESeq::HL::ShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREESeq::HL::ShapeOp op,
                                     PatternRewriter &rewriter) const {
    auto *shapeMemRef =
        rewriter
            .create<IREESeq::LL::AllocHeapOp>(
                op.getLoc(),
                MemRefType::get(
                    {op.getResult()->getType().cast<ShapedType>().getRank()},
                    rewriter.getIntegerType(64)),
                ArrayRef<Value *>{})
            .getResult();
    op.replaceAllUsesWith(shapeMemRef);
    rewriter.replaceOpWithNewOp<IREESeq::LL::ShapeOp>(op, op.getOperand(),
                                                      shapeMemRef);
    return matchSuccess();
  }
};

struct LowerCopyOpPattern : public OpRewritePattern<IREESeq::HL::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREESeq::HL::CopyOp op,
                                     PatternRewriter &rewriter) const {
    auto *srcOffsetMemRef =
        computeOffset(op.getLoc(), op.src(), op.srcIndices(), rewriter);
    auto dstRange = computeRange(op.getLoc(), op.dst(), op.dstIndices(),
                                 op.lengths(), rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::LL::DynamicCopyOp>(
        op, op.src(), srcOffsetMemRef, op.dst(), dstRange.first,
        dstRange.second);
    return matchSuccess();
  }
};

struct LowerFillOpPattern : public OpRewritePattern<IREESeq::HL::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREESeq::HL::FillOp op,
                                     PatternRewriter &rewriter) const {
    auto dstRange = computeRange(op.getLoc(), op.dst(), op.dstIndices(),
                                 op.lengths(), rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::LL::DynamicFillOp>(
        op, op.value(), op.dst(), dstRange.first, dstRange.second);
    return matchSuccess();
  }
};

struct LowerBranchOpPattern : public OpRewritePattern<IREESeq::HL::BranchOp> {
  using OpRewritePattern<IREESeq::HL::BranchOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREESeq::HL::BranchOp op,
                                     PatternRewriter &rewriter) const {
    SmallVector<Value *, 8> operands{op.getOperation()->getOperands()};

    rewriter.replaceOpWithNewOp<IREESeq::LL::BranchOp>(op, op.getDest(),
                                                       operands);
    return matchSuccess();
  }
};

struct LowerCondCondBranchOpPattern
    : public OpRewritePattern<IREESeq::HL::CondBranchOp> {
  using OpRewritePattern<IREESeq::HL::CondBranchOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREESeq::HL::CondBranchOp op,
                                     PatternRewriter &rewriter) const {
    SmallVector<Value *, 8> trueOperands{op.getTrueOperands()};
    SmallVector<Value *, 8> falseOperands{op.getFalseOperands()};

    rewriter.replaceOpWithNewOp<IREESeq::LL::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), trueOperands,
        op.getFalseDest(), falseOperands);
    return matchSuccess();
  }
};

// Rewrites an op into one with all the same operands, results, and attributes.
// Operands and results in the ops must have the same order and attributes must
// have the same name. They must also be constructed properly by the default
// builders.
template <typename SRC, typename DST>
struct LowerIdenticalOpPattern : public OpRewritePattern<SRC> {
  using OpRewritePattern<SRC>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(SRC op, PatternRewriter &rewriter) const {
    SmallVector<Type, 8> resultTypes{op.getOperation()->getResultTypes()};
    SmallVector<Value *, 8> operands{op.getOperation()->getOperands()};

    rewriter.replaceOpWithNewOp<DST>(op, resultTypes, operands, op.getAttrs());
    return this->matchSuccess();
  }
};

}  // namespace

class LowerSequencerDialectPass
    : public FunctionPass<LowerSequencerDialectPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<
        LowerIdenticalOpPattern<IREE::ConstantOp, IREESeq::LL::ConstantOp>,
        LowerIdenticalOpPattern<IREESeq::HL::DispatchOp,
                                IREESeq::LL::DynamicDispatchOp>,
        LowerShapeOpPattern, LowerCopyOpPattern, LowerSliceOpPattern,
        LowerBranchOpPattern, LowerCondCondBranchOpPattern>(&getContext());
#define IDENTICAL_OP_LOWERING(op_name) \
  LowerIdenticalOpPattern<IREESeq::HL::op_name, IREESeq::LL::op_name>
    patterns.insert<
        IDENTICAL_OP_LOWERING(AllocHeapOp), IDENTICAL_OP_LOWERING(CloneOp),
        IDENTICAL_OP_LOWERING(ReshapeOp), IDENTICAL_OP_LOWERING(CallOp),
        IDENTICAL_OP_LOWERING(ReturnOp)>(&getContext());
#undef IDENTICAL_OP_LOWERING

    ConversionTarget target(getContext());
    target.addLegalDialect<IREELLSequencerDialect>();
    target.addLegalOp<FuncOp>();

    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createLowerSequencerDialectPass() {
  return std::make_unique<LowerSequencerDialectPass>();
}

static PassRegistration<LowerSequencerDialectPass> pass(
    "iree-lower-sequencer-dialect",
    "Lowers the IREE HL sequencer dialect to the LL sequencer dialect.");

}  // namespace iree_compiler
}  // namespace mlir
