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

#include "compiler/IR/Dialect.h"
#include "compiler/IR/Ops.h"
#include "compiler/IR/Sequencer/HLDialect.h"
#include "compiler/IR/Sequencer/HLOps.h"
#include "compiler/IR/Sequencer/LLDialect.h"
#include "compiler/IR/Sequencer/LLOps.h"
#include "compiler/IR/StructureOps.h"
#include "compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
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

template <typename SrcOp>
class SequencerLoweringPattern : public OpConversionPattern<SrcOp> {
 public:
  SequencerLoweringPattern(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern<SrcOp>(context), typeConverter_(typeConverter) {}

 protected:
  TypeConverter &typeConverter_;
};

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

struct LowerSliceOpPattern
    : public SequencerLoweringPattern<IREESeq::HL::SliceOp> {
  using SequencerLoweringPattern::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      IREESeq::HL::SliceOp op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<IREESeq::HL::SliceOp> operandAdaptor(operands);
    auto range = computeRange(op.getLoc(), operandAdaptor.src(),
                              operandAdaptor.indices(),
                              operandAdaptor.lengths(), rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::LL::DynamicSliceOp>(
        op, typeConverter_.convertType(op.getType()),
        ArrayRef<Value *>{operandAdaptor.src(), range.first, range.second},
        op.getAttrs());
    return matchSuccess();
  }
};

struct LowerShapeOpPattern
    : public SequencerLoweringPattern<IREESeq::HL::ShapeOp> {
  using SequencerLoweringPattern::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      IREESeq::HL::ShapeOp op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *shapeMemRef =
        rewriter
            .create<IREESeq::LL::AllocHeapOp>(
                op.getLoc(),
                MemRefType::get({op.getType().cast<ShapedType>().getRank()},
                                rewriter.getIntegerType(64)),
                ArrayRef<Value *>{})
            .getResult();
    op.replaceAllUsesWith(shapeMemRef);
    rewriter.replaceOpWithNewOp<IREESeq::LL::ShapeOp>(op, operands[0],
                                                      shapeMemRef);
    return matchSuccess();
  }
};

struct LowerCopyOpPattern
    : public SequencerLoweringPattern<IREESeq::HL::CopyOp> {
  using SequencerLoweringPattern::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      IREESeq::HL::CopyOp op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<IREESeq::HL::CopyOp> operandAdaptor(operands);
    auto *srcOffsetMemRef =
        computeOffset(op.getLoc(), operandAdaptor.src(),
                      operandAdaptor.srcIndices(), rewriter);
    auto dstRange = computeRange(op.getLoc(), operandAdaptor.dst(),
                                 operandAdaptor.dstIndices(),
                                 operandAdaptor.lengths(), rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::LL::DynamicCopyOp>(
        op, operandAdaptor.src(), srcOffsetMemRef, operandAdaptor.dst(),
        dstRange.first, dstRange.second);
    return matchSuccess();
  }
};

struct LowerFillOpPattern
    : public SequencerLoweringPattern<IREESeq::HL::FillOp> {
  using SequencerLoweringPattern::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      IREESeq::HL::FillOp op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<IREESeq::HL::FillOp> operandAdaptor(operands);
    auto dstRange = computeRange(op.getLoc(), operandAdaptor.dst(),
                                 operandAdaptor.dstIndices(),
                                 operandAdaptor.lengths(), rewriter);
    rewriter.replaceOpWithNewOp<IREESeq::LL::DynamicFillOp>(
        op, operandAdaptor.value(), operandAdaptor.dst(), dstRange.first,
        dstRange.second);
    return matchSuccess();
  }
};

struct LowerBranchOpPattern
    : public SequencerLoweringPattern<IREESeq::HL::BranchOp> {
  using SequencerLoweringPattern<
      IREESeq::HL::BranchOp>::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      IREESeq::HL::BranchOp op, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::LL::BranchOp>(op, destinations[0],
                                                       operands[0]);
    return matchSuccess();
  }
};

struct LowerCondCondBranchOpPattern
    : public SequencerLoweringPattern<IREESeq::HL::CondBranchOp> {
  using SequencerLoweringPattern<
      IREESeq::HL::CondBranchOp>::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      IREESeq::HL::CondBranchOp op, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREESeq::LL::CondBranchOp>(
        op, properOperands[0],
        destinations[IREESeq::HL::CondBranchOp::trueIndex],
        operands[IREESeq::HL::CondBranchOp::trueIndex],
        destinations[IREESeq::HL::CondBranchOp::falseIndex],
        operands[IREESeq::HL::CondBranchOp::falseIndex]);
    return matchSuccess();
  }
};

// Rewrites an op into one with all the same operands, results, and attributes.
// Operands and results in the ops must have the same order and attributes must
// have the same name. They must also be constructed properly by the default
// builders.
template <typename SRC, typename DST>
struct LowerIdenticalOpPattern : public SequencerLoweringPattern<SRC> {
  using SequencerLoweringPattern<SRC>::SequencerLoweringPattern;

  PatternMatchResult matchAndRewrite(
      SRC op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 8> originalResultTypes{
        op.getOperation()->getResultTypes()};
    SmallVector<Type, 8> resultTypes;
    if (failed(this->typeConverter_.convertTypes(originalResultTypes,
                                                 resultTypes))) {
      op.emitOpError() << "Failed to convert result types";
      return this->matchFailure();
    }
    rewriter.replaceOpWithNewOp<DST>(op, resultTypes, operands, op.getAttrs());
    return this->matchSuccess();
  }
};

}  // namespace

class LowerSequencerDialectPass : public ModulePass<LowerSequencerDialectPass> {
 public:
  void runOnModule() override {
    auto *ctx = &getContext();
    LLTypeConverter typeConverter(ctx);
    OwningRewritePatternList patterns;
    patterns.insert<
        LowerIdenticalOpPattern<IREE::ConstantOp, IREESeq::LL::ConstantOp>,
        LowerIdenticalOpPattern<IREESeq::HL::DispatchOp,
                                IREESeq::LL::DynamicDispatchOp>,
        LowerShapeOpPattern, LowerCopyOpPattern, LowerSliceOpPattern,
        LowerBranchOpPattern, LowerCondCondBranchOpPattern>(ctx, typeConverter);
#define IDENTICAL_OP_LOWERING(op_name) \
  LowerIdenticalOpPattern<IREESeq::HL::op_name, IREESeq::LL::op_name>
    patterns.insert<
        IDENTICAL_OP_LOWERING(AllocHeapOp), IDENTICAL_OP_LOWERING(CloneOp),
        IDENTICAL_OP_LOWERING(ReshapeOp), IDENTICAL_OP_LOWERING(CallOp),
        IDENTICAL_OP_LOWERING(ReturnOp)>(ctx, typeConverter);
#undef IDENTICAL_OP_LOWERING

    mlir::populateFuncOpTypeConversionPattern(patterns, ctx, typeConverter);
    ConversionTarget target(*ctx);
    target.addLegalDialect<IREELLSequencerDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    // TODO(b/142791494): The conversion framework will recurse into the
    // executable if we just call it on the top-level module. This can't be a
    // function pass because type conversion replaces the original functions.
    auto funcsIt = getModule().getOps<FuncOp>();
    SmallVector<Operation *, 4> funcs(funcsIt.begin(), funcsIt.end());

    if (failed(applyFullConversion(funcs, target, patterns, &typeConverter))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createLowerSequencerDialectPass() {
  return std::make_unique<LowerSequencerDialectPass>();
}

static PassRegistration<LowerSequencerDialectPass> pass(
    "iree-lower-sequencer-dialect",
    "Lowers the IREE HL sequencer dialect to the LL sequencer dialect.");

}  // namespace iree_compiler
}  // namespace mlir
