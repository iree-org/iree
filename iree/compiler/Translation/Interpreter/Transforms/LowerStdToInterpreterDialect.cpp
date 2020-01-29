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

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/Interpreter/IR/HLDialect.h"
#include "iree/compiler/Translation/Interpreter/IR/HLOps.h"
#include "iree/compiler/Translation/Interpreter/IR/LLDialect.h"
#include "iree/compiler/Translation/Interpreter/Transforms/ConversionUtils.h"
#include "iree/compiler/Translation/Interpreter/Utils/MemRefUtils.h"
#include "iree/compiler/Translation/Interpreter/Utils/OpCreationUtils.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Allocator.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct CallOpLowering : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      CallOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<CallOp>(op);
    auto calleeType = callOp.getCalleeType();
    rewriter.replaceOpWithNewOp<IREEInterp::HL::CallOp>(
        op, callOp.getCallee(), calleeType.getResults(), operands);
    return matchSuccess();
  }
};

struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREEInterp::HL::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

struct BranchOpLowering : public OpConversionPattern<BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      BranchOp op, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREEInterp::HL::BranchOp>(op, destinations[0],
                                                          operands[0]);
    return this->matchSuccess();
  }
};

struct CondBranchOpLowering : public OpConversionPattern<CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      CondBranchOp op, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto condValue = loadAccessValue(op.getLoc(), properOperands[0], rewriter);
    rewriter.replaceOpWithNewOp<IREEInterp::HL::CondBranchOp>(
        op, condValue, destinations[IREEInterp::HL::CondBranchOp::trueIndex],
        operands[IREEInterp::HL::CondBranchOp::trueIndex],
        destinations[IREEInterp::HL::CondBranchOp::falseIndex],
        operands[IREEInterp::HL::CondBranchOp::falseIndex]);
    return this->matchSuccess();
  }
};

template <typename SrcOp, typename DstOp>
struct CompareOpLowering : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      SrcOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto lhValue = loadAccessValue(op.getLoc(), operands[0], rewriter);
    auto rhValue = loadAccessValue(op.getLoc(), operands[1], rewriter);

    lhValue = wrapAsMemRef(lhValue, op, rewriter);
    rhValue = wrapAsMemRef(rhValue, op, rewriter);

    // TODO(benvanik): map predicate to stable value.
    auto predicate =
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getPredicate()));

    auto dstType = convertLegacyTypeToMemRef(op.getResult());
    auto midOp = rewriter.create<DstOp>(op.getLoc(), dstType, predicate,
                                        lhValue, rhValue);

    auto result = wrapAsTensor(midOp.getResult(), op, rewriter);
    rewriter.replaceOp(
        op, {loadResultValue(op.getLoc(), op.getType(), result, rewriter)});
    return this->matchSuccess();
  }
};

struct CmpIOpLowering
    : public CompareOpLowering<CmpIOp, IREEInterp::HL::CmpIOp> {
  using CompareOpLowering::CompareOpLowering;
};

struct CmpFOpLowering
    : public CompareOpLowering<CmpFOp, IREEInterp::HL::CmpFOp> {
  using CompareOpLowering::CompareOpLowering;
};

struct AllocOpLowering : public OpConversionPattern<AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      AllocOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): replace with length computation.
    rewriter.replaceOpWithNewOp<IREEInterp::HL::AllocHeapOp>(op, op.getType(),
                                                             operands);
    return matchSuccess();
  }
};

struct DeallocOpLowering : public OpConversionPattern<DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      DeallocOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREEInterp::HL::DiscardOp>(op, operands[0]);
    return matchSuccess();
  }
};

struct LoadOpLowering : public OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(LoadOp loadOp,
                                     PatternRewriter &rewriter) const override {
    if (loadOp.getMemRefType().getRank() != 0) {
      loadOp.emitError() << "Cannot lower load of non-scalar";
      return matchFailure();
    }
    ArrayRef<Value> dimPieces;
    auto dst =
        rewriter
            .create<AllocOp>(loadOp.getLoc(), loadOp.getMemRefType(), dimPieces)
            .getResult();
    auto emptyArrayMemref = createArrayConstant(rewriter, loadOp.getLoc(), {});
    rewriter.create<IREEInterp::HL::CopyOp>(
        loadOp.getLoc(), loadOp.getMemRef(),
        /*srcIndices=*/emptyArrayMemref, dst,
        /*dstIndices=*/emptyArrayMemref, /*lengths=*/emptyArrayMemref);

    rewriter.replaceOpWithNewOp<IREEInterp::MemRefToScalarOp>(loadOp, dst);

    return matchSuccess();
  }
};

struct StoreOpLowering : public OpRewritePattern<StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(StoreOp storeOp,
                                     PatternRewriter &rewriter) const override {
    if (storeOp.getMemRefType().getRank() != 0) {
      storeOp.emitError() << "Cannot lower store of non-scalar";
      return matchFailure();
    }

    auto src = rewriter.create<IREEInterp::ScalarToMemRefOp>(
        storeOp.getLoc(), storeOp.getValueToStore());

    auto emptyArrayMemref = createArrayConstant(rewriter, storeOp.getLoc(), {});
    rewriter.replaceOpWithNewOp<IREEInterp::HL::CopyOp>(
        storeOp, src, /*srcIndices=*/emptyArrayMemref, storeOp.getMemRef(),
        /*dstIndices=*/emptyArrayMemref, /*lengths=*/emptyArrayMemref);

    return matchSuccess();
  }
};

#define UNARY_OP_LOWERING(StdOpType, IREEOpType)                               \
  struct StdOpType##Lowering : public UnaryOpLowering<StdOpType, IREEOpType> { \
    using UnaryOpLowering::UnaryOpLowering;                                    \
  };

#define BINARY_OP_LOWERING(StdOpType, IREEOpType)        \
  struct StdOpType##Lowering                             \
      : public BinaryOpLowering<StdOpType, IREEOpType> { \
    using BinaryOpLowering::BinaryOpLowering;            \
  };

#define TERNARY_OP_LOWERING(StdOpType, IREEOpType)        \
  struct StdOpType##Lowering                              \
      : public TernaryOpLowering<StdOpType, IREEOpType> { \
    using TernaryOpLowering::TernaryOpLowering;           \
  };

// UNARY_OP_LOWERING(RankOp, IREEInterp::HL::RankOp);
UNARY_OP_LOWERING(DimOp, IREEInterp::HL::DimOp);
// UNARY_OP_LOWERING(ShapeOp, IREEInterp::HL::ShapeOp);
// UNARY_OP_LOWERING(LengthOp, IREEInterp::HL::LengthOp);

// UNARY_OP_LOWERING(NotOp, IREEInterp::HL::NotOp);
BINARY_OP_LOWERING(AndOp, IREEInterp::HL::AndOp);
BINARY_OP_LOWERING(OrOp, IREEInterp::HL::OrOp);
// BINARY_OP_LOWERING(XorOp, IREEInterp::HL::XorOp);
// BINARY_OP_LOWERING(ShiftLeftOp, IREEInterp::HL::ShiftLeftOp);
// BINARY_OP_LOWERING(ShiftRightLogicalOp, IREEInterp::HL::ShiftRightLogicalOp);
// BINARY_OP_LOWERING(ShiftRightArithmeticOp,
// IREEInterp::HL::ShiftRightArithmeticOp);

BINARY_OP_LOWERING(AddIOp, IREEInterp::HL::AddIOp);
BINARY_OP_LOWERING(AddFOp, IREEInterp::HL::AddFOp);
BINARY_OP_LOWERING(SubIOp, IREEInterp::HL::SubIOp);
BINARY_OP_LOWERING(SubFOp, IREEInterp::HL::SubFOp);
// UNARY_OP_LOWERING(AbsIOp, IREEInterp::HL::AbsIOp);
// UNARY_OP_LOWERING(AbsFOp, IREEInterp::HL::AbsFOp);
BINARY_OP_LOWERING(MulIOp, IREEInterp::HL::MulIOp);
BINARY_OP_LOWERING(MulFOp, IREEInterp::HL::MulFOp);
BINARY_OP_LOWERING(SignedDivIOp, IREEInterp::HL::DivISOp);
BINARY_OP_LOWERING(UnsignedDivIOp, IREEInterp::HL::DivIUOp);
BINARY_OP_LOWERING(DivFOp, IREEInterp::HL::DivFOp);
BINARY_OP_LOWERING(SignedRemIOp, IREEInterp::HL::RemISOp);
BINARY_OP_LOWERING(UnsignedRemIOp, IREEInterp::HL::RemIUOp);
BINARY_OP_LOWERING(RemFOp, IREEInterp::HL::RemFOp);
// BINARY_OP_LOWERING(MulAddIOp, IREEInterp::HL::MulAddIOp);
// BINARY_OP_LOWERING(MulAddFOp, IREEInterp::HL::MulAddFOp);
// UNARY_OP_LOWERING(ExpFOp, IREEInterp::HL::ExpFOp);
// UNARY_OP_LOWERING(LogFOp, IREEInterp::HL::LogFOp);
// UNARY_OP_LOWERING(RsqrtFOp, IREEInterp::HL::RsqrtFOp);
// UNARY_OP_LOWERING(SqrtFOp, IREEInterp::HL::SqrtFOp);
// UNARY_OP_LOWERING(CosFOp, IREEInterp::HL::CosFOp);
// UNARY_OP_LOWERING(SinFOp, IREEInterp::HL::SinFOp);
// UNARY_OP_LOWERING(TanhFOp, IREEInterp::HL::TanhFOp);
// UNARY_OP_LOWERING(Atan2FOp, IREEInterp::HL::Atan2FOp);

// BINARY_OP_LOWERING(MinISOp, IREEInterp::HL::MinISOp);
// BINARY_OP_LOWERING(MinIUOp, IREEInterp::HL::MinIUOp);
// BINARY_OP_LOWERING(MinFOp, IREEInterp::HL::MinFOp);
// BINARY_OP_LOWERING(MaxISOp, IREEInterp::HL::MaxISOp);
// BINARY_OP_LOWERING(MaxIUOp, IREEInterp::HL::MaxIUOp);
// BINARY_OP_LOWERING(MaxFOp, IREEInterp::HL::MaxFOp);
// TERNARY_OP_LOWERING(ClampFOp, IREEInterp::HL::ClampFOp);
// UNARY_OP_LOWERING(FloorFOp, IREEInterp::HL::FloorFOp);
// UNARY_OP_LOWERING(CeilFOp, IREEInterp::HL::CeilFOp);

}  // namespace

void populateLowerStdToInterpreterPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  patterns.insert<
      // Control flow.
      CallOpLowering, ReturnOpLowering, BranchOpLowering, CondBranchOpLowering,
      CmpIOpLowering, CmpFOpLowering,
      // Memory management.
      AllocOpLowering, DeallocOpLowering, LoadOpLowering, StoreOpLowering,
      // Shape operations.
      DimOpLowering,
      // Logical ops.
      AndOpLowering, OrOpLowering,
      // Arithmetic ops.
      AddIOpLowering, AddFOpLowering, SubIOpLowering, SubFOpLowering,
      MulIOpLowering, MulFOpLowering, SignedDivIOpLowering,
      UnsignedDivIOpLowering, DivFOpLowering, RemFOpLowering,
      SignedRemIOpLowering, UnsignedRemIOpLowering>(ctx);
}

}  // namespace iree_compiler
}  // namespace mlir
