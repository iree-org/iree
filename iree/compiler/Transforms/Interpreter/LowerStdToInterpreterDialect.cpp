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
#include "iree/compiler/IR/Interpreter/HLDialect.h"
#include "iree/compiler/IR/Interpreter/HLOps.h"
#include "iree/compiler/IR/Interpreter/LLDialect.h"
#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/Transforms/ConversionUtils.h"
#include "iree/compiler/Utils/MemRefUtils.h"
#include "third_party/llvm/llvm/include/llvm/ADT/DenseSet.h"
#include "third_party/llvm/llvm/include/llvm/Support/Allocator.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Dialect/StandardOps/Ops.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Attributes.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/BlockAndValueMapping.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Function.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Location.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Module.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/StandardTypes.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/PassRegistry.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

struct ConstantOpLowering : public ConversionPattern {
  explicit ConstantOpLowering(MLIRContext *context)
      : ConversionPattern(ConstantOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto midOp = rewriter.create<IREE::ConstantOp>(
        op->getLoc(), cast<ConstantOp>(op).getValue());

    auto result = wrapAsTensor(midOp.getResult(), op, rewriter);
    rewriter.replaceOp(
        op, {loadResultValue(op->getLoc(), op->getResult(0)->getType(), result,
                             rewriter)});
    return matchSuccess();
  }
};

class CallOpLowering : public ConversionPattern {
 public:
  explicit CallOpLowering(MLIRContext *context)
      : ConversionPattern(CallOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<CallOp>(op);
    auto calleeType = callOp.getCalleeType();
    rewriter.replaceOpWithNewOp<IREEInterp::HL::CallOp>(
        op, callOp.getCallee(), calleeType.getResults(), operands);
    return matchSuccess();
  }
};

class CallIndirectOpLowering : public ConversionPattern {
 public:
  explicit CallIndirectOpLowering(MLIRContext *context)
      : ConversionPattern(CallIndirectOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<CallIndirectOp>(op);
    rewriter.replaceOpWithNewOp<IREEInterp::HL::CallIndirectOp>(
        op, callOp.getCallee(), operands);
    return matchSuccess();
  }
};

struct ReturnOpLowering : public ConversionPattern {
  explicit ReturnOpLowering(MLIRContext *context)
      : ConversionPattern(ReturnOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREEInterp::HL::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

struct BranchOpLowering : public ConversionPattern {
  explicit BranchOpLowering(MLIRContext *context)
      : ConversionPattern(BranchOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREEInterp::HL::BranchOp>(op, destinations[0],
                                                          operands[0]);
    return this->matchSuccess();
  }
};

struct CondBranchOpLowering : public ConversionPattern {
  explicit CondBranchOpLowering(MLIRContext *context)
      : ConversionPattern(CondBranchOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value *>> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *condValue =
        loadAccessValue(op->getLoc(), properOperands[0], rewriter);
    rewriter.replaceOpWithNewOp<IREEInterp::HL::CondBranchOp>(
        op, condValue, destinations[IREEInterp::HL::CondBranchOp::trueIndex],
        operands[IREEInterp::HL::CondBranchOp::trueIndex],
        destinations[IREEInterp::HL::CondBranchOp::falseIndex],
        operands[IREEInterp::HL::CondBranchOp::falseIndex]);
    return this->matchSuccess();
  }
};

template <typename SrcOp, typename DstOp>
struct CompareOpLowering : public ConversionPattern {
  explicit CompareOpLowering(MLIRContext *context)
      : ConversionPattern(SrcOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto lhValue = loadAccessValue(op->getLoc(), operands[0], rewriter);
    auto rhValue = loadAccessValue(op->getLoc(), operands[1], rewriter);

    lhValue = wrapAsMemRef(lhValue, op, rewriter);
    rhValue = wrapAsMemRef(rhValue, op, rewriter);

    // TODO(benvanik): map predicate to stable value.
    auto predicate = rewriter.getI32IntegerAttr(
        static_cast<int32_t>(dyn_cast<SrcOp>(op).getPredicate()));

    auto dstType = getMemRefType(op->getResult(0), rewriter);
    auto midOp = rewriter.create<DstOp>(op->getLoc(), dstType, predicate,
                                        lhValue, rhValue);

    auto result = wrapAsTensor(midOp.getResult(), op, rewriter);
    rewriter.replaceOp(
        op, {loadResultValue(op->getLoc(), op->getResult(0)->getType(), result,
                             rewriter)});
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

struct AllocOpLowering : public ConversionPattern {
  explicit AllocOpLowering(MLIRContext *context)
      : ConversionPattern(AllocOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): replace with length computation.
    rewriter.replaceOpWithNewOp<IREEInterp::HL::AllocHeapOp>(
        op, *op->getResultTypes().begin(), operands);
    return matchSuccess();
  }
};

struct DeallocOpLowering : public ConversionPattern {
  explicit DeallocOpLowering(MLIRContext *context)
      : ConversionPattern(DeallocOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREEInterp::HL::DiscardOp>(op, operands[0]);
    return matchSuccess();
  }
};

struct ExtractElementOpLowering : public ConversionPattern {
  explicit ExtractElementOpLowering(MLIRContext *context)
      : ConversionPattern(ExtractElementOp::getOperationName(), 1, context) {}
  PatternMatchResult matchAndRewrite(
      Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto extractOp = cast<ExtractElementOp>(op);
    Value *memRefInput = wrapAsMemRef(
        loadAccessValue(op->getLoc(), extractOp.getAggregate(), rewriter), op,
        rewriter);

    SmallVector<Value *, 4> indices = {extractOp.indices().begin(),
                                       extractOp.indices().end()};
    rewriter.replaceOpWithNewOp<LoadOp>(op, memRefInput, indices);
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
BINARY_OP_LOWERING(DivISOp, IREEInterp::HL::DivISOp);
BINARY_OP_LOWERING(DivIUOp, IREEInterp::HL::DivIUOp);
BINARY_OP_LOWERING(DivFOp, IREEInterp::HL::DivFOp);
// BINARY_OP_LOWERING(MulAddIOp, IREEInterp::HL::MulAddIOp);
// BINARY_OP_LOWERING(MulAddFOp, IREEInterp::HL::MulAddFOp);
// UNARY_OP_LOWERING(ExpFOp, IREEInterp::HL::ExpFOp);
// UNARY_OP_LOWERING(LogFOp, IREEInterp::HL::LogFOp);
// UNARY_OP_LOWERING(RsqrtFOp, IREEInterp::HL::RsqrtFOp);
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

class LowerStdToInterpreterDialectPass
    : public FunctionPass<LowerStdToInterpreterDialectPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<
        ConstantOpLowering,
        // Control flow.
        CallOpLowering, CallIndirectOpLowering, ReturnOpLowering,
        BranchOpLowering, CondBranchOpLowering, CmpIOpLowering, CmpFOpLowering,
        // Memory management.
        AllocOpLowering, DeallocOpLowering, ExtractElementOpLowering,
        // Shape operations.
        DimOpLowering,
        // Logical ops.
        AndOpLowering, OrOpLowering,
        // Arithmetic ops.
        AddIOpLowering, AddFOpLowering, SubIOpLowering, SubFOpLowering,
        MulIOpLowering, MulFOpLowering, DivISOpLowering, DivIUOpLowering,
        DivFOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<IREEHLInterpreterDialect, IREELLInterpreterDialect,
                           IREEDialect>();
    target.addLegalOp<LoadOp, StoreOp, FuncOp, ModuleOp>();
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp constOp) {
      // std.constant is legal for index integers.
      return constOp.getValue().isa<IntegerAttr>() &&
             constOp.getType().isIndex();
    });
    // This is only able to be a full conversion because we know that we do it
    // last. Other conversions (e.g. XLA->IREE) can't be full because we might
    // still have e.g. std ops.
    // TODO(b/139012931) Move fully to the conversion framework.
    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createLowerStdToInterpreterDialectPass() {
  return std::make_unique<LowerStdToInterpreterDialectPass>();
}

static PassRegistration<LowerStdToInterpreterDialectPass> pass(
    "lower-std-to-iree-interpreter",
    "Convert all std functions to the IREE dialect");

}  // namespace iree_compiler
}  // namespace mlir
