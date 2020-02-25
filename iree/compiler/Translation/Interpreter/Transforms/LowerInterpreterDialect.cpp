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
#include "iree/compiler/Translation/Interpreter/IR/CommonOps.h"
#include "iree/compiler/Translation/Interpreter/IR/HLDialect.h"
#include "iree/compiler/Translation/Interpreter/IR/HLOps.h"
#include "iree/compiler/Translation/Interpreter/IR/LLDialect.h"
#include "iree/compiler/Translation/Interpreter/IR/LLOps.h"
#include "iree/compiler/Translation/Interpreter/Serialization/BytecodeTables.h"
#include "iree/schemas/bytecode/interpreter_bytecode_v0.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LowerBranchOpPattern
    : public OpRewritePattern<IREEInterp::HL::BranchOp> {
  using OpRewritePattern<IREEInterp::HL::BranchOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREEInterp::HL::BranchOp op,
                                     PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<IREEInterp::LL::BranchOp>(
        op, op.getDest(), op.getOperation()->getOperands());
    return matchSuccess();
  }
};

struct LowerCondCondBranchOpPattern
    : public OpRewritePattern<IREEInterp::HL::CondBranchOp> {
  using OpRewritePattern<IREEInterp::HL::CondBranchOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IREEInterp::HL::CondBranchOp op,
                                     PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<IREEInterp::LL::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), op.getTrueOperands(),
        op.getFalseDest(), op.getFalseOperands());
    return matchSuccess();
  }
};

// Returns true if the op defined by |opName| (like 'iree_ll_interp.reshape')
// uses output operands for results (like iree_ll_interp.add_i) or returns real
// results.
bool opTakesOutputOperands(llvm::StringRef opName) {
  if (!opName.consume_front("iree_ll_interp.")) {
    assert(false && "op not part of IREE LL Interpreter dialect");
    return false;
  }
  auto opcode = GetInterpreterOpcodeByName(opName.str());
  assert(opcode.hasValue() && "op has no corresponding opcode");
  const auto &info = GetInterpreterOpcodeInfo(opcode.getValue());
  for (auto &operand : info.operands) {
    if (operand == iree::OperandEncoding::kOutputSlot ||
        operand == iree::OperandEncoding::kVariadicOutputSlots) {
      return true;
    }
  }
  return false;
}

template <typename SrcOp, typename DstOp>
class SimpleOpLowering : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(SrcOp op,
                                     PatternRewriter &rewriter) const {
    SmallVector<Value, 8> operands{op.getOperation()->getOperands()};

    // Most ops take results as output operands to populate during execution.
    // Certain ops, like reshape, return references to existing memrefs and
    // should still retain their results.
    if (!opTakesOutputOperands(DstOp::getOperationName())) {
      rewriter.replaceOpWithNewOp<DstOp>(
          op, op.getOperation()->getResultTypes(), operands, op.getAttrs());
      return this->matchSuccess();
    }

    SmallVector<Value, 4> replacementValues;
    for (Value result : op.getOperation()->getResults()) {
      auto memRefType = result.getType().cast<MemRefType>();
      if (!memRefType.hasStaticShape()) {
        // TODO(benvanik): real thing here - dynamic shaping required.
        // This should emit a shape calculation based on the operation. Most
        // are likely simple and by running DCE after this we can clean up
        // parts that are static or unused.
        op.emitOpError() << "uses unsupported dynamic shapes";
        return this->matchFailure();
      }
      ArrayRef<Value> dim_pieces;
      auto allocOp = rewriter.create<IREEInterp::LL::AllocHeapOp>(
          op.getLoc(), memRefType, dim_pieces);
      operands.push_back(allocOp);
      replacementValues.push_back(allocOp);
    }
    ArrayRef<Type> resultTypes;
    rewriter.create<DstOp>(op.getLoc(), resultTypes, operands, op.getAttrs());
    rewriter.replaceOp(op, replacementValues);
    return this->matchSuccess();
  }
};

}  // namespace

void populateInterpreterLoweringPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx) {
  patterns.insert<LowerBranchOpPattern, LowerCondCondBranchOpPattern>(ctx);
  patterns.insert<
      SimpleOpLowering<IREEInterp::ConstantOp, IREEInterp::LL::ConstantOp>,
      SimpleOpLowering<IREEInterp::HL::CopyOp, IREEInterp::LL::DynamicCopyOp>,
      SimpleOpLowering<IREEInterp::HL::SliceOp,
                       IREEInterp::LL::DynamicSliceOp>>(ctx);
#define SAME_NAME_SIMPLE_PATTERN(op_name) \
  SimpleOpLowering<IREEInterp::HL::op_name, IREEInterp::LL::op_name>
  // clang-format off
  patterns.insert<
      SAME_NAME_SIMPLE_PATTERN(AssignOp),
      SAME_NAME_SIMPLE_PATTERN(AbsFOp),
      SAME_NAME_SIMPLE_PATTERN(AbsIOp),
      SAME_NAME_SIMPLE_PATTERN(AddFOp),
      SAME_NAME_SIMPLE_PATTERN(AddIOp),
      SAME_NAME_SIMPLE_PATTERN(AllocHeapOp),
      SAME_NAME_SIMPLE_PATTERN(AndOp),
      SAME_NAME_SIMPLE_PATTERN(Atan2FOp),
      SAME_NAME_SIMPLE_PATTERN(BroadcastOp),
      SAME_NAME_SIMPLE_PATTERN(CallOp),
      SAME_NAME_SIMPLE_PATTERN(CeilFOp),
      SAME_NAME_SIMPLE_PATTERN(ClampFOp),
      SAME_NAME_SIMPLE_PATTERN(CloneOp),
      SAME_NAME_SIMPLE_PATTERN(CmpFOp),
      SAME_NAME_SIMPLE_PATTERN(CmpIOp),
      SAME_NAME_SIMPLE_PATTERN(CondAssignOp),
      SAME_NAME_SIMPLE_PATTERN(ConvertSSOp),
      SAME_NAME_SIMPLE_PATTERN(ConvertUUOp),
      SAME_NAME_SIMPLE_PATTERN(ConvertSUOp),
      SAME_NAME_SIMPLE_PATTERN(ConvertUSOp),
      SAME_NAME_SIMPLE_PATTERN(CosFOp),
      SAME_NAME_SIMPLE_PATTERN(DimOp),
      SAME_NAME_SIMPLE_PATTERN(DivFOp),
      SAME_NAME_SIMPLE_PATTERN(DivISOp),
      SAME_NAME_SIMPLE_PATTERN(DivIUOp),
      SAME_NAME_SIMPLE_PATTERN(ExpFOp),
      SAME_NAME_SIMPLE_PATTERN(LogFOp),
      SAME_NAME_SIMPLE_PATTERN(RsqrtFOp),
      SAME_NAME_SIMPLE_PATTERN(SqrtFOp),
      SAME_NAME_SIMPLE_PATTERN(FloorFOp),
      SAME_NAME_SIMPLE_PATTERN(LengthOp),
      SAME_NAME_SIMPLE_PATTERN(MatMulFOp),
      SAME_NAME_SIMPLE_PATTERN(MatMulIOp),
      SAME_NAME_SIMPLE_PATTERN(MaxFOp),
      SAME_NAME_SIMPLE_PATTERN(MaxISOp),
      SAME_NAME_SIMPLE_PATTERN(MaxIUOp),
      SAME_NAME_SIMPLE_PATTERN(MinFOp),
      SAME_NAME_SIMPLE_PATTERN(MinISOp),
      SAME_NAME_SIMPLE_PATTERN(MinIUOp),
      SAME_NAME_SIMPLE_PATTERN(MulAddFOp),
      SAME_NAME_SIMPLE_PATTERN(MulAddIOp),
      SAME_NAME_SIMPLE_PATTERN(MulFOp),
      SAME_NAME_SIMPLE_PATTERN(MulIOp),
      SAME_NAME_SIMPLE_PATTERN(NotOp),
      SAME_NAME_SIMPLE_PATTERN(OrOp),
      SAME_NAME_SIMPLE_PATTERN(PadOp),
      SAME_NAME_SIMPLE_PATTERN(RankOp),
      SAME_NAME_SIMPLE_PATTERN(ReduceSumIOp),
      SAME_NAME_SIMPLE_PATTERN(ReduceSumFOp),
      SAME_NAME_SIMPLE_PATTERN(ReduceMinIOp),
      SAME_NAME_SIMPLE_PATTERN(ReduceMinFOp),
      SAME_NAME_SIMPLE_PATTERN(ReduceMaxIOp),
      SAME_NAME_SIMPLE_PATTERN(ReduceMaxFOp),
      SAME_NAME_SIMPLE_PATTERN(RemFOp),
      SAME_NAME_SIMPLE_PATTERN(RemISOp),
      SAME_NAME_SIMPLE_PATTERN(RemIUOp),
      SAME_NAME_SIMPLE_PATTERN(ReshapeOp),
      SAME_NAME_SIMPLE_PATTERN(ReturnOp),
      SAME_NAME_SIMPLE_PATTERN(SelectOp),
      SAME_NAME_SIMPLE_PATTERN(ShapeOp),
      SAME_NAME_SIMPLE_PATTERN(ShiftLeftOp),
      SAME_NAME_SIMPLE_PATTERN(ShiftRightArithmeticOp),
      SAME_NAME_SIMPLE_PATTERN(ShiftRightLogicalOp),
      SAME_NAME_SIMPLE_PATTERN(SinFOp),
      SAME_NAME_SIMPLE_PATTERN(SubFOp),
      SAME_NAME_SIMPLE_PATTERN(SubIOp),
      SAME_NAME_SIMPLE_PATTERN(TanhFOp),
      SAME_NAME_SIMPLE_PATTERN(TileOp),
      SAME_NAME_SIMPLE_PATTERN(TransposeOp),
      SAME_NAME_SIMPLE_PATTERN(ReverseOp),
      SAME_NAME_SIMPLE_PATTERN(XorOp)>(ctx);
  // clang-format on
#undef SAME_NAME_SIMPLE_PATTERN
}

namespace {
class LowerInterpreterDialectPass
    : public FunctionPass<LowerInterpreterDialectPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateInterpreterLoweringPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<IREELLInterpreterDialect>();
    target.addLegalOp<FuncOp, mlir::ReturnOp>();
    if (failed(applyFullConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLowerInterpreterDialectPass() {
  return std::make_unique<LowerInterpreterDialectPass>();
}

static PassRegistration<LowerInterpreterDialectPass> pass(
    "lower-iree-interpreter-hl-to-ll", "Lowers IREE HL ops to IREE LL ops");

}  // namespace iree_compiler
}  // namespace mlir
