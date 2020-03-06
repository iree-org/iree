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

#include "iree/compiler/Translation/Interpreter/IR/HLOps.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/Interpreter/Utils/OpCreationUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREEInterp {
namespace HL {

//===----------------------------------------------------------------------===//
// iree_hl_interp.call
//===----------------------------------------------------------------------===//

FunctionType CallOp::getCalleeType() {
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, getResultTypes(), getContext());
}

//===----------------------------------------------------------------------===//
// iree_hl_interp.br
//===----------------------------------------------------------------------===//

Block *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

Optional<OperandRange> BranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return getOperands();
}

bool BranchOp::canEraseSuccessorOperand() { return true; }

//===----------------------------------------------------------------------===//
// iree_hl_interp.cond_br
//===----------------------------------------------------------------------===//

static ParseResult parseCondBranchOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Value, 4> destOperands;
  Block *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the condition.
  Type int1Ty = parser.getBuilder().getI1Type();
  if (parser.parseOperand(condInfo) || parser.parseComma() ||
      parser.resolveOperand(condInfo, int1Ty, result.operands)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected condition type was boolean (i1)");
  }

  // Parse the true successor.
  SmallVector<Value, 4> trueOperands;
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, trueOperands))
    return failure();
  result.addSuccessors(dest);
  result.addOperands(trueOperands);

  // Parse the false successor.
  SmallVector<Value, 4> falseOperands;
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, falseOperands))
    return failure();
  result.addSuccessors(dest);
  result.addOperands(falseOperands);
  result.addAttribute(CondBranchOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI32VectorAttr(
                          {1, static_cast<int32_t>(trueOperands.size()),
                           static_cast<int32_t>(falseOperands.size())}));

  return success();
}

static void printCondBranchOp(OpAsmPrinter &p, CondBranchOp op) {
  p << "iree_hl_interp.cond_br ";
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.trueDest(), op.trueDestOperands());
  p << ", ";
  p.printSuccessorAndUseList(op.falseDest(), op.falseDestOperands());
}

Optional<OperandRange> CondBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? getTrueOperands() : getFalseOperands();
}

bool CondBranchOp::canEraseSuccessorOperand() { return true; }

//===----------------------------------------------------------------------===//
// iree_hl_interp.clone
//===----------------------------------------------------------------------===//

OpFoldResult CloneOp::fold(ArrayRef<Attribute> operands) {
  // If this is the only usage, we know the clone is unnecessary.
  // TODO(b/135053584) More sophisticated analysis.
  if (src().hasOneUse()) return src();
  return {};
}

//===----------------------------------------------------------------------===//
// iree_hl_interp.concat
//===----------------------------------------------------------------------===//

namespace {
struct ConcatToCopies : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(ConcatOp concatOp,
                                     PatternRewriter &rewriter) const override {
    auto finalType = concatOp.getResult().getType().cast<ShapedType>();
    auto loc = concatOp.getLoc();
    std::vector<Value> dimPieces;
    auto dst =
        rewriter.create<IREEInterp::HL::AllocHeapOp>(loc, finalType, dimPieces);

    llvm::SmallVector<int64_t, 4> zeroOffset(finalType.getRank(), 0);
    auto srcIndices = createArrayConstant(rewriter, loc, zeroOffset);

    auto concatDimension = concatOp.dimension().getZExtValue();
    llvm::SmallVector<int64_t, 4> dstIndices(finalType.getRank(), 0);
    for (auto src : concatOp.srcs()) {
      auto srcShape = src.getType().cast<ShapedType>().getShape();
      auto lengths = createArrayConstant(rewriter, loc, srcShape);
      auto dstIndicesOp = createArrayConstant(rewriter, loc, dstIndices);
      rewriter.create<IREEInterp::HL::CopyOp>(loc, src, srcIndices, dst,
                                              dstIndicesOp, lengths);
      dstIndices[concatDimension] += srcShape[concatDimension];
    }

    concatOp.replaceAllUsesWith(dst.getResult());

    return matchSuccess();
  }
};
}  // namespace

void ConcatOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<ConcatToCopies>(context);
}

#define GET_OP_CLASSES
#include "iree/compiler/Translation/Interpreter/IR/HLOps.cpp.inc"

}  // namespace HL
}  // namespace IREEInterp
}  // namespace iree_compiler
}  // namespace mlir
