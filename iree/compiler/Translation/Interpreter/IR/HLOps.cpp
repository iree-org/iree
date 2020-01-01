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

#include "iree/compiler/IR/Ops.h"
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

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &state) {
  FlatSymbolRefAttr calleeAttr;
  FunctionType calleeType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto calleeLoc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, "callee", state.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.addTypesToList(calleeType.getResults(), state.types) ||
      parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                             state.operands)) {
    return failure();
  }
  return success();
}

static void printCallOp(OpAsmPrinter &p, CallOp op) {
  p << "iree_hl_interp.call " << op.getAttr("callee") << '(';
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : ";
  p.printType(op.getCalleeType());
}

FunctionType CallOp::getCalleeType() {
  SmallVector<Type, 4> resultTypes(getResultTypes());
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// iree_hl_interp.call_indirect
//===----------------------------------------------------------------------===//

static ParseResult parseCallIndirectOp(OpAsmParser &parser,
                                       OperationState &result) {
  FunctionType calleeType;
  OpAsmParser::OperandType callee;
  llvm::SMLoc operandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  return failure(
      parser.parseOperand(callee) || parser.getCurrentLocation(&operandsLoc) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.resolveOperand(callee, calleeType, result.operands) ||
      parser.resolveOperands(operands, calleeType.getInputs(), operandsLoc,
                             result.operands) ||
      parser.addTypesToList(calleeType.getResults(), result.types));
}

static void printCallIndirectOp(OpAsmPrinter &p, CallIndirectOp op) {
  p << "iree_hl_interp.call_indirect ";
  p.printOperand(op.getCallee());
  p << '(';
  auto operandRange = op.getOperands();
  p.printOperands(++operandRange.begin(), operandRange.end());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : " << op.getCallee()->getType();
}

//===----------------------------------------------------------------------===//
// iree_hl_interp.return
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, state.operands));
}

static void printReturnOp(OpAsmPrinter &p, ReturnOp op) {
  p << "iree_hl_interp.return";
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.operand_begin(), op.operand_end());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// iree_hl_interp.br
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState &result) {
  Block *dest;
  SmallVector<Value, 4> destOperands;
  if (parser.parseSuccessorAndUseList(dest, destOperands)) return failure();
  result.addSuccessor(dest, destOperands);
  return success();
}

static void printBranchOp(OpAsmPrinter &p, BranchOp op) {
  p << "iree_hl_interp.br ";
  p.printSuccessorAndUseList(op.getOperation(), 0);
}

Block *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

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
  if (parser.parseSuccessorAndUseList(dest, destOperands)) return failure();
  result.addSuccessor(dest, destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  result.addSuccessor(dest, destOperands);

  return success();
}

static void printCondBranchOp(OpAsmPrinter &p, CondBranchOp op) {
  p << "iree_hl_interp.cond_br ";
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
}

//===----------------------------------------------------------------------===//
// iree_hl_interp.clone
//===----------------------------------------------------------------------===//

OpFoldResult CloneOp::fold(ArrayRef<Attribute> operands) {
  // If this is the only usage, we know the clone is unnecessary.
  // TODO(b/135053584) More sophisticated analysis.
  if (src()->hasOneUse()) return src();
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
    auto finalType = concatOp.getResult()->getType().cast<ShapedType>();
    auto loc = concatOp.getLoc();
    std::vector<Value> dimPieces;
    auto dst =
        rewriter.create<IREEInterp::HL::AllocHeapOp>(loc, finalType, dimPieces);

    llvm::SmallVector<int64_t, 4> zeroOffset(finalType.getRank(), 0);
    auto srcIndices = createArrayConstant(rewriter, loc, zeroOffset);

    auto concatDimension = concatOp.dimension().getZExtValue();
    llvm::SmallVector<int64_t, 4> dstIndices(finalType.getRank(), 0);
    for (auto src : concatOp.srcs()) {
      auto srcShape = src->getType().cast<ShapedType>().getShape();
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
