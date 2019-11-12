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

#include "iree/compiler/IR/Sequencer/HLOps.h"

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/Types.h"
#include "iree/compiler/Utils/OpCreationUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREESeq {
namespace HL {

namespace {

static LogicalResult verifyWorkload(Operation *op, Value *workload) {
  if (auto workloadType = workload->getType().dyn_cast<MemRefType>()) {
    if (workloadType.getNumElements() != 3) {
      return op->emitOpError("workload must be specified as (x,y,z) but has ")
             << workloadType.getNumElements()
             << " elements (type=" << workload->getType() << ")";
    }
    return success();
  }
  return op->emitOpError(
             "workload must be specified as an (x,y,z) memref but has type ")
         << workload->getType();
}

}  // namespace

//===----------------------------------------------------------------------===//
// iree_hl_seq.call
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
  p << "iree_hl_seq.call " << op.getAttr("callee") << '(';
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
// iree_hl_seq.call_indirect
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
  p << "iree_hl_seq.call_indirect ";
  p.printOperand(op.getCallee());
  p << '(';
  auto operandRange = op.getOperands();
  p.printOperands(++operandRange.begin(), operandRange.end());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : " << op.getCallee()->getType();
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.return
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
  p << "iree_hl_seq.return";
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.operand_begin(), op.operand_end());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.br
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState &result) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser.parseSuccessorAndUseList(dest, destOperands)) return failure();
  result.addSuccessor(dest, destOperands);
  return success();
}

static void printBranchOp(OpAsmPrinter &p, BranchOp op) {
  p << "iree_hl_seq.br ";
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
// iree_hl_seq.cond_br
//===----------------------------------------------------------------------===//

static ParseResult parseCondBranchOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Value *, 4> destOperands;
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
  p << "iree_hl_seq.cond_br ";
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.dispatch
//===----------------------------------------------------------------------===//

static ParseResult parseDispatchOp(OpAsmParser &parser, OperationState &state) {
  auto executableLoc = parser.getNameLoc();

  FlatSymbolRefAttr executableAttr;
  FlatSymbolRefAttr entryPointAttr;
  FunctionType entryPointType;
  if (failed(parser.parseAttribute(executableAttr, "executable",
                                   state.attributes)) ||
      failed(parser.parseColon()) || failed(parser.parseColon()) ||
      failed(parser.parseAttribute(entryPointAttr, "entry_point",
                                   state.attributes))) {
    return failure();
  }

  OpAsmParser::OperandType workloadArg;
  Type workloadArgType;
  if (failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(workloadArg)) ||
      failed(parser.parseColonType(workloadArgType)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperand(workloadArg, workloadArgType,
                                   state.operands))) {
    return failure();
  }

  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (failed(
          parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) ||
      failed(parser.parseOptionalAttrDict(state.attributes)) ||
      failed(parser.parseColonType(entryPointType)) ||
      failed(parser.addTypesToList(entryPointType.getResults(), state.types)) ||
      failed(parser.resolveOperands(operands, entryPointType.getInputs(),
                                    executableLoc, state.operands))) {
    return failure();
  }
  return success();
}

static void printDispatchOp(OpAsmPrinter &p, DispatchOp op) {
  p << "iree_hl_seq.dispatch " << op.getExecutable()
    << "::" << op.getEntryPoint();
  p << "[";
  p.printOperand(op.getWorkload());
  p << " : ";
  p.printType(op.getWorkload()->getType());
  p << "](";
  p.printOperands(op.getArgOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "executable",
                              "entry_point",
                          });
  p << " : ";
  p.printType(op.getEntryPointType());
}

static LogicalResult verifyDispatchOp(DispatchOp op) {
  if (failed(verifyWorkload(op, op.getWorkload()))) {
    return failure();
  }
  return success();
}

FunctionType DispatchOp::getEntryPointType() {
  SmallVector<Type, 4> resultTypes(getResultTypes());
  SmallVector<Type, 8> argTypes(getArgOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.rank
//===----------------------------------------------------------------------===//

OpFoldResult RankOp::fold(ArrayRef<Attribute> operands) {
  Builder builder(getContext());
  if (auto op0 = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    return builder.getIntegerAttr(builder.getIntegerType(32),
                                  op0.getType().getRank());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.shape
//===----------------------------------------------------------------------===//

void ShapeOp::build(Builder *builder, OperationState &state, Value *operand) {
  state.addOperands(operand);
  int64_t rank = 0;
  if (auto shapedType = operand->getType().dyn_cast<ShapedType>()) {
    rank = shapedType.getRank();
  }
  state.addTypes(MemRefType::get({rank}, builder->getIntegerType(32)));
}

OpFoldResult ShapeOp::fold(ArrayRef<Attribute> operands) {
  Builder builder(getContext());
  if (auto op0 = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    return DenseIntElementsAttr::get(
        RankedTensorType::get({op0.getType().getRank()},
                              builder.getIntegerType(32)),
        op0.getType().getShape());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.length
//===----------------------------------------------------------------------===//

OpFoldResult LengthOp::fold(ArrayRef<Attribute> operands) {
  Builder builder(getContext());
  if (auto op0 = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    return builder.getIntegerAttr(builder.getIntegerType(32),
                                  op0.getNumElements());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// iree_hl_seq.concat
//===----------------------------------------------------------------------===//

namespace {
struct ConcatToCopies : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(ConcatOp concatOp,
                                     PatternRewriter &rewriter) const override {
    auto finalType = concatOp.getResult()->getType().cast<ShapedType>();
    auto loc = concatOp.getLoc();
    std::vector<Value *> dimPieces;
    auto dst =
        rewriter.create<IREESeq::HL::AllocHeapOp>(loc, finalType, dimPieces);

    llvm::SmallVector<int64_t, 4> zeroOffset(finalType.getRank(), 0);
    auto srcIndices = createArrayConstant(rewriter, loc, zeroOffset);

    auto concatDimension = concatOp.dimension().getZExtValue();
    llvm::SmallVector<int64_t, 4> dstIndices(finalType.getRank(), 0);
    for (auto *src : concatOp.srcs()) {
      auto srcShape = src->getType().cast<ShapedType>().getShape();
      auto lengths = createArrayConstant(rewriter, loc, srcShape);
      auto dstIndicesOp = createArrayConstant(rewriter, loc, dstIndices);
      rewriter.create<IREESeq::HL::CopyOp>(loc, src, srcIndices, dst,
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
#include "iree/compiler/IR/Sequencer/HLOps.cpp.inc"

}  // namespace HL
}  // namespace IREESeq
}  // namespace iree_compiler
}  // namespace mlir
