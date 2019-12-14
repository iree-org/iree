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

#include "iree/compiler/IR/Sequencer/LLOps.h"

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREESeq {
namespace LL {

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

static LogicalResult verifyWorkload(Operation *op, ElementsAttr workload) {
  if (workload.getNumElements() != 3) {
    return op->emitOpError("workload must be specified as (x,y,z) but has ")
           << workload.getNumElements() << " elements (value=" << workload
           << ")";
  }
  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// iree_ll_seq.constant
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.call
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
  p << "iree_ll_seq.call " << op.getAttr("callee") << '(';
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
// iree_ll_seq.call_import
//===----------------------------------------------------------------------===//

static ParseResult parseCallImportOp(OpAsmParser &parser,
                                     OperationState &state) {
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

static void printCallImportOp(OpAsmPrinter &p, CallImportOp op) {
  p << "iree_ll_seq.call_import " << op.getAttr("callee") << '(';
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : ";
  p.printType(op.getCalleeType());
}

FunctionType CallImportOp::getCalleeType() {
  SmallVector<Type, 4> resultTypes(getResultTypes());
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.call_indirect
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
  p << "iree_ll_seq.call_indirect ";
  p.printOperand(op.getCallee());
  p << '(';
  auto operandRange = op.getOperands();
  p.printOperands(++operandRange.begin(), operandRange.end());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : " << op.getCallee()->getType();
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.return
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
  p << "iree_ll_seq.return";
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.operand_begin(), op.operand_end());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.br
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState &result) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser.parseSuccessorAndUseList(dest, destOperands)) return failure();
  result.addSuccessor(dest, destOperands);
  return success();
}

static void printBranchOp(OpAsmPrinter &p, BranchOp op) {
  p << "iree_ll_seq.br ";
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
// iree_ll_seq.cond_br
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
  p << "iree_ll_interp.cond_br ";
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.dynamic_dispatch
//===----------------------------------------------------------------------===//

static ParseResult parseDynamicDispatchOp(OpAsmParser &parser,
                                          OperationState &state) {
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

static void printDynamicDispatchOp(OpAsmPrinter &p, DynamicDispatchOp op) {
  p << "iree_ll_seq.dynamic_dispatch " << op.getExecutable()
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

static LogicalResult verifyDynamicDispatchOp(DynamicDispatchOp op) {
  if (failed(verifyWorkload(op, op.getWorkload()))) {
    return failure();
  }
  return success();
}

FunctionType DynamicDispatchOp::getEntryPointType() {
  SmallVector<Type, 4> resultTypes(getResultTypes());
  SmallVector<Type, 8> argTypes(getArgOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

namespace {
struct MakeDynamicDispatchOpStatic
    : public OpRewritePattern<DynamicDispatchOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(DynamicDispatchOp dynamicDispatchOp,
                                     PatternRewriter &rewriter) const override {
    ElementsAttr workloadAttr;
    if (!matchPattern(dynamicDispatchOp.getWorkload(),
                      m_Constant(&workloadAttr))) {
      return matchFailure();
    }

    SmallVector<Type, 8> resultTypes{dynamicDispatchOp.getResultTypes()};
    rewriter.replaceOpWithNewOp<IREESeq::LL::StaticDispatchOp>(
        dynamicDispatchOp, dynamicDispatchOp.getExecutable(),
        dynamicDispatchOp.getEntryPoint(), workloadAttr, resultTypes,
        dynamicDispatchOp.getArgOperands());
    return matchSuccess();
  }
};
}  // namespace

void DynamicDispatchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MakeDynamicDispatchOpStatic>(context);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.static_dispatch
//===----------------------------------------------------------------------===//

static ParseResult parseStaticDispatchOp(OpAsmParser &parser,
                                         OperationState &state) {
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

  ElementsAttr workloadAttr;
  if (failed(parser.parseLSquare()) ||
      failed(
          parser.parseAttribute(workloadAttr, "workload", state.attributes)) ||
      failed(parser.parseRSquare())) {
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

static void printStaticDispatchOp(OpAsmPrinter &p, StaticDispatchOp op) {
  p << "iree_ll_seq.static_dispatch " << op.getExecutable()
    << "::" << op.getEntryPoint();
  p << "[";
  p.printAttribute(op.getWorkload());
  p << "](";
  p.printOperands(op.getArgOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "executable",
                              "entry_point",
                              "workload",
                          });
  p << " : ";
  p.printType(op.getEntryPointType());
}

static LogicalResult verifyStaticDispatchOp(StaticDispatchOp op) {
  if (failed(verifyWorkload(op, op.getWorkload()))) {
    return failure();
  }
  return success();
}

FunctionType StaticDispatchOp::getEntryPointType() {
  SmallVector<Type, 4> resultTypes(getResultTypes());
  SmallVector<Type, 8> argTypes(getArgOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.shape
//===----------------------------------------------------------------------===//

namespace {
struct FoldShapeOp : public OpRewritePattern<ShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(ShapeOp shapeOp,
                                     PatternRewriter &rewriter) const override {
    auto memRefType = shapeOp.input()->getType().cast<MemRefType>();
    if (memRefType.hasStaticShape()) {
      auto constantOp = rewriter.create<IREESeq::LL::ConstantOp>(
          shapeOp.getLoc(),
          MemRefType::get({memRefType.getRank()}, rewriter.getIntegerType(64)),
          DenseIntElementsAttr::get(
              RankedTensorType::get({memRefType.getRank()},
                                    rewriter.getIntegerType(64)),
              memRefType.getShape()));
      replaceSubsequentUses(shapeOp, shapeOp.dst(), constantOp.getResult());
      rewriter.eraseOp(shapeOp);
      return matchSuccess();
    }
    return matchFailure();
  }
};
}  // namespace

void ShapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<FoldShapeOp>(context);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.length
//===----------------------------------------------------------------------===//

namespace {
struct FoldLengthOp : public OpRewritePattern<LengthOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(LengthOp lengthOp,
                                     PatternRewriter &rewriter) const override {
    auto memRefType = lengthOp.input()->getType().cast<MemRefType>();
    if (memRefType.hasStaticShape()) {
      auto constantOp = rewriter.create<IREESeq::LL::ConstantOp>(
          lengthOp.getLoc(), MemRefType::get({}, rewriter.getIntegerType(64)),
          DenseIntElementsAttr::get(
              RankedTensorType::get({}, rewriter.getIntegerType(64)),
              {memRefType.getNumElements()}));
      replaceSubsequentUses(lengthOp, lengthOp.dst(), constantOp.getResult());
      rewriter.eraseOp(lengthOp);
      return matchSuccess();
    }
    return matchFailure();
  }
};
}  // namespace

void LengthOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<FoldLengthOp>(context);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.compute_offset
//===----------------------------------------------------------------------===//

namespace {
struct FoldComputeOffsetOp : public OpRewritePattern<ComputeOffsetOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(ComputeOffsetOp computeOffsetOp,
                                     PatternRewriter &rewriter) const override {
    ElementsAttr shapeAttr;
    ElementsAttr indicesAttr;
    if (!matchPattern(computeOffsetOp.shape(), m_Constant(&shapeAttr)) ||
        !matchPattern(computeOffsetOp.indices(), m_Constant(&indicesAttr))) {
      return matchFailure();
    }

    int64_t offset = 0;
    for (unsigned i = 0; i < indicesAttr.getNumElements(); ++i) {
      int64_t axisOffset =
          indicesAttr.getValue({i}).cast<IntegerAttr>().getInt();
      for (unsigned j = i + 1; j < shapeAttr.getNumElements(); ++j) {
        axisOffset *= shapeAttr.getValue({j}).cast<IntegerAttr>().getInt();
      }
      offset += axisOffset;
    }
    offset *= computeOffsetOp.elementSize().getZExtValue();

    auto constantOp = rewriter.create<IREESeq::LL::ConstantOp>(
        computeOffsetOp.getLoc(),
        MemRefType::get({}, rewriter.getIntegerType(64)),
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, rewriter.getIntegerType(64)), {offset}));
    replaceSubsequentUses(computeOffsetOp, computeOffsetOp.dst(),
                          constantOp.getResult());
    rewriter.eraseOp(computeOffsetOp);
    return matchSuccess();
  }
};
}  // namespace

void ComputeOffsetOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldComputeOffsetOp>(context);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.compute_range
//===----------------------------------------------------------------------===//

namespace {
struct FoldComputeRangeOp : public OpRewritePattern<ComputeRangeOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(ComputeRangeOp computeRangeOp,
                                     PatternRewriter &rewriter) const override {
    ElementsAttr shapeAttr;
    ElementsAttr indicesAttr;
    ElementsAttr lengthsAttr;
    if (!matchPattern(computeRangeOp.shape(), m_Constant(&shapeAttr)) ||
        !matchPattern(computeRangeOp.indices(), m_Constant(&indicesAttr)) ||
        !matchPattern(computeRangeOp.lengths(), m_Constant(&lengthsAttr))) {
      return matchFailure();
    }

    int64_t offset = 0;
    int64_t length = computeRangeOp.elementSize().getZExtValue();
    for (unsigned i = 0; i < indicesAttr.getNumElements(); ++i) {
      int64_t axisOffset =
          indicesAttr.getValue({i}).cast<IntegerAttr>().getInt();
      for (unsigned j = i + 1; j < shapeAttr.getNumElements(); ++j) {
        axisOffset *= shapeAttr.getValue({j}).cast<IntegerAttr>().getInt();
      }
      offset += axisOffset;
      length *= lengthsAttr.getValue({i}).cast<IntegerAttr>().getInt();
    }
    offset *= computeRangeOp.elementSize().getZExtValue();

    auto offsetConstantOp = rewriter.create<IREESeq::LL::ConstantOp>(
        computeRangeOp.getLoc(),
        MemRefType::get({}, rewriter.getIntegerType(64)),
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, rewriter.getIntegerType(64)), {offset}));
    replaceSubsequentUses(computeRangeOp, computeRangeOp.dstOffset(),
                          offsetConstantOp.getResult());
    auto lengthConstantOp = rewriter.create<IREESeq::LL::ConstantOp>(
        computeRangeOp.getLoc(),
        MemRefType::get({}, rewriter.getIntegerType(64)),
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, rewriter.getIntegerType(64)), {length}));
    replaceSubsequentUses(computeRangeOp, computeRangeOp.dstLength(),
                          lengthConstantOp.getResult());
    rewriter.eraseOp(computeRangeOp);
    return matchSuccess();
  }
};
}  // namespace

void ComputeRangeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldComputeRangeOp>(context);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.dynamic_copy
//===----------------------------------------------------------------------===//

namespace {
struct MakeDynamicCopyOpStatic : public OpRewritePattern<DynamicCopyOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(DynamicCopyOp dynamicCopyOp,
                                     PatternRewriter &rewriter) const override {
    ElementsAttr srcOffsetAttr;
    ElementsAttr dstOffsetAttr;
    ElementsAttr lengthAttr;
    if (!matchPattern(dynamicCopyOp.srcOffset(), m_Constant(&srcOffsetAttr)) ||
        !matchPattern(dynamicCopyOp.dstOffset(), m_Constant(&dstOffsetAttr)) ||
        !matchPattern(dynamicCopyOp.length(), m_Constant(&lengthAttr))) {
      return matchFailure();
    }

    rewriter.replaceOpWithNewOp<IREESeq::LL::StaticCopyOp>(
        dynamicCopyOp, dynamicCopyOp.src(),
        srcOffsetAttr.getValue({}).cast<IntegerAttr>(), dynamicCopyOp.dst(),
        dstOffsetAttr.getValue({}).cast<IntegerAttr>(),
        lengthAttr.getValue({}).cast<IntegerAttr>());
    return matchSuccess();
  }
};
}  // namespace

void DynamicCopyOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MakeDynamicCopyOpStatic>(context);
}

//===----------------------------------------------------------------------===//
// iree_ll_seq.dynamic_fill
//===----------------------------------------------------------------------===//

namespace {
struct MakeDynamicFillOpStatic : public OpRewritePattern<DynamicFillOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(DynamicFillOp dynamicFillOp,
                                     PatternRewriter &rewriter) const override {
    ElementsAttr valueAttr;
    ElementsAttr dstOffsetAttr;
    ElementsAttr lengthAttr;
    if (!matchPattern(dynamicFillOp.value(), m_Constant(&valueAttr)) ||
        !matchPattern(dynamicFillOp.dstOffset(), m_Constant(&dstOffsetAttr)) ||
        !matchPattern(dynamicFillOp.length(), m_Constant(&lengthAttr))) {
      return matchFailure();
    }

    rewriter.replaceOpWithNewOp<IREESeq::LL::StaticFillOp>(
        dynamicFillOp, valueAttr.getValue({}).cast<IntegerAttr>(),
        dynamicFillOp.dst(), dstOffsetAttr.getValue({}).cast<IntegerAttr>(),
        lengthAttr.getValue({}).cast<IntegerAttr>());
    return matchSuccess();
  }
};
}  // namespace

void DynamicFillOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<MakeDynamicFillOpStatic>(context);
}

#define GET_OP_CLASSES
#include "iree/compiler/IR/Sequencer/LLOps.cpp.inc"

}  // namespace LL
}  // namespace IREESeq
}  // namespace iree_compiler
}  // namespace mlir
