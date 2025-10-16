// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include <numeric>
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// AllocOps
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  if (getDynamicSizes().size() != getResultType().getNumDynamicDims()) {
    return emitOpError(
        "dimension operand count does not equal sref dynamic dimension count");
  }

  // TODO: Restrict legal parents for this op.
  return success();
}

//===----------------------------------------------------------------------===//
// StructuralOps
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult verifyParallelBodyOp(OpTy op, int64_t numLeadingArgs,
                                          int64_t numIndexBodyArgs,
                                          ArrayRef<BlockArgument> indexArgs) {
  // Verify tied/token array lengths.
  ArrayRef<bool> isTied = op.getIsTied();
  int64_t numResults = op.getNumResults();
  if (isTied.size() != numResults) {
    return op.emitOpError(
               "`is_tied` mask length expected to match number of results ")
           << numResults;
  }

  int64_t numInits =
      std::accumulate(isTied.begin(), isTied.end(), (int64_t)(0));
  if (op.getInits().size() != numInits) {
    return op.emitOpError("number of inits ")
           << op.getInits().size()
           << " does not match the number of results marked as tied "
           << numInits;
  }

  if (op.getRegion().getArguments().size() !=
      numLeadingArgs + numResults + numIndexBodyArgs) {
    return op.emitOpError("expected region to have |numLeadingArgs| + "
                          "|numIndexArgs| + |numResults| "
                          "total arguments");
  }

  for (BlockArgument countArg : indexArgs) {
    if (!countArg.getType().isIndex()) {
      return op.emitOpError(
          "expected index type for thread count/id region arguments");
    }
  }

  PCF::ScopeAttr scope = op.getScope();
  int64_t currIsTiedIndex = 0;
  int64_t currResultIndex = 0;
  for (auto [resultType, refArg, isTied] : llvm::zip_equal(
           op.getResultTypes(), op.getRegionRefArgs(), op.getIsTied())) {
    auto srefType = dyn_cast<PCF::ShapedRefType>(refArg.getType());
    if (!srefType || srefType.getScope() != scope) {
      return op.emitOpError(
                 "expected region ref argument to be of type !pcf.sref "
                 "with scope ")
             << scope;
    }
    if (!srefType.isParentScopeOnlySync() && srefType.getSyncScope()) {
      return op.emitOpError(
          "expected region ref argument to have none or parent sync scope");
    }

    // Traits guarantee this cast to be valid.
    auto shapedResultType = cast<ShapedType>(resultType);
    if (shapedResultType.getShape() != srefType.getShape()) {
      return op.emitOpError("region arg at index ")
             << currResultIndex << " with type " << srefType
             << " shape mismatch with tied result of type " << resultType;
    }

    if (shapedResultType.getElementType() != srefType.getElementType()) {
      return op.emitOpError("region arg at index ")
             << currResultIndex << " element type mismatch of "
             << srefType.getElementType() << " vs "
             << shapedResultType.getElementType();
    }

    if (isTied) {
      Value init = op.getInits()[currIsTiedIndex];
      if (init.getType() != resultType) {
        return op.emitOpError("tied init at index ")
               << currIsTiedIndex << " does not match the type " << resultType
               << " at result index " << currResultIndex;
      }
      ++currIsTiedIndex;
    }
    ++currResultIndex;
  }
  return success();
}

static ParseResult parseParallelExecutionBody(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inits,
    SmallVectorImpl<Type> &initTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizes,
    SmallVectorImpl<Type> &resultTypes, SmallVectorImpl<bool> &isTied,
    Region &body, int64_t &numLeadingArgs, bool parseOptionalLeadingArgs) {
  SmallVector<OpAsmParser::Argument> regionLeadingArgs;
  if (parseOptionalLeadingArgs) {
    if (succeeded(parser.parseOptionalArrow())) {
      if (failed(parser.parseArgumentList(regionLeadingArgs,
                                          OpAsmParser::Delimiter::Paren,
                                          /*allowType=*/true))) {
        return failure();
      }
    }
    numLeadingArgs = regionLeadingArgs.size();
  }
  if (failed(parser.parseKeyword("execute")))
    return failure();
  SmallVector<OpAsmParser::Argument> regionRefArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      // Reserve entries in the lists.
      regionRefArgs.emplace_back();
      if (failed(parser.parseArgument(regionRefArgs.back(),
                                      /*allowType=*/false,
                                      /*allowAttrs=*/true))) {
        return failure();
      }

      // Parse the tied init if present.
      if (succeeded(parser.parseOptionalEqual())) {
        inits.emplace_back();
        if (failed(parser.parseOperand(inits.back()))) {
          return failure();
        }
        isTied.push_back(true);
      } else {
        isTied.push_back(false);
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  SmallVector<OpAsmParser::Argument> indexArgs;
  if (failed(parser.parseLSquare())) {
    return failure();
  }

  if (failed(parser.parseArgumentList(
          indexArgs, /*delimiter=*/OpAsmParser::Delimiter::None,
          /*allowType=*/true, /*allowAttrs=*/true))) {
    return failure();
  }

  if (failed(parser.parseRSquare())) {
    return failure();
  }

  // If there is at least one region arg the arg types and op result types need
  // to be parsed.
  if (!regionRefArgs.empty()) {
    if (failed(parser.parseColon()) || failed(parser.parseLParen())) {
      return failure();
    }

    // Parse all types except the last followed by commas.
    for (OpAsmParser::Argument &arg :
         MutableArrayRef<OpAsmParser::Argument>(regionRefArgs.begin(),
                                                regionRefArgs.end())
             .drop_back()) {
      if (failed(parser.parseType(arg.type)) || failed(parser.parseComma())) {
        return failure();
      }
    }

    // Parse the last type.
    if (failed(parser.parseType(regionRefArgs.back().type))) {
      return failure();
    }

    if (failed(parser.parseRParen()) || failed(parser.parseArrow()) ||
        failed(parser.parseLParen())) {
      return failure();
    }

    int64_t numResults = isTied.size();
    resultTypes.resize(numResults);
    for (auto [i, isTied] : llvm::enumerate(isTied)) {
      if (failed(parser.parseType(resultTypes[i]))) {
        return failure();
      }

      auto shapedType = dyn_cast<ShapedType>(resultTypes[i]);
      if (!shapedType) {
        return failure();
      }

      if (isTied) {
        initTypes.push_back(resultTypes[i]);
      } else if (succeeded(parser.parseOptionalLBrace())) {
        // Only parse dynamic dims for non-tied operands.
        SmallVector<OpAsmParser::UnresolvedOperand> dims;
        if (failed(parser.parseOperandList(dims))) {
          return failure();
        }
        size_t numDynamicDims = shapedType.getNumDynamicDims();
        if (dims.size() != numDynamicDims) {
          return failure();
        }
        if (failed(parser.parseRBrace())) {
          return failure();
        }
        dynamicSizes.append(dims);
      }

      if (i < numResults - 1 && failed(parser.parseComma())) {
        return failure();
      }
    }

    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  // The stored argument order is:
  // (initialized vals) (result tied refs) (num threads).
  SmallVector<OpAsmParser::Argument> args;
  args.append(regionLeadingArgs);
  args.append(regionRefArgs);
  args.append(indexArgs);
  return parser.parseRegion(body, args, /*enableNameShadowing=*/false);
}

static ParseResult parseParallelExecutionBody(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inits,
    SmallVectorImpl<Type> &initTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizes,
    SmallVectorImpl<Type> &resultTypes, SmallVectorImpl<bool> &isTied,
    Region &body) {
  int64_t numLeadingArgs = 0;
  return parseParallelExecutionBody(parser, inits, initTypes, dynamicSizes,
                                    resultTypes, isTied, body, numLeadingArgs,
                                    false);
}

static void printParallelExecutionBody(
    OpAsmPrinter &p, Operation *op, OperandRange inits, TypeRange initTypes,
    OperandRange dynamicSizes, TypeRange resultTypes, ArrayRef<bool> isTied,
    Region &body, int64_t numLeadingArgs, bool printOptionalLeadingArgs) {
  if (printOptionalLeadingArgs && numLeadingArgs > 0) {
    p << "-> (";
    MutableArrayRef<BlockArgument> leadingArgRange =
        body.getArguments().take_front(numLeadingArgs);
    llvm::interleaveComma(leadingArgRange, p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ")";
  }

  p.printNewline();
  p << "  execute";

  int64_t numResults = resultTypes.size();
  int64_t numIndexArgs = body.getNumArguments() - numResults - numLeadingArgs;
  MutableArrayRef<BlockArgument> threadCountArgRange =
      body.getArguments().take_back(numIndexArgs);
  MutableArrayRef<BlockArgument> refArgRange =
      body.getArguments().drop_back(numIndexArgs).take_back(numResults);

  if (numResults != 0) {
    p << "(";
    int64_t currInitIndex = 0;
    for (int64_t i = 0, e = numResults; i < e; ++i) {
      p << refArgRange[i];
      if (isTied[i]) {
        p << " = ";
        p << inits[currInitIndex];
        ++currInitIndex;
      }
      if (i < numResults - 1) {
        p << ", ";
      }
    }
    p << ")";
  }
  p << "[";
  llvm::interleaveComma(threadCountArgRange, p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << "]";

  // Now print the function type.
  if (numResults != 0) {
    p.printNewline();
    // Whitespace to line up parentheses.
    //   |--execute(
    //   |--_____: (
    p << "       : (";
    llvm::interleaveComma(refArgRange, p,
                          [&](BlockArgument arg) { p << arg.getType(); });
    p << ")";
    p.printNewline();
    //   |--execute(
    //   |--____-> (
    p << "      -> (";
    OperandRange currSizes = dynamicSizes;
    for (int64_t i = 0, e = numResults; i < e; ++i) {
      ShapedType resultType = cast<ShapedType>(resultTypes[i]);
      bool isResultTied = isTied[i];
      p << resultType;
      if (!isResultTied && !resultType.hasStaticShape()) {
        int64_t numDynamicDims = resultType.getNumDynamicDims();
        p << "{";
        llvm::interleaveComma(currSizes.take_front(numDynamicDims), p,
                              [&](Value dim) { p << dim; });
        currSizes = currSizes.drop_front(numDynamicDims);
        p << "}";
      }
      if (i < numResults - 1) {
        p << ", ";
      }
    }
    p << ") ";
  } else {
    // Print a space before the region brace if there are no loop results.
    p << " ";
  }
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static void printParallelExecutionBody(OpAsmPrinter &p, Operation *op,
                                       OperandRange inits, TypeRange initTypes,
                                       OperandRange dynamicSizes,
                                       TypeRange resultTypes,
                                       ArrayRef<bool> isTied, Region &body) {
  return printParallelExecutionBody(p, op, inits, initTypes, dynamicSizes,
                                    resultTypes, isTied, body, 0, false);
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

static ParseResult parseInferNumIndexArgs(OpAsmParser &parser, Region &body,
                                          int64_t &numLeadingArgs,
                                          int64_t &numIndexArgs) {
  numIndexArgs = 0;
  for (BlockArgument bbArg :
       llvm::reverse(body.getArguments().drop_front(numLeadingArgs))) {
    if (!bbArg.getType().isIndex()) {
      return success();
    }
    ++numIndexArgs;
  }
  return success();
}

static void printInferNumIndexArgs(OpAsmPrinter &, Operation *, Region &,
                                   int64_t &, int64_t) {
  // Nothing to do. The number of count args gets parsed solely from the region.
}

void GenericOp::getAsmBlockArgumentNames(Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  if (&region == &getInitializer()) {
    return;
  }

  assert(&region == &getRegion() && "Unexpected region");
  for (Value v : getIdArgs()) {
    setNameFn(v, "id");
  }
  for (Value v : getCountArgs()) {
    setNameFn(v, "count");
  }
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
}

LogicalResult GenericOp::verify() {
  if (getNumIndexArgs() % 2 != 0) {
    return emitOpError("expected even number of id + count args");
  }
  if (getRegion().front().getNumArguments() < getNumIndexArgs()) {
    return emitOpError(
        "fewer body arguments than specified number of counts/ids.");
  }
  return verifyParallelBodyOp(*this, getNumLeadingArgs(), getNumIndexArgs(),
                              getIdAndCountArgs());
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      ScopeAttr scope, int64_t numIterators,
                      bool syncOnReturn) {
  GenericOp::build(b, result, TypeRange(), scope, ArrayRef<Value>{},
                   ArrayRef<Value>{}, ArrayRef<bool>{}, numIterators,
                   syncOnReturn);
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      ScopeAttr scope, ValueRange inits, int64_t numIterators,
                      bool syncOnReturn) {
  SmallVector<bool> isTied(inits.size(), true);
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(inits, [](Value v) -> Type { return v.getType(); });
  GenericOp::build(b, result, resultTypes, scope, inits, ArrayRef<Value>{},
                   isTied, numIterators, syncOnReturn);
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      TypeRange resultTypes, ScopeAttr scope,
                      ValueRange dynamicSizes, int64_t numIterators,
                      bool syncOnReturn) {
  SmallVector<bool> isTied(resultTypes.size(), false);
  GenericOp::build(b, result, resultTypes, scope, ArrayRef<Value>{},
                   dynamicSizes, isTied, numIterators, syncOnReturn);
}

void GenericOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                      TypeRange resultTypes, ScopeAttr scope, ValueRange inits,
                      ValueRange dynamicSizes, ArrayRef<bool> isTied,
                      int64_t numIterators, bool syncOnReturn) {

  result.addAttribute(GenericOp::getScopeAttrName(result.name), scope);
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  result.addAttribute(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(inits.size()),
                              static_cast<int32_t>(dynamicSizes.size())}));

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);
  inherentAttrs.setNumIndexArgs(2 * numIterators);

  // Add the initializer region.
  result.addRegion();

  // Add the main region.
  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Add block arguments.

  // sref args.
  for (Type resultType : resultTypes) {
    auto shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Thread count/id args.
  Type indexType = b.getIndexType();
  for (int64_t i = 0; i < 2 * numIterators; ++i) {
    entryBlock.addArgument(indexType, result.location);
  }
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  for (Value v : getIdArgs()) {
    setNameFn(v, "id");
  }
  for (Value v : getRegionRefArgs()) {
    setNameFn(v, "ref");
  }
}

LogicalResult LoopOp::verify() {
  if (getCount().empty()) {
    return emitOpError("expected at least one iteration count argument");
  }
  if (getBody()->getNumArguments() < getNumIdArgs()) {
    return emitOpError("fewer body arguments than specified number of ids.");
  }
  return verifyParallelBodyOp(*this, /*numLeadingArgs=*/0, getNumIdArgs(),
                              getIdArgs());
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   ScopeAttr scope, ValueRange count, bool syncOnReturn) {
  LoopOp::build(b, result, TypeRange(), scope, count, ArrayRef<Value>{},
                ArrayRef<Value>{}, ArrayRef<bool>{}, syncOnReturn);
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   ScopeAttr scope, ValueRange count, ValueRange inits,
                   bool syncOnReturn) {
  SmallVector<bool> isTied(inits.size(), true);
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(inits, [](Value v) -> Type { return v.getType(); });
  LoopOp::build(b, result, resultTypes, scope, count, inits, ArrayRef<Value>{},
                isTied, syncOnReturn);
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   TypeRange resultTypes, ScopeAttr scope, ValueRange count,
                   ValueRange dynamicSizes, bool syncOnReturn) {
  SmallVector<bool> isTied(resultTypes.size(), false);
  LoopOp::build(b, result, resultTypes, scope, count, ArrayRef<Value>{},
                dynamicSizes, isTied, syncOnReturn);
}

void LoopOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                   TypeRange resultTypes, ScopeAttr scope, ValueRange count,
                   ValueRange inits, ValueRange dynamicSizes,
                   ArrayRef<bool> isTied, bool syncOnReturn) {

  result.addAttribute(LoopOp::getScopeAttrName(result.name), scope);
  result.addOperands(count);
  result.addOperands(inits);
  result.addOperands(dynamicSizes);
  result.addTypes(resultTypes);

  result.addAttribute(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(count.size()),
                              static_cast<int32_t>(inits.size()),
                              static_cast<int32_t>(dynamicSizes.size())}));

  Properties &inherentAttrs = result.getOrAddProperties<Properties>();
  inherentAttrs.setIsTied(isTied);
  inherentAttrs.setSyncOnReturn(syncOnReturn);

  Region *region = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(region);
  Block &entryBlock = region->front();

  // Add block arguments.

  // sref args.
  for (Type resultType : resultTypes) {
    auto shapedType = cast<ShapedType>(resultType);
    entryBlock.addArgument(
        PCF::ShapedRefType::get(b.getContext(), shapedType.getShape(),
                                shapedType.getElementType(), scope),
        result.location);
  }

  // Thread count args.
  Type indexType = b.getIndexType();
  int64_t numCountArgs = count.size() == 0 ? 1 : count.size();
  for (int64_t i = 0; i < numCountArgs; ++i) {
    entryBlock.addArgument(indexType, result.location);
  }
}

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the GenericOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

//===----------------------------------------------------------------------===//
// Control Flow Ops
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BranchCondReturnOp
//===----------------------------------------------------------------------===//

void BranchCondReturnOp::setDest(Block *block) { return setSuccessor(block); }

void BranchCondReturnOp::eraseOperand(unsigned index) {
  (*this)->eraseOperand(index);
}

SuccessorOperands BranchCondReturnOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  // Single index operand produced by this op.
  return SuccessorOperands(getDestOperandsMutable());
}

Block *
BranchCondReturnOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr =
          llvm::dyn_cast_or_null<IntegerAttr>(operands.front())) {
    return condAttr.getValue().isOne() ? nullptr : getDest();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// WriteOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

// Build a WriteSliceOp with mixed static and dynamic entries.
void WriteSliceOp::build(OpBuilder &b, OperationState &result, Value source,
                         Value dest, ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         ArrayRef<OpFoldResult> strides,
                         ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  result.addAttributes(attrs);
  build(b, result, {}, source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
}

/// Build an WriteSliceOp with mixed static and dynamic entries
/// packed into a Range vector.
void WriteSliceOp::build(OpBuilder &b, OperationState &result, Value source,
                         Value dest, ArrayRef<Range> ranges,
                         ArrayRef<NamedAttribute> attrs) {
  auto [offsets, sizes, strides] = getOffsetsSizesAndStrides(ranges);
  build(b, result, source, dest, offsets, sizes, strides, attrs);
}

// Build a WriteSliceOp with dynamic entries.
void WriteSliceOp::build(OpBuilder &b, OperationState &result, Value source,
                         Value dest, ValueRange offsets, ValueRange sizes,
                         ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerOperations() {
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF

//===----------------------------------------------------------------------===//
// TableGen definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.cpp.inc" // IWYU pragma: keep
