// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ParsingUtils.h"

namespace mlir::iree_compiler {

const char *kNonIndexedSymbol = "None";

constexpr int kConstantValue = -1;
constexpr int kIndexedValue = -2;

ParseResult
parseIndexVecs(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indexVecs,
               SmallVectorImpl<Type> &indexVecTypes,
               DenseI64ArrayAttr &indexed) {
  if (parser.parseLSquare())
    return failure();

  SMLoc loc;
  SmallVector<int64_t> indexedArr;
  while (parser.parseOptionalRSquare()) {
    // Check if this dimension is indexed using a constant.
    if (!parser.parseOptionalKeyword(kNonIndexedSymbol)) {
      indexedArr.push_back(kConstantValue);
      (void)parser.parseOptionalComma();
      continue;
    }

    // Check if this dimension is contigious.
    int64_t dim;
    if (parser.parseOptionalInteger(dim).has_value()) {
      indexedArr.push_back(dim);
      (void)parser.parseOptionalComma();
      continue;
    }

    OpAsmParser::UnresolvedOperand indexVec;
    Type indexVecType;
    if (parser.getCurrentLocation(&loc) || parser.parseOperand(indexVec) ||
        parser.parseColon() || parser.parseType(indexVecType)) {
      return parser.emitError(loc, "Expected `none` or `operand : type`");
    }

    indexVecs.push_back(indexVec);
    indexVecTypes.push_back(indexVecType);
    indexedArr.push_back(kIndexedValue);

    (void)parser.parseOptionalComma();
  }

  // OpBuilder is only used as a helper to build an BoolArrayAttr.
  OpBuilder b(parser.getContext());
  indexed = b.getDenseI64ArrayAttr(indexedArr);
  return success();
}

void printIndexVecs(OpAsmPrinter &p, Operation *op, OperandRange indexVecs,
                    TypeRange indexVecTypes, DenseI64ArrayAttr indexed) {
  int64_t rank = indexed.size();
  ArrayRef<int64_t> indexedArr = indexed.asArrayRef();

  int64_t currIndexDim = 0;
  p << "[";
  for (int64_t i : llvm::seq<int64_t>(rank)) {
    if (indexedArr[i] == kConstantValue) {
      p << kNonIndexedSymbol;
    } else if (indexedArr[i] == kIndexedValue) {
      p << indexVecs[currIndexDim] << ": " << indexVecTypes[currIndexDim];
      ++currIndexDim;
    } else {
      p << indexedArr[i];
    }

    if (i != rank - 1) {
      p << ", ";
    }
  }
  p << "]";
}

} // namespace mlir::iree_compiler
