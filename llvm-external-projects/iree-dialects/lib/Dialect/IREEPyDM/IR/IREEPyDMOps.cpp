// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOps.h"

#include "iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::iree_pydm;

using PyCallOp = mlir::iree_pydm::CallOp;
using PyFuncOp = mlir::iree_pydm::FuncOp;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

LogicalResult PyFuncOp::verifyType() {
  // TODO: Enforce arg/result invariants.
  return success();
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(PyFuncOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_like_impl::printFunctionLikeOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

static LogicalResult verify(PyFuncOp op) {
  // TODO: Enforce invariants.
  return success();
}

//===----------------------------------------------------------------------===//
// PatternMatchCallOp
//===----------------------------------------------------------------------===//

LogicalResult PatternMatchCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto verifySymbols = [&](ArrayAttr symbols) -> LogicalResult {
    for (auto symbolAttr : symbols) {
      auto symbol = symbolAttr.cast<FlatSymbolRefAttr>();
      PyFuncOp fn =
          symbolTable.lookupNearestSymbolFrom<PyFuncOp>(*this, symbol);
      if (!fn)
        return emitOpError() << "'" << symbol.getValue()
                             << "' does not reference a valid function";
    }
    return success();
  };
  auto genericsAttr = (*this)->getAttrOfType<ArrayAttr>("generic_match");
  if (!genericsAttr)
    return emitOpError(
        "requires a 'generic_match' array of symbol reference attributes");
  if (failed(verifySymbols(genericsAttr))) return failure();

  auto specificsAttr = (*this)->getAttrOfType<ArrayAttr>("specific_match");
  if (!specificsAttr)
    return emitOpError(
        "requires a 'specific_match' array of symbol reference attributes");
  if (failed(verifySymbols(specificsAttr))) return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult PyCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  PyFuncOp fn = symbolTable.lookupNearestSymbolFrom<PyFuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (getOperand(i).getType() != fnType.getInput(i)) {
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }
  }

  return success();
}

FunctionType PyCallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// DynamicCallOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  Operation *fn = symbolTable.lookupNearestSymbolFrom(*this, fnAttr);
  if (!fn || !isa<PyFuncOp>(fn))
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  return success();
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOps.cpp.inc"
