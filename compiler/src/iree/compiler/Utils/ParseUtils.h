// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PARSEUTILS_H_
#define IREE_COMPILER_UTILS_PARSEUTILS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::iree_compiler {

template <typename ParserTy>
static llvm::ParseResult parseBoolAsUnitAttr(ParserTy &parser, bool &b,
                                             llvm::StringRef name) {
  b = succeeded(parser.parseOptionalKeyword(name));
  return success();
}

template <typename PrinterTy>
static void printBoolAsUnitAttr(PrinterTy &printer, bool b,
                                llvm::StringRef name) {
  if (b) {
    printer << name;
  }
}

template <typename PrinterTy, typename UnderlyingTy>
static void printBoolAsUnitAttr(PrinterTy &printer, UnderlyingTy, bool b,
                                llvm::StringRef name) {
  printBoolAsUnitAttr(printer, b, name);
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_PARSEUTILS_H_
