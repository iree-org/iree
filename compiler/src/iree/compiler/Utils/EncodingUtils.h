// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_ENCODINGUTILS_H_
#define IREE_COMPILER_UTILS_ENCODINGUTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

/// Parse a list of integer values and/or dynamic values ('?')
FailureOr<SmallVector<int64_t>> parseDynamicI64IntegerList(AsmParser &parser);

/// Print a list of integer values and/or dynamic values ('?')
void printDynamicI64IntegerList(AsmPrinter &printer, ArrayRef<int64_t> vals);

/// Parse a list of integer values and/or dynamic values ('?') into an ArrayAttr
ParseResult parseDynamicI64ArrayAttr(AsmParser &parser, ArrayAttr &attr);

/// Print an ArrayAttr of integer values and/or dynamic values ('?')
void printDynamicI64ArrayAttr(AsmPrinter &printer, ArrayAttr attrs);

/// Parse a list of integer values and/or dynamic values ('?') into a
/// DenseI64ArrayAttr
ParseResult parseDynamicI64DenseArrayAttr(AsmParser &parser,
                                          DenseI64ArrayAttr &attr);

/// Print a DenseI64ArrayAttr as a list of integer values and/or dynamic values
/// ('?')
void printDynamicI64DenseArrayAttr(AsmPrinter &printer, DenseI64ArrayAttr attr);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_ENCODINGUTILS_H_
