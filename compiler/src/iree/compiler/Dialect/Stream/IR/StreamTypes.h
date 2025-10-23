// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_IR_STREAMTYPES_H_
#define IREE_COMPILER_DIALECT_STREAM_IR_STREAMTYPES_H_

#include <optional>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/IntegerSet.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/Stream/IR/StreamEnums.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::Stream {
class AffinityAttr;
} // namespace mlir::iree_compiler::IREE::Stream

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamAttrs.h.inc" // IWYU pragma: keep
// clang-format on

#include "iree/compiler/Dialect/Stream/IR/StreamAttrInterfaces.h.inc" // IWYU pragma: export

#include "iree/compiler/Dialect/Stream/IR/StreamTypeInterfaces.h.inc" // IWYU pragma: export

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

struct AsyncAccessRange {
  ResourceAccessBitfield access;
  Value resource;
  Value start; // may be nullptr to indicate 0
  Value end;
  Value length;

  // Returns true if the access is read-only.
  bool isReadOnly() const { return access == ResourceAccessBitfield::Read; }

  // Prints a textual representation of the range.
  void print(llvm::raw_ostream &os, AsmState &asmState);

  // Returns true if |lhs| and |rhs| may overlap and false only if it can be
  // locally proven that they do not.
  static bool mayOverlap(const AsyncAccessRange &lhs,
                         const AsyncAccessRange &rhs);
};

// Joins all of |timepoints| with a stream.timepoint.join, representing the
// timepoint when all timepoints have been reached (AND semantics).
// Returns nullptr if no timepoints were provided and identity if only one was.
Value joinTimepoints(Location loc, ValueRange timepoints, OpBuilder &builder);

} // namespace mlir::iree_compiler::IREE::Stream

#include "iree/compiler/Dialect/Stream/IR/StreamOpInterfaces.h.inc" // IWYU pragma: export

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// custom<ParameterReference>($scope, $key)
//===----------------------------------------------------------------------===//

ParseResult parseParameterReference(AsmParser &parser, StringAttr &scopeAttr,
                                    StringAttr &keyAttr);
void printParameterReference(AsmPrinter &p, StringAttr scopeAttr,
                             StringAttr keyAttr);
static inline void printParameterReference(AsmPrinter &p, Operation *op,
                                           StringAttr scopeAttr,
                                           StringAttr keyAttr) {
  printParameterReference(p, scopeAttr, keyAttr);
}

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_IR_STREAMTYPES_H_
