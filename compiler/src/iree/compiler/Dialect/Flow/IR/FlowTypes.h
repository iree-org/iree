// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_
#define IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
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
#include "iree/compiler/Dialect/Flow/IR/FlowEnums.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::Flow {

#include "iree/compiler/Dialect/Flow/IR/FlowOpInterfaces.h.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/Flow/IR/FlowTypeInterfaces.h.inc" // IWYU pragma: export

} // namespace mlir::iree_compiler::IREE::Flow

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowAttrs.h.inc" // IWYU pragma: keep
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Flow {

// Create an attribute corresponding to the underlying numeric element type.
// If there no such correspondence a null attribute is returned.
IREE::Flow::CollectiveElementTypeAttr
getCollectiveElementTypeAttr(RankedTensorType type);

// Convert the numeric type `type` to the corresponding enum value.
// If there is not correspondence nullopt is returned.
std::optional<IREE::Flow::CollectiveElementType>
convertToFlowCollectiveElementType(Type type);

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

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_
