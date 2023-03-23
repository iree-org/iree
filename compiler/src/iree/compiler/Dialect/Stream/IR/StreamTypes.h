// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_IR_STREAMTYPES_H_
#define IREE_COMPILER_DIALECT_STREAM_IR_STREAMTYPES_H_

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
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
#include "iree/compiler/Dialect/Stream/IR/StreamEnums.h.inc"  // IWYU pragma: export
// clang-format on

// It's unfortunate this is required.
namespace mlir {

template <>
struct FieldParser<
    std::optional<mlir::iree_compiler::IREE::Stream::CollectiveReductionOp>> {
  static FailureOr<mlir::iree_compiler::IREE::Stream::CollectiveReductionOp>
  parse(AsmParser &parser) {
    std::string value;
    if (parser.parseKeywordOrString(&value)) return failure();
    auto result = mlir::iree_compiler::IREE::Stream::symbolizeEnum<
        mlir::iree_compiler::IREE::Stream::CollectiveReductionOp>(value);
    if (!result.has_value()) return failure();
    return result.value();
  }
};
static inline AsmPrinter &operator<<(
    AsmPrinter &printer,
    std::optional<mlir::iree_compiler::IREE::Stream::CollectiveReductionOp>
        param) {
  printer << (param.has_value()
                  ? mlir::iree_compiler::IREE::Stream::stringifyEnum(
                        param.value())
                  : StringRef{""});
  return printer;
}

}  // namespace mlir

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamAttrs.h.inc"  // IWYU pragma: keep
// clang-format on

#include "iree/compiler/Dialect/Stream/IR/StreamAttrInterfaces.h.inc"  // IWYU pragma: export

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

#include "iree/compiler/Dialect/Stream/IR/StreamTypeInterfaces.h.inc"  // IWYU pragma: export

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

struct AsyncAccessRange {
  ResourceAccessBitfield access;
  Value resource;
  Value start;  // may be nullptr to indicate 0
  Value end;
  Value length;
};

#include "iree/compiler/Dialect/Stream/IR/StreamOpInterfaces.h.inc"  // IWYU pragma: export

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_STREAM_IR_STREAMTYPES_H_
