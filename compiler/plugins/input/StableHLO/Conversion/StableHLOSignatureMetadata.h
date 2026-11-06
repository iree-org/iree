// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOSIGNATUREMETADATA_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOSIGNATUREMETADATA_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
class TypeConverter;

namespace iree_compiler::stablehlo {

/// Per-value metadata and optional result-to-operand ties on a function or
/// call.
struct SignatureMetadata {
  ArrayAttr argAttrs;
  ArrayAttr resultAttrs;
  ArrayAttr tiedOperands;
};

/// Remaps metadata through the converter's signature expansion without changing
/// IR. Attributes on tuple or one-to-many slots, and references to those slots,
/// have no scalar-slot contract and are diagnosed. Untied results expand to
/// untied parts; attributes on other slots and their indexed references follow
/// the original value. Null arrays remain absent.
FailureOr<SignatureMetadata>
remapSignatureMetadata(Operation *op, FunctionType type,
                       SignatureMetadata metadata,
                       const TypeConverter &converter);

} // namespace iree_compiler::stablehlo
} // namespace mlir

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOSIGNATUREMETADATA_H_
