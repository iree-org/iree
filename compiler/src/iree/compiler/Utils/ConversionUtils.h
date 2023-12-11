// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_CONVERSIONUTILS_H_
#define IREE_COMPILER_UTILS_CONVERSIONUTILS_H_

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

/// Report whether the given operation, and all nested operations, are legal as
/// specified by the give ConversionTarget. Returns failure and emits error
/// diagnostics if any operations are not legal as well as a summary of the
/// illegal operations. Does not alter the IR.
LogicalResult verifyAllOperationsAreLegal(Operation *op,
                                          const ConversionTarget &target);

// Returns |oldAttr| converted to its new type via |typeConverter|, if needed.
Attribute convertAttribute(Location loc, Attribute oldAttr,
                           const TypeConverter &typeConverter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_CONVERSIONUTILS_H_
