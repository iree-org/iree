// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VMVX_CONVERSION_HALTOVMVX_CONVERTHALTOVMVX_H_
#define IREE_COMPILER_DIALECT_VMVX_CONVERSION_HALTOVMVX_CONVERTHALTOVMVX_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Converts a `() -> ()` function to the calling convention used by VMVX for
// passing in bindings, constants, and workgroup parameters.
LogicalResult updateHALToVMVXEntryFuncOp(mlir::FunctionOpInterface funcOp,
                                         TypeConverter &typeConverter);

// Populates conversion patterns from the IREE HAL dialect interface to the
// VMVX dialect interface.
void populateHALToVMVXPatterns(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               RewritePatternSet &patterns,
                               TypeConverter &typeConverter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_VMVX_CONVERSION_HALTOVMVX_CONVERTHALTOVMVX_H_
