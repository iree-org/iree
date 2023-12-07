// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONTARGET_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// A conversion target for the HAL dialect that ensures that tensor types are
// fully removed. Conversions targeting the HAL dialect should always use this.
class HALConversionTarget : public ConversionTarget {
public:
  HALConversionTarget(MLIRContext *context, TypeConverter &typeConverter);
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONTARGET_H_
