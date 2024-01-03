// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONDIALECTINTERFACE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// An interface for dialects to expose HAL conversion functionality.
// The HAL conversion pass will query used dialects via this interface to find
// conversion patterns that map from a higher-level dialect containing ops that
// work on tensors to lower-level ops that work with HAL buffers and raw shapes.
//
// Implementations may choose to have different dialects for these levels of IR
// or for simplicity (and reduction of boilerplate) define the ops within the
// same dialect.
class HALConversionDialectInterface
    : public DialectInterface::Base<HALConversionDialectInterface> {
public:
  HALConversionDialectInterface(Dialect *dialect) : Base(dialect) {}

  // Populates |patterns| with rewrites that convert from a higher-level
  // tensor-based dialect to ops that interoperate with HAL types.
  // |target| must have newly legal and illegal ops/dialects specified to ensure
  // the conversion takes place.
  virtual void setupConversionTarget(ConversionTarget &target,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) const = 0;

  // Converts a type.
  // Will be called from the corresponding TypeConverter hook.
  virtual LogicalResult convertType(Type t,
                                    SmallVectorImpl<Type> &results) const {
    return failure();
  }
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONDIALECTINTERFACE_H_
