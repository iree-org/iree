// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToHAL/ConvertHALToHAL.h"

#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

namespace mlir {
namespace iree_compiler {

// Populates only the hal.constant.* conversion patterns.
void populateHALConstantToHALPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns,
                                      TypeConverter &converter);

void setupHALToHALLegality(MLIRContext *context,
                           ConversionTarget &conversionTarget,
                           TypeConverter &typeConverter) {
  conversionTarget.addIllegalOp<IREE::HAL::ConstantSubspanOp>();
}

void populateHALToHALPatterns(MLIRContext *context,
                              OwningRewritePatternList &patterns,
                              TypeConverter &typeConverter) {
  populateHALConstantToHALPatterns(context, patterns, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
