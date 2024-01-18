// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/HAL/Conversion/HALToHAL/Patterns.h"
#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/Patterns.h"
#include "iree/compiler/Dialect/HAL/Conversion/StreamToHAL/Patterns.h"
#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/Conversion/UtilToHAL/Patterns.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_CONVERTTOHALPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-conversion
//===----------------------------------------------------------------------===//

struct ConvertToHALPass
    : public IREE::HAL::impl::ConvertToHALPassBase<ConvertToHALPass> {
  void runOnOperation() override {
    auto *context = &getContext();

    // Gather all interfaces from registered dialects.
    // These will perform the tensor->buffer mapping for their ops.
    SmallVector<const HALConversionDialectInterface *> conversionInterfaces;
    for (auto *dialect : context->getLoadedDialects()) {
      if (auto *conversionInterface =
              dialect
                  ->getRegisteredInterface<HALConversionDialectInterface>()) {
        conversionInterfaces.emplace_back(conversionInterface);
      }
    }

    HALTypeConverter typeConverter(conversionInterfaces);
    HALConversionTarget conversionTarget(context, typeConverter);

    RewritePatternSet patterns(&getContext());

    populateHALToHALPatterns(context, conversionTarget, typeConverter,
                             patterns);
    populateUtilToHALPatterns(context, conversionTarget, typeConverter,
                              patterns);
    populateStandardToHALPatterns(context, conversionTarget, typeConverter,
                                  patterns);
    populateStreamToHALPatterns(context, conversionTarget, typeConverter,
                                patterns);

    // Gather all HAL dialect conversion patterns from custom dialects.
    // These will perform the tensor->buffer mapping for their ops.
    for (auto *conversionInterface : conversionInterfaces) {
      conversionInterface->setupConversionTarget(conversionTarget, patterns,
                                                 typeConverter);
    }

    // NOTE: we allow ops that we don't know about to allow custom dialects
    // that don't need anything HAL-specific to pass through.
    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
