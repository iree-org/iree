// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/Conversion/HALToHAL/ConvertHALToHAL.h"
#include "iree/compiler/Dialect/HAL/Conversion/IREEToHAL/ConvertIREEToHAL.h"
#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/ConvertStandardToHAL.h"
#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/Conversion/PreserveCompilerHints.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

// A pass converting the IREE flow dialect into the IREE HAL dialect.
class ConvertToHALPass
    : public PassWrapper<ConvertToHALPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect>();
    registry.insert<HALDialect>();
    registry.insert<StandardOpsDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    // Gather all interfaces from registered dialects.
    // These will perform the tensor->buffer mapping for their ops.
    SmallVector<const HALConversionDialectInterface *, 4> conversionInterfaces;
    for (auto *dialect : context->getLoadedDialects()) {
      if (auto *conversionInterface =
              dialect
                  ->getRegisteredInterface<HALConversionDialectInterface>()) {
        conversionInterfaces.emplace_back(conversionInterface);
      }
    }

    HALTypeConverter typeConverter(conversionInterfaces);
    HALConversionTarget conversionTarget(context, typeConverter);

    OwningRewritePatternList patterns(&getContext());

    populateIREEToHALPatterns(context, conversionTarget, typeConverter,
                              patterns);

    setupCompilerHintsLegality(context, conversionTarget, typeConverter);
    populatePreserveCompilerHintsPatterns(context, patterns);

    setupStandardToHALLegality(context, conversionTarget, typeConverter);
    populateStandardToHALPatterns(context, patterns, typeConverter);

    setupFlowToHALLegality(context, conversionTarget, typeConverter);
    populateFlowToHALPatterns(context, patterns, typeConverter);

    setupHALToHALLegality(context, conversionTarget, typeConverter);
    populateHALToHALPatterns(context, patterns, typeConverter);

    // Gather all HAL dialect conversion patterns from custom dialects.
    // These will perform the tensor->buffer mapping for their ops.
    for (auto *conversionInterface : conversionInterfaces) {
      conversionInterface->setupConversionTarget(conversionTarget, patterns,
                                                 typeConverter);
    }

    // NOTE: we allow ops that we don't know about to allow custom dialects
    // that don't need anything HAL-specific to pass through. This is handled by
    // the fallback type legality support of the
    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertToHALPass() {
  return std::make_unique<ConvertToHALPass>();
}

static PassRegistration<ConvertToHALPass> pass(
    "iree-convert-to-hal",
    "Convert input flow/std/etc dialects to the IREE HAL dialect.");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
