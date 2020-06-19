// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/Conversion/ConvertToHAL.h"
#include "iree/compiler/Dialect/IREE/Conversion/PreserveCompilerHints.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates only the flow.stream.* conversion patterns.
void populateFlowStreamToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter);

// Populates only the structural (module/function/etc) conversion patterns.
void populateFlowStructuralToHALPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns,
                                         TypeConverter &converter);

// Populates only the flow.tensor.* conversion patterns.
void populateFlowTensorToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter);

// Populates only the flow.variable.* conversion patterns.
void populateFlowVariableToHALPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns,
                                       TypeConverter &converter);

// Populates only the std.dim and std.rank conversion patterns.
void populateHalBufferViewShapePatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns,
                                        TypeConverter &converter);

namespace {

// A pass converting the IREE flow dialect into the IREE HAL dialect.
class ConvertFlowToHALPass
    : public PassWrapper<ConvertFlowToHALPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();

    SmallVector<const HALConversionDialectInterface *, 4> conversionInterfaces;
    // Gather all interfaces from registered dialects.
    // These will perform the tensor->buffer mapping for their ops.
    for (auto *dialect : context->getRegisteredDialects()) {
      if (auto *conversionInterface =
              dialect
                  ->getRegisteredInterface<HALConversionDialectInterface>()) {
        conversionInterfaces.emplace_back(conversionInterface);
      }
    }
    HALTypeConverter typeConverter(conversionInterfaces);
    HALConversionTarget target(context, typeConverter);
    target.addIllegalDialect<IREE::Flow::FlowDialect>();

    OwningRewritePatternList patterns;
    populateFlowStreamToHALPatterns(context, patterns, typeConverter);
    populateFlowStructuralToHALPatterns(context, patterns, typeConverter);
    populateFlowTensorToHALPatterns(context, patterns, typeConverter);
    populateFlowVariableToHALPatterns(context, patterns, typeConverter);
    populateHalBufferViewShapePatterns(context, patterns, typeConverter);
    populateIREEToHALPatterns(context, patterns);
    setupIREEToHALLegality(context, target);
    populatePreserveCompilerHintsPatterns(context, patterns);
    setupCompilerHintsLegality(context, target, typeConverter);

    // Gather all HAL dialect conversion patterns from custom dialects.
    // These will perform the tensor->buffer mapping for their ops.
    for (auto *conversionInterface : conversionInterfaces) {
      conversionInterface->setupConversionTarget(target, patterns,
                                                 typeConverter);
    }

    // NOTE: we allow ops that we don't know about to allow custom dialects
    // that don't need anything HAL-specific to pass through. This is handled by
    // the fallback type legality support of the
    if (failed(applyPartialConversion(getOperation(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertFlowToHALPass() {
  return std::make_unique<ConvertFlowToHALPass>();  // NOLINT
}

static PassRegistration<ConvertFlowToHALPass> pass(
    "iree-convert-flow-to-hal",
    "Convert input flow ops to the IREE HAL dialect");

}  // namespace iree_compiler
}  // namespace mlir
