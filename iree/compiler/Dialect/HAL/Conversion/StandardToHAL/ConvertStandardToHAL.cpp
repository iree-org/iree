// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/ConvertStandardToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateStandardStructuralToHALPatterns(MLIRContext *context,
                                             OwningRewritePatternList &patterns,
                                             TypeConverter &converter);

void setupStandardToHALLegality(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter) {
  conversionTarget.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.
  conversionTarget.addDynamicallyLegalDialect<mlir::StandardOpsDialect>();
  conversionTarget.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });
}

void populateStandardToHALPatterns(MLIRContext *context,
                                   OwningRewritePatternList &patterns,
                                   TypeConverter &typeConverter) {
  populateStandardStructuralToHALPatterns(context, patterns, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
