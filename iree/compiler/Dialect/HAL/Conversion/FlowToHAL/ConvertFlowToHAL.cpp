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
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Types.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

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

namespace {

// Converts types to Flow tensors and shapes to HAL types.
class FlowTensorTypeConverter : public TypeConverter {
 public:
  explicit FlowTensorTypeConverter(MLIRContext *context) {}

  Type convertType(Type type) override {
    if (type.isa<TensorType>()) {
      return IREE::RefPtrType::get(
          IREE::HAL::BufferType::get(type.getContext()));
    }
    return type;
  }

  // TODO(benvanik): signature conversion for output buffers.
};

// A pass converting the IREE flow dialect into the IREE HAL dialect.
class ConvertFlowToHALPass : public ModulePass<ConvertFlowToHALPass> {
 public:
  void runOnModule() override {
    auto *context = &getContext();

    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp, ModuleOp, ModuleTerminatorOp>();
    target.addLegalDialect<IREE::HAL::HALDialect>();
    target.addIllegalDialect<IREE::Flow::FlowDialect>();

    target.addLegalOp<IREE::HAL::ExecutableOp>();
    target.markOpRecursivelyLegal<IREE::HAL::ExecutableOp>();

    FlowTensorTypeConverter typeConverter(context);

    OwningRewritePatternList patterns;
    populateFlowStructuralToHALPatterns(context, patterns, typeConverter);
    populateFlowTensorToHALPatterns(context, patterns, typeConverter);
    populateFlowVariableToHALPatterns(context, patterns, typeConverter);

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });
    target.addDynamicallyLegalOp<ConstantOp>(
        [&](ConstantOp op) { return !op.getType().isa<TensorType>(); });

    // NOTE: we allow other dialects besides just VM during this pass as we are
    // only trying to eliminate the std ops. When used as part of a larger set
    // of rewrites a full conversion should be used instead.
    if (failed(applyPartialConversion(getModule(), target, patterns,
                                      &typeConverter))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> createConvertFlowToHALPass() {
  return std::make_unique<ConvertFlowToHALPass>();  // NOLINT
}

static PassRegistration<ConvertFlowToHALPass> pass(
    "iree-convert-flow-to-hal",
    "Convert input flow ops to the IREE HAL dialect");

}  // namespace iree_compiler
}  // namespace mlir
