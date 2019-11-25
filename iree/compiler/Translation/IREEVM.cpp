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

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/Dialect/Flow/Conversion/HLOToFlow/ConvertHLOToFlow.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

LogicalResult convertToFlowModule(ModuleOp moduleOp) {
  // Convert to canonical HLO and - for some special ops - the flow dialect.
  ConversionTarget conversionTarget(*moduleOp.getContext());
  OwningRewritePatternList conversionPatterns;
  setupHLOToFlowConversion(moduleOp.getContext(), conversionTarget,
                           conversionPatterns);
  if (failed(applyFullConversion(moduleOp, conversionTarget,
                                 conversionPatterns))) {
    return moduleOp.emitError() << "module is not in a compatible input format";
  }

  // Run the flow transform pipeline to partition and produce the flow module.
  PassManager passManager(moduleOp.getContext());
  IREE::Flow::buildFlowTransformPassPipeline(passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }

  return success();
}

// TODO(benvanik): replace with the real HAL dialect.
static LogicalResult convertToTemporaryHALModule(
    ModuleOp moduleOp, IREE::HAL::ExecutableTargetOptions executableOptions) {
  PassManager passManager(moduleOp.getContext());
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError()
           << "failed to run flow transformation pass pipeline";
  }
  return success();
}

// Converts the lowered module to a canonical vm.module containing only vm ops.
// This uses patterns to convert from standard ops and other dialects to their
// vm ABI form.
static LogicalResult convertToVMModule(ModuleOp moduleOp) {
  ConversionTarget conversionTarget(*moduleOp.getContext());
  conversionTarget.addLegalDialect<IREE::VM::VMDialect>();
  conversionTarget.addIllegalDialect<IREE::HAL::HALDialect>();
  conversionTarget.addIllegalDialect<StandardOpsDialect>();
  // NOTE: we need to allow the outermost std.module to be legal.
  conversionTarget.addDynamicallyLegalOp<mlir::ModuleOp>(
      [&](mlir::ModuleOp op) { return op.getParentOp() == nullptr; });
  conversionTarget.addDynamicallyLegalOp<mlir::ModuleTerminatorOp>(
      [&](mlir::ModuleTerminatorOp op) {
        return op.getParentOp()->getParentOp() == nullptr;
      });

  OwningRewritePatternList conversionPatterns;
  populateHALToVMPatterns(moduleOp.getContext(), conversionPatterns);
  populateStandardToVMPatterns(moduleOp.getContext(), conversionPatterns);

  // TODO(benvanik): HAL -> VM conversion.

  if (failed(applyFullConversion(moduleOp, conversionTarget, conversionPatterns,
                                 getStandardToVMTypeConverter()))) {
    return moduleOp.emitError() << "conversion to vm.module failed";
  }

  return success();
}

LogicalResult translateFromMLIRToVMBytecodeModule(
    ModuleOp moduleOp, IREE::HAL::ExecutableTargetOptions executableOptions,
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    llvm::raw_ostream &output) {
  // Convert from our source to a vm.module in canonical form.
  // After this completes we have a non-bytecode-specific vm.module that we
  // could lower to other forms (LLVM IR, C, etc).
  if (failed(convertToFlowModule(moduleOp)) ||
      failed(convertToTemporaryHALModule(moduleOp, executableOptions)) ||
      failed(convertToVMModule(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> vm failed";
  }

  // Serialize to bytecode.
  return translateModuleToBytecode(moduleOp, bytecodeOptions, output);
}

}  // namespace iree_compiler
}  // namespace mlir
