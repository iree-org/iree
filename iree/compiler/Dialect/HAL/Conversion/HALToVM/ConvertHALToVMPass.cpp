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

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// A pass converting the IREE HAL dialect into the IREE VM dialect.
// Used only for testing as in the common case we only rely on rewrite patterns.
class ConvertHALToVMPass : public ModulePass<ConvertHALToVMPass> {
  void runOnModule() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<IREE::VM::VMDialect>();
    target.addIllegalDialect<IREE::HAL::HALDialect>();

    OwningRewritePatternList patterns;
    populateHALToVMPatterns(&getContext(), patterns);

    // NOTE: we allow other dialects besides just VM during this pass as we are
    // only trying to eliminate the std ops. When used as part of a larger set
    // of rewrites a full conversion should be used instead.
    if (failed(applyPartialConversion(getModule(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

static PassRegistration<ConvertHALToVMPass> pass(
    "iree-convert-hal-to-vm", "Convert HAL ops to the IREE VM dialect");

}  // namespace iree_compiler
}  // namespace mlir
