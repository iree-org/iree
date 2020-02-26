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

#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

LogicalResult makeLegacyExecutableABI(IREE::HAL::ExecutableSourceOp sourceOp) {
  PassManager passManager(sourceOp.getContext());

  // Rewrite the hal.interface IO shim to use the legacy memref-based functions.
  passManager.addPass(createRewriteLegacyIOPass());

  if (failed(passManager.run(sourceOp.getInnerModule()))) {
    return sourceOp.emitError()
           << "required legacy rewriting/inlining failed (possibly "
              "due to symbol resolution issues)";
  }
  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
