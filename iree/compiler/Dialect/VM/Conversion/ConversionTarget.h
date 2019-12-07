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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONTARGET_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONTARGET_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// A conversion target for the VM dialect that handles some boilerplate for
// nested module conversion.
// Conversions targeting the VM dialect should always use this.
class VMConversionTarget : public ConversionTarget {
 public:
  // Ensures that a module has double-nesting to allow for module conversion.
  // If the module is already nested then this is a no-op.
  // Returns a pair of (outer module, inner module).
  //
  // Example:
  //  module { func @foo() { ... } }
  // ->
  //  module { module { func @foo() { ... } } }
  static std::pair<mlir::ModuleOp, mlir::ModuleOp> nestModuleForConversion(
      mlir::ModuleOp outerModuleOp);

  VMConversionTarget(MLIRContext *context);
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_CONVERSIONTARGET_H_
