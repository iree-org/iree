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

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_

#include "iree/compiler/Dialect/Types.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

// Returns a symbol ref to the operation with the standard _ prefix.
// Runtime-imported modules always have a _ prefix to prevent conflicts with
// other VM-defined modules.
//
// Example:
//   MyFooOp 'my.foo' resolves to '_my.foo'
SymbolRefAttr getOpImportSymbolName(Operation *op);

// Appends a set of vm.import ops from a module to a target VM module.
// Imports will only be added if they are not already present in the target
// module.
LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp);
LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_IMPORTUTILS_H_
