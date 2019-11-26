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

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {

SymbolRefAttr getOpImportSymbolName(Operation *op) {
  return SymbolRefAttr::get(("_" + op->getName().getStringRef()).str(),
                            op->getContext());
}

LogicalResult appendImportModule(IREE::VM::ModuleOp importModuleOp,
                                 ModuleOp targetModuleOp) {
  SymbolTable symbolTable(targetModuleOp);
  OpBuilder targetBuilder(targetModuleOp);
  targetBuilder.setInsertionPoint(&targetModuleOp.getBody()->back());
  importModuleOp.walk([&](IREE::VM::ImportOp importOp) {
    std::string fullName =
        (importModuleOp.getName() + "." + importOp.getName()).str();
    auto *existingOp = symbolTable.lookup(fullName);
    // TODO(benvanik): verify that the imports match.
    if (!existingOp) {
      auto clonedOp = cast<IREE::VM::ImportOp>(targetBuilder.clone(*importOp));
      clonedOp.setName(fullName);
    }
  });
  return success();
}

LogicalResult appendImportModule(StringRef importModuleSrc,
                                 ModuleOp targetModuleOp) {
  auto importModuleRef =
      mlir::parseSourceString(importModuleSrc, targetModuleOp.getContext());
  if (!importModuleRef) {
    return targetModuleOp.emitError()
           << "unable to append import module; import module failed to parse";
  }
  for (auto importModuleOp : importModuleRef->getOps<IREE::VM::ModuleOp>()) {
    if (failed(appendImportModule(importModuleOp, targetModuleOp))) {
      importModuleOp.emitError() << "failed to import module";
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
