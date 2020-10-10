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

#include <utility>

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class HoistInlinedRodataPass
    : public PassWrapper<HoistInlinedRodataPass,
                         OperationPass<IREE::VM::ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect>();
    registry.insert<IREE::VM::VMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable moduleSymbolTable(moduleOp);

    // Find all inline byte buffers in the module.
    auto funcOps = llvm::to_vector<4>(moduleOp.getOps<IREE::VM::FuncOp>());
    for (auto funcOp : funcOps) {
      auto inlineOps =
          llvm::to_vector<4>(funcOp.getOps<IREE::VM::RodataInlineOp>());
      if (inlineOps.empty()) continue;

      OpBuilder moduleBuilder(moduleOp.getContext());
      moduleBuilder.setInsertionPoint(funcOp);
      for (auto inlineOp : inlineOps) {
        auto rodataOp =
            OpBuilder(moduleOp.getContext())
                .create<IREE::VM::RodataOp>(inlineOp.getLoc(),
                                            (funcOp.getName() + "_const").str(),
                                            inlineOp.value());
        moduleSymbolTable.insert(rodataOp, moduleBuilder.getInsertionPoint());
        SymbolTable::setSymbolVisibility(rodataOp,
                                         SymbolTable::Visibility::Private);
        replaceInlineOpWithRodataRef(inlineOp, rodataOp);
      }
    }
  }

 private:
  // Replaces a vm.rodata.inline op with a vm.const.ref.rodata op that
  // references the module-level |rodataOp|.
  void replaceInlineOpWithRodataRef(IREE::VM::RodataInlineOp inlineOp,
                                    IREE::VM::RodataOp rodataOp) {
    OpBuilder builder(inlineOp);
    auto refOp =
        builder.create<IREE::VM::ConstRefRodataOp>(inlineOp.getLoc(), rodataOp);
    inlineOp.replaceAllUsesWith(refOp.value());
    inlineOp.erase();
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createHoistInlinedRodataPass() {
  return std::make_unique<HoistInlinedRodataPass>();
}

static PassRegistration<HoistInlinedRodataPass> pass(
    "iree-vm-hoist-inlined-rodata",
    "Hoists inline iree.byte_buffer values to module-level constant storage.");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
