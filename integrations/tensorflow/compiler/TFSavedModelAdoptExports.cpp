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

#include "integrations/tensorflow/compiler/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace iree_compiler {

class TFSavedModelAdoptExportsPass
    : public ModulePass<TFSavedModelAdoptExportsPass> {
 public:
  void runOnModule() override {
    mlir::Builder builder(getModule());

    // TODO(laurenzo): Import tf_saved_model.global_tensor ops.
    for (auto global_tensor :
         getModule().getOps<mlir::tf_saved_model::GlobalTensorOp>()) {
      global_tensor.emitError()
          << "This pass doesn't support global tensors yet";
      signalPassFailure();
      return;
    }

    // Handle saved model exported functions.
    for (auto func : getModule().getOps<FuncOp>()) {
      // Transfer exported names to IREE.
      auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
      if (exported_names.empty()) continue;

      // TODO(laurenzo): Validate that no one calls this (they shouldn't)
      // before modifying in place.
      if (!mlir::SymbolTable::symbolKnownUseEmpty(func.getName(),
                                                  getModule())) {
        func.emitError()
            << "Exported function is also called, which is not supported yet";
        signalPassFailure();
        return;
      }

      // TODO(laurenzo): After sequencer rework, we should just keep the
      // function name as-is and create explicit export ops for each exported
      // function.
      if (exported_names.size() > 1) {
        func.emitError() << "Multiple exported names not supported yet";
        signalPassFailure();
        return;
      }
      func.setName(exported_names.front());

      // Tag it as an IREE exported function.
      func.setAttr("iree.module.export", builder.getUnitAttr());

      // TODO(laurenzo): Validate and map structured arguments signaled via
      // non-monotonic tf_saved_model.index_path attributes. For now, just fail
      // if we encounter such arguments.
      for (int i = 0, e = func.getNumArguments(); i < e; i++) {
        auto array = func.getArgAttrOfType<mlir::ArrayAttr>(
            i, "tf_saved_model.index_path");
        if (!array) continue;
        auto attrs = array.getValue();
        if (attrs.size() == 1) {
          if (auto integer = attrs.front().dyn_cast<IntegerAttr>()) {
            if (integer.getValue() == i) {
              continue;
            }
          }
        }
        func.emitError()
            << "This pass doesn't support structured arguments yet";
        signalPassFailure();
        return;
      }

      // TODO(laurenzo): Also accept structured results. For now, just fail
      // if any are found.
      if (func.getNumResults() > 1) {
        func.emitError() << "This pass doesn't support multiple results yet";
        signalPassFailure();
        return;
      }
      for (int i = 0, e = func.getNumResults(); i < e; i++) {
        auto array = func.getResultAttrOfType<mlir::ArrayAttr>(
            i, "tf_saved_model.index_path");
        if (array && array.size() != 0) {
          func.emitError()
              << "This pass doesn't support structured results yet";
          signalPassFailure();
          return;
        }
      }

      // TODO(laurenzo): Handle bound inputs.
      for (int i = 0, e = func.getNumArguments(); i < e; i++) {
        if (func.getArgAttrOfType<mlir::SymbolRefAttr>(
                i, "tf_saved_model.bound_input")) {
          // emit error and signal pass failure
          func.emitError() << "This pass doesn't support bound inputs yet";
          signalPassFailure();
          return;
        }
      }

      // Remove its designation as a saved model export.
      func.removeAttr("tf_saved_model.exported_names");
    }

    // We should have now removed anything requiring saved model semantics.
    getModule().removeAttr("tf_saved_model.semantics");
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createTFSavedModelAdoptExportsPass() {
  return std::make_unique<TFSavedModelAdoptExportsPass>();
}

static PassRegistration<TFSavedModelAdoptExportsPass> pass(
    "iree-tf-saved-model-adopt-exports", "Adopts TF saved model exports");

}  // namespace iree_compiler
}  // namespace mlir
