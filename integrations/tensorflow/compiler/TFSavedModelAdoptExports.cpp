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
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace iree_compiler {

namespace {

LogicalResult ImportTfSavedModelGlobalTensorsToIREEFlow(ModuleOp module) {
  OpBuilder global_builder(module.getBodyRegion());
  SymbolTable symbol_table(module);

  DenseMap<StringRef, std::string> sym_name_to_flow_sym_name;
  for (auto global_tensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
    auto exported_names = tf_saved_model::GetExportedNames(global_tensor);
    std::string flow_sym_name;
    if (exported_names.empty()) {
      flow_sym_name = "__iree_flow_" + global_tensor.sym_name().str();
    } else if (exported_names.size() == 1) {
      flow_sym_name = exported_names[0].str();
    } else {
      return global_tensor.emitError()
             << "Multiple exported names for global tensor not supported yet";
    }
    sym_name_to_flow_sym_name[global_tensor.sym_name()] = flow_sym_name;
    global_builder.create<IREE::Flow::VariableOp>(
        global_tensor.getLoc(), flow_sym_name, global_tensor.is_mutable(),
        global_tensor.type(), global_tensor.value());
  }

  for (auto func : module.getOps<FuncOp>()) {
    SmallVector<unsigned, 4> args_to_erase;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      tf_saved_model::GlobalTensorOp global_tensor =
          tf_saved_model::LookupBoundInput(func, i, symbol_table);
      if (!global_tensor) {
        continue;
      }
      args_to_erase.push_back(i);
      auto flow_sym_ref = global_builder.getSymbolRefAttr(
          sym_name_to_flow_sym_name[global_tensor.sym_name()]);
      Value *arg = func.getArgument(i);
      if (global_tensor.is_mutable()) {
        // The value is a tensor<*x!tf.resource> type, which flows into
        // tf.ReadVariableOp/tf.AssignVariableOp.
        // XLA resource functionalization should have canonicalized everything
        // to uses of those two ops in the body of the tf_saved_model exported
        // function.
        for (OpOperand &operand : llvm::make_early_inc_range(arg->getUses())) {
          if (auto read_variable =
                  dyn_cast<TF::ReadVariableOp>(operand.getOwner())) {
            auto load = OpBuilder(read_variable)
                            .create<IREE::Flow::VariableLoadOp>(
                                read_variable.getLoc(),
                                read_variable.value()->getType(), flow_sym_ref);
            read_variable.value()->replaceAllUsesWith(load.result());
            read_variable.erase();
            continue;
          }
          if (auto assign_variable =
                  dyn_cast<TF::AssignVariableOp>(operand.getOwner())) {
            OpBuilder(assign_variable)
                .create<IREE::Flow::VariableStoreOp>(assign_variable.getLoc(),
                                                     flow_sym_ref,
                                                     assign_variable.value());
            assign_variable.erase();
            continue;
          }
          return operand.getOwner()->emitError()
                 << "unknown op operating on resource for global tensor";
        }
      } else {
        // The value is already a tensor value type. Just RAUW it with a
        // `flow.variable.load`.
        auto load =
            OpBuilder(func.getBody())
                .create<IREE::Flow::VariableLoadOp>(
                    global_tensor.getLoc(), arg->getType(), flow_sym_ref);
        arg->replaceAllUsesWith(load.result());
      }
    }
    func.eraseArguments(args_to_erase);
  }

  // Erase all the global tensors.
  for (auto global_tensor : llvm::make_early_inc_range(
           module.getOps<tf_saved_model::GlobalTensorOp>())) {
    global_tensor.erase();
  }
  return success();
}

}  // namespace

class TFSavedModelAdoptExportsPass
    : public ModulePass<TFSavedModelAdoptExportsPass> {
 public:
  void runOnModule() override {
    mlir::Builder builder(getModule());

    if (failed(ImportTfSavedModelGlobalTensorsToIREEFlow(getModule()))) {
      return signalPassFailure();
    }

    // Handle saved model exported functions.
    for (auto func : getModule().getOps<FuncOp>()) {
      // Transfer exported names to IREE.
      auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
      if (exported_names.empty()) continue;

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
