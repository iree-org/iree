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
#include "iree/base/signature_mangle.h"
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
      Value arg = func.getArgument(i);
      if (global_tensor.is_mutable()) {
        // The value is a tensor<*x!tf.resource> type, which flows into
        // tf.ReadVariableOp/tf.AssignVariableOp.
        // XLA resource functionalization should have canonicalized everything
        // to uses of those two ops in the body of the tf_saved_model exported
        // function.
        for (OpOperand &operand : llvm::make_early_inc_range(arg.getUses())) {
          if (auto read_variable =
                  dyn_cast<TF::ReadVariableOp>(operand.getOwner())) {
            auto load = OpBuilder(read_variable)
                            .create<IREE::Flow::VariableLoadOp>(
                                read_variable.getLoc(),
                                read_variable.value().getType(), flow_sym_ref);
            read_variable.value().replaceAllUsesWith(load.result());
            read_variable.erase();
            continue;
          }
          if (auto assign_variable =
                  dyn_cast<TF::AssignVariableOp>(operand.getOwner())) {
            OpBuilder(assign_variable)
                .create<IREE::Flow::VariableStoreOp>(assign_variable.getLoc(),
                                                     assign_variable.value(),
                                                     flow_sym_ref);
            assign_variable.erase();
            continue;
          }
          return operand.getOwner()->emitError()
                 << "unknown op operating on resource for global tensor : "
                 << operand.getOwner()->getName();
        }
      } else {
        // The value is already a tensor value type. Just RAUW it with a
        // `flow.variable.load`.
        auto load =
            OpBuilder(func.getBody())
                .create<IREE::Flow::VariableLoadOp>(
                    global_tensor.getLoc(), arg.getType(), flow_sym_ref);
        arg.replaceAllUsesWith(load.result());
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

class TFSavedModelLowerGlobalTensors
    : public ModulePass<TFSavedModelLowerGlobalTensors> {
 public:
  void runOnModule() override {
    if (failed(ImportTfSavedModelGlobalTensorsToIREEFlow(getModule()))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createTFSavedModelLowerGlobalTensors() {
  return std::make_unique<TFSavedModelLowerGlobalTensors>();
}

static PassRegistration<TFSavedModelLowerGlobalTensors> pass(
    "iree-tf-saved-model-lower-global-tensors",
    "Lowers tf_saved_model global tensors to flow dialect.");

}  // namespace iree_compiler
}  // namespace mlir
