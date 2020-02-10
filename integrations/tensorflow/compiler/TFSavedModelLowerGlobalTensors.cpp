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

static LogicalResult importTfSavedModelGlobalTensorsToIREEFlow(
    ModuleOp module) {
  OpBuilder globalBuilder(module.getBodyRegion());
  SymbolTable symbolTable(module);

  DenseMap<StringRef, std::string> symNameToFlowSymName;
  for (auto globalTensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
    auto exportedNames = tf_saved_model::GetExportedNames(globalTensor);
    std::string flowSymName;
    if (exportedNames.empty()) {
      flowSymName = "__iree_flow_" + globalTensor.sym_name().str();
    } else if (exportedNames.size() == 1) {
      flowSymName = exportedNames[0].str();
    } else {
      return globalTensor.emitError()
             << "Multiple exported names for global tensor not supported yet";
    }
    symNameToFlowSymName[globalTensor.sym_name()] = flowSymName;
    globalBuilder.create<IREE::Flow::VariableOp>(
        globalTensor.getLoc(), flowSymName, globalTensor.is_mutable(),
        globalTensor.type(), globalTensor.value());
  }

  for (auto func : module.getOps<FuncOp>()) {
    SmallVector<unsigned, 4> argsToErase;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      tf_saved_model::GlobalTensorOp globalTensor =
          tf_saved_model::LookupBoundInput(func, i, symbolTable);
      if (!globalTensor) {
        continue;
      }
      argsToErase.push_back(i);
      auto flowSymRef = globalBuilder.getSymbolRefAttr(
          symNameToFlowSymName[globalTensor.sym_name()]);
      Value arg = func.getArgument(i);
      if (globalTensor.is_mutable()) {
        // The value is a tensor<*x!tf.resource> type, which flows into
        // tf.ReadVariableOp/tf.AssignVariableOp.
        // XLA resource functionalization should have canonicalized everything
        // to uses of those two ops in the body of the tf_saved_model exported
        // function.
        for (OpOperand &operand : llvm::make_early_inc_range(arg.getUses())) {
          if (auto readVariable =
                  dyn_cast<TF::ReadVariableOp>(operand.getOwner())) {
            auto load = OpBuilder(readVariable)
                            .create<IREE::Flow::VariableLoadOp>(
                                readVariable.getLoc(),
                                readVariable.value().getType(), flowSymRef);
            readVariable.value().replaceAllUsesWith(load.result());
            readVariable.erase();
            continue;
          }
          if (auto assignVariable =
                  dyn_cast<TF::AssignVariableOp>(operand.getOwner())) {
            OpBuilder(assignVariable)
                .create<IREE::Flow::VariableStoreOp>(assignVariable.getLoc(),
                                                     assignVariable.value(),
                                                     flowSymRef);
            assignVariable.erase();
            continue;
          }
          return operand.getOwner()->emitError()
                 << "unknown op operating on resource for global tensor : "
                 << operand.getOwner()->getName();
        }
      } else {
        // The value is already a tensor value type. Just RAUW it with a
        // `flow.variable.load`.
        auto load = OpBuilder(func.getBody())
                        .create<IREE::Flow::VariableLoadOp>(
                            globalTensor.getLoc(), arg.getType(), flowSymRef);
        arg.replaceAllUsesWith(load.result());
      }
    }
    func.eraseArguments(argsToErase);
  }

  // Erase all the global tensors.
  for (auto globalTensor : llvm::make_early_inc_range(
           module.getOps<tf_saved_model::GlobalTensorOp>())) {
    globalTensor.erase();
  }
  return success();
}

class TFSavedModelLowerGlobalTensors
    : public ModulePass<TFSavedModelLowerGlobalTensors> {
 public:
  void runOnModule() override {
    if (failed(importTfSavedModelGlobalTensorsToIREEFlow(getModule()))) {
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
