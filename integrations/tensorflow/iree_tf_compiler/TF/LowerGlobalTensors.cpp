// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "tf-lower-global"

namespace mlir {
namespace iree_integrations {
namespace TF {

// The value of our lattice represents the tf_saved_model::GlobalTensorOp
// matching the value.
struct ResourceLatticeValue {
  explicit ResourceLatticeValue(tf_saved_model::GlobalTensorOp op = nullptr) {
    if (op) ops.insert(op);
  }

  static ResourceLatticeValue getPessimisticValueState(MLIRContext *context) {
    return ResourceLatticeValue();
  }

  static ResourceLatticeValue getPessimisticValueState(Value value) {
    if (auto barg = value.dyn_cast<BlockArgument>()) {
      if (func::FuncOp func =
              dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp())) {
        SymbolTable symbolTable(func->getParentOfType<ModuleOp>());
        auto global_tensor = tf_saved_model::LookupBoundInputOfType<
            tf_saved_model::GlobalTensorOp>(func, barg.getArgNumber(),
                                            symbolTable);
        return ResourceLatticeValue(global_tensor);
      }
    }
    return ResourceLatticeValue();
  }

  bool operator==(const ResourceLatticeValue &rhs) const {
    return ops == rhs.ops;
  }

  static ResourceLatticeValue join(const ResourceLatticeValue &lhs,
                                   const ResourceLatticeValue &rhs) {
    // Take union of both sets of possible tf_saved_model::GlobalTensorOp values
    // that can be referenced here.
    ResourceLatticeValue ret;
    ret.ops.insert(lhs.ops.begin(), lhs.ops.end());
    ret.ops.insert(rhs.ops.begin(), rhs.ops.end());
    return ret;
  }

  void print(raw_ostream &os) const {
    llvm::interleaveComma(ops, os << "["), os << "]";
  }

  // The location which originated the int value.
  // IR constructs (i.e., tf_saved_model::GlobalTensorOp) are not const-correct.
  mutable DenseSet<tf_saved_model::GlobalTensorOp> ops;
};

class ResourceAnalysis : public ::mlir::dataflow::SparseDataFlowAnalysis<
                             dataflow::Lattice<ResourceLatticeValue>> {
 public:
  using StateT = dataflow::Lattice<ResourceLatticeValue>;
  using ::mlir::dataflow::SparseDataFlowAnalysis<
      StateT>::SparseDataFlowAnalysis;
  ~ResourceAnalysis() override = default;

  void visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                      ArrayRef<StateT *> results) override {
    LLVM_DEBUG(llvm::dbgs() << "ResAn: Visiting operation: " << *op << "\n");
    setAllToEntryStates(results);
  }

  void setToEntryState(StateT *lattice) {
    propagateIfChanged(
        lattice, lattice->join(ResourceLatticeValue::getPessimisticValueState(
                     lattice->getPoint())));
  }
};

class LowerGlobalTensors
    : public PassWrapper<LowerGlobalTensors, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    mlir::ml_program::MLProgramDialect>();
  }

  StringRef getArgument() const override {
    return "iree-tf-saved-model-lower-global-tensors";
  }

  StringRef getDescription() const override {
    return "Lowers tf_saved_model global tensors to IREE flow dialect "
           "variables";
  }

  void runOnOperation() override;
};

void LowerGlobalTensors::runOnOperation() {
  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) return;
  if (auto sessionInitializer =
          tf_saved_model::GetSessionInitializerOp(module)) {
    sessionInitializer.emitError()
        << "session initializer is not supported yet";
    return signalPassFailure();
  }

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<ResourceAnalysis>();
  if (failed(solver.initializeAndRun(module))) return signalPassFailure();

  OpBuilder globalBuilder(module);
  globalBuilder.setInsertionPointToStart(&module.getRegion().front());

  DenseMap<tf_saved_model::GlobalTensorOp, ml_program::GlobalOp> symbolRefMap;
  for (auto globalTensor : module.getOps<tf_saved_model::GlobalTensorOp>()) {
    auto exportedNames = tf_saved_model::GetExportedNames(globalTensor);
    std::string name;
    if (exportedNames.empty()) {
      name = globalTensor.getSymName().str();
    } else if (exportedNames.size() == 1) {
      name = exportedNames[0].str();
    } else {
      globalTensor.emitError()
          << "multiple exported names for global tensor not supported yet";
      signalPassFailure();
      return;
    }
    auto global = globalBuilder.create<mlir::ml_program::GlobalOp>(
        globalTensor.getLoc(), name, globalTensor.getValue()->getType(),
        globalTensor.getIsMutable(), *globalTensor.getValue(), nullptr);
    global.setPrivate();
    symbolRefMap[globalTensor] = global;
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!tf_saved_model::IsExported(func)) continue;

    llvm::BitVector argsToErase(func.getNumArguments());
    DenseMap<Operation *, llvm::BitVector> removeOperands;

    for (BlockArgument val : func.getArguments()) {
      if (!getElementTypeOrSelf(val.getType()).isa<::mlir::TF::ResourceType>())
        continue;

      // Check that there is only a single global tensor associated with arg.
      const ResourceAnalysis::StateT *latticeElement =
          solver.lookupState<ResourceAnalysis::StateT>(val);
      if (!latticeElement || latticeElement->getValue().ops.size() != 1) {
        func.emitError() << "unable to determine unique global handle for func "
                         << func.getSymNameAttr() << "'s " << val.getArgNumber()
                         << " argument";
      }
      argsToErase.set(val.getArgNumber());

      tf_saved_model::GlobalTensorOp globalTensor =
          *latticeElement->getValue().ops.begin();
      ml_program::GlobalOp globalOp = symbolRefMap[globalTensor];

      for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
        if (auto read = llvm::dyn_cast<::mlir::TF::ReadVariableOp>(user)) {
          OpBuilder builder(read);
          Value load = builder.create<mlir::ml_program::GlobalLoadOp>(
              read.getLoc(), globalOp.getType(),
              FlatSymbolRefAttr::get(globalOp.getSymNameAttr()));
          if (read.getType() != load.getType()) {
            load = builder
                       .create<UnrealizedConversionCastOp>(read.getLoc(),
                                                           read.getType(), load)
                       .getResult(0);
          }
          read.getResult().replaceAllUsesWith(load);
          read.erase();
          continue;
        }

        if (auto assign = llvm::dyn_cast<::mlir::TF::AssignVariableOp>(user)) {
          OpBuilder builder(assign);
          Value value = assign.getValue();
          if (globalOp.getType() != value.getType()) {
            value = builder
                        .create<UnrealizedConversionCastOp>(
                            assign.getLoc(), globalOp.getType(), value)
                        .getResult(0);
          }
          builder.create<mlir::ml_program::GlobalStoreOp>(
              assign.getLoc(),
              FlatSymbolRefAttr::get(globalOp.getSymNameAttr()), value);
          assign.erase();
          continue;
        }

        if (llvm::dyn_cast<CallOpInterface>(user)) {
          llvm::BitVector &bvector = removeOperands[user];
          bvector.resize(user->getNumOperands());
          for (OpOperand &use : user->getOpOperands())
            bvector.set(use.getOperandNumber());
          continue;
        }

        user->emitError("could not lower resource op ") << user->getName();
        signalPassFailure();
        return;
      }
    }

    // As the other uses are call operations, we simply remove the arguments
    // as the function arguments will be removed below once that function is
    // processed.
    for (auto it : removeOperands) it.first->eraseOperands(it.second);

    func.eraseArguments(argsToErase);
  }

  // Erase all the global tensors.
  for (auto globalTensor : llvm::make_early_inc_range(
           module.getOps<tf_saved_model::GlobalTensorOp>())) {
    globalTensor.erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass() {
  return std::make_unique<LowerGlobalTensors>();
}

static PassRegistration<LowerGlobalTensors> pass;

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
