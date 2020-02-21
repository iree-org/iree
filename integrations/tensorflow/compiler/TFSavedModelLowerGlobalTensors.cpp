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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {

namespace {

// Lattice value with monotonic merge operation for dataflow analysis.
//
// See LatticeTracker for more details about what dataflow analysis we are
// performing.
struct LatticeValue {
  // The set of possible global tensors which a particular Value might
  // dynamically correspond to.
  std::set<Operation *> possibleGlobalTensors;

  static LatticeValue singleGlobalTensor(
      tf_saved_model::GlobalTensorOp globalTensor) {
    LatticeValue ret;
    ret.possibleGlobalTensors.insert(globalTensor.getOperation());
    return ret;
  }

  static LatticeValue merge(LatticeValue a, LatticeValue b) {
    LatticeValue ret = a;
    for (Operation *globalTensor : b.possibleGlobalTensors)
      ret.possibleGlobalTensors.insert(globalTensor);
    return ret;
  }
};

bool operator==(LatticeValue a, LatticeValue b) {
  return a.possibleGlobalTensors == b.possibleGlobalTensors;
}
bool operator!=(LatticeValue a, LatticeValue b) { return !operator==(a, b); }

}  // namespace

static bool isResourceVarType(Type type) {
  if (auto tensorType = type.dyn_cast<TensorType>())
    if (tensorType.getElementType().isa<TF::ResourceType>()) return true;
  return false;
}

namespace {
// Map used for dataflow analysis to determine which global tensors might
// dynamically be represented by a resource variable.
class LatticeTracker {
 public:
  // Merge dataflow information from `mergeFromVal` into `v`.
  void mergeFrom(Value v, Value mergeFromVal) {
    assert(isResourceVarType(v.getType()));
    assert(isResourceVarType(mergeFromVal.getType()));
    mergeFromLatticeValue(v, lattice[mergeFromVal]);
  }

  // Merge the latticeValue `mergeFromLatticeValue` into the dataflow
  // information for `v`.
  void mergeFromLatticeValue(Value v, LatticeValue mergeFromLatticeValue) {
    assert(isResourceVarType(v.getType()));
    LatticeValue &latticeValue = lattice[v];
    LatticeValue originalValue = latticeValue;
    latticeValue = LatticeValue::merge(latticeValue, mergeFromLatticeValue);
    changed |= (originalValue != latticeValue);
  }

  LatticeValue getLatticeValue(Value v) {
    assert(v);
    assert(isResourceVarType(v.getType()));
    return lattice[v];
  }

  // Methods for tracking if a fixed-point was reached on this particular
  // dataflow iteration.
  void resetChanged() { changed = false; }
  bool hasChanged() { return changed; }

 private:
  bool changed = false;
  DenseMap<Value, LatticeValue> lattice;
};

}  // namespace

/// For each predecessor of `block`, return the argument lists it uses to invoke
/// `block`.
/// Note that a single predecessor can result in multiple arg lists, for example
/// a conditional branch with both sides jumping to the same `block`.
static SmallVector<Operation::operand_range, 4> getPredecessorArgLists(
    Block *block) {
  SmallVector<Operation::operand_range, 4> argLists;
  for (Block *pred : block->getPredecessors()) {
    Operation *term = pred->getTerminator();
    for (int i = 0, e = term->getNumSuccessors(); i < e; i++)
      if (term->getSuccessor(i) == block)
        argLists.push_back(term->getSuccessorOperands(i));
  }
  return argLists;
}

static void analyzeModule(LatticeTracker &latticeTracker, ModuleOp module,
                          const SymbolTable &symbolTable) {
  // TODO(silvasean): Make this analysis interprocedural.
  // TODO(silvasean): Add !flow.variable_ref type to avoid needing to do this in
  // the first place.

  // Simple dataflow analysis that only looks through phi nodes.
  // We don't handle "internally created resources" at all. That is, we assume
  // that all values of resource type are function arguments or block arguments
  // derived from function arguments.
  for (auto func : module.getOps<FuncOp>()) {
    if (!tf_saved_model::IsExported(func)) continue;

    // Initialize the lattice.
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto globalTensor =
          tf_saved_model::LookupBoundInput(func, i, symbolTable);
      if (globalTensor) {
        latticeTracker.mergeFromLatticeValue(
            func.getArgument(i),
            LatticeValue::singleGlobalTensor(globalTensor));
      }
    }
    // Propagate to a fixed-point.
    Block *entry = &func.front();
    llvm::ReversePostOrderTraversal<Block *> rpo(entry);
    do {
      latticeTracker.resetChanged();
      for (Block *block : llvm::make_range(rpo.begin(), rpo.end())) {
        if (block == entry) continue;
        SmallVector<Operation::operand_range, 4> argLists =
            getPredecessorArgLists(block);
        for (auto &argList : argLists)
          for (int i = 0, e = argList.size(); i < e; i++)
            if (isResourceVarType(block->getArgument(i).getType()))
              latticeTracker.mergeFrom(block->getArgument(i), argList[i]);
      }
    } while (latticeTracker.hasChanged());
  }
}

// Return a name that would be useful in a diagnostic mentioning this
// global tensor.
static StringRef getNameForDiagnostics(
    tf_saved_model::GlobalTensorOp globalTensor) {
  // If there is an exported name, that's probably the most useful.
  auto exportedNames = tf_saved_model::GetExportedNames(globalTensor);
  if (!exportedNames.empty()) {
    return exportedNames[0];
  }
  // Otherwise, the importer hopefully chose a useful sym_name.
  return globalTensor.sym_name();
}

namespace iree_compiler {

// This function, together with `rewriteResourceForKnownOp`, define the lowering
// of resource ops to the flow dialect.
static Value findResourceForKnownOp(Operation &op) {
  if (auto readVariable = dyn_cast<TF::ReadVariableOp>(op)) {
    return readVariable.resource();
  } else if (auto assignVariable = dyn_cast<TF::AssignVariableOp>(op)) {
    return assignVariable.resource();
  }
  return nullptr;
}

// For an op whose resource was found by `findResourceForKnownOp`, rewrite it
// to a corresponding flow variable, on the assumption that the resource is
// guaranteed to correspond to `correspondingFlowSymbol`.
static void rewriteResourceForKnownOp(
    Operation &op, FlatSymbolRefAttr correspondingFlowSymbol) {
  if (auto readVariable = dyn_cast<TF::ReadVariableOp>(op)) {
    auto load = OpBuilder(readVariable)
                    .create<IREE::Flow::VariableLoadOp>(
                        readVariable.getLoc(), readVariable.value().getType(),
                        correspondingFlowSymbol);
    readVariable.value().replaceAllUsesWith(load.result());
    readVariable.erase();
  } else if (auto assignVariable = dyn_cast<TF::AssignVariableOp>(op)) {
    OpBuilder(assignVariable)
        .create<IREE::Flow::VariableStoreOp>(assignVariable.getLoc(),
                                             assignVariable.value(),
                                             correspondingFlowSymbol);
    assignVariable.erase();
  } else {
    llvm_unreachable("attempting to rewrite resource for unknown op");
  }
}

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

  LatticeTracker latticeTracker;
  analyzeModule(latticeTracker, module, symbolTable);

  for (auto func : module.getOps<FuncOp>()) {
    // Our analysis only handles exported functions now.
    if (!tf_saved_model::IsExported(func)) continue;

    for (Block &block : func) {
      for (Operation &op : llvm::make_early_inc_range(block)) {
        // Identify if this an op that we can rewrite and find what resource it
        // operates on.
        Value resource = findResourceForKnownOp(op);
        if (!resource) {
          // Resources in successor operands are fine and will be rewritten
          // below. Otherwise, we cannot transform the program.
          if (llvm::any_of(op.getNonSuccessorOperands(), [](Value v) {
                return isResourceVarType(v.getType());
              })) {
            return op.emitError()
                   << "unknown op operating on resource for global tensor: "
                   << op.getName();
          }
          continue;
        }

        // Extract out the unique global tensor referred to by this op, or
        // emit a diagnostic.
        auto latticeValue = latticeTracker.getLatticeValue(resource);
        if (latticeValue.possibleGlobalTensors.size() != 1) {
          SmallVector<StringRef, 4> possibleGlobalTensorNames;
          for (Operation *globalTensor : latticeValue.possibleGlobalTensors) {
            possibleGlobalTensorNames.push_back(getNameForDiagnostics(
                cast<tf_saved_model::GlobalTensorOp>(globalTensor)));
          }
          llvm::sort(possibleGlobalTensorNames);
          std::string allGlobalTensorNames;
          for (auto globalTensorName : possibleGlobalTensorNames) {
            if (!allGlobalTensorNames.empty()) {
              allGlobalTensorNames += ", ";
            }
            allGlobalTensorNames += "'" + globalTensorName.str() + "'";
          }
          return op.emitError() << "cannot prove resource op uses a single "
                                   "global tensor: potential global tensors: "
                                << allGlobalTensorNames;
        }
        auto globalTensor = cast<tf_saved_model::GlobalTensorOp>(
            *latticeValue.possibleGlobalTensors.begin());

        // Rewrite the op.
        auto correspondingFlowSymbol = globalBuilder.getSymbolRefAttr(
            symNameToFlowSymName[globalTensor.sym_name()]);
        rewriteResourceForKnownOp(op, correspondingFlowSymbol);
      }
    }

    for (Block &block : func) {
      // The entry block will be rewritten below.
      if (&block == &func.front()) {
        continue;
      }
      for (int i = 0, e = block.getNumArguments(); i < e; i++) {
        int argNo = e - i - 1;
        // Erase in reverse to avoid shifting later arguments when erasing
        // earlier arguments.
        if (isResourceVarType(block.getArgument(argNo).getType())) {
          block.getArgument(argNo).dropAllUses();
          block.eraseArgument(argNo);
        }
      }
    }
  }

  for (auto func : module.getOps<FuncOp>()) {
    // Our analysis only handles exported functions now.
    if (!tf_saved_model::IsExported(func)) continue;
    SmallVector<unsigned, 4> argsToErase;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      if (auto globalTensor =
              tf_saved_model::LookupBoundInput(func, i, symbolTable)) {
        argsToErase.push_back(i);
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
