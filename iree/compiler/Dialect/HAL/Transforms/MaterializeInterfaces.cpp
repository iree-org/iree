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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/ExecutableTarget.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Adds IO ops (such as hal.io.binding) and updates function signatures to use
// them for their IO. We do this in a target-independent manner today so that we
// can share the same descriptor set logic and parameter population code on the
// scheduling side. In the future we could allow backends to opt into different
// behavior.
static llvm::Optional<IREE::HAL::InterfaceOp> declareInterfaceIO(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp) {
  auto moduleOp = sourceOp.getInnerModule();
  OpBuilder executableBuilder(targetOp.getContext());
  executableBuilder.setInsertionPointToStart(&targetOp.getBlock());

  // NOTE: we assume right now that all entry points have the same signature.
  SmallVector<FuncOp, 1> entryFuncOps;
  SmallVector<Location, 1> entryLocs;
  for (auto& op : sourceOp.getBlock()) {
    if (auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
      auto funcOp =
          moduleOp.lookupSymbol<FuncOp>(dispatchEntryOp.function_ref());
      entryFuncOps.push_back(funcOp);
      entryLocs.push_back(dispatchEntryOp.getLoc());
    }
  }
  auto interfaceLoc = executableBuilder.getFusedLoc(entryLocs);
  auto interfaceOp = executableBuilder.create<IREE::HAL::InterfaceOp>(
      interfaceLoc, "legacy_io");
  OpBuilder interfaceBuilder(interfaceOp);
  interfaceBuilder.setInsertionPointToStart(&interfaceOp.getBlock());

  // Add one binding per argument and result. This matches the legacy interface
  // and allows us to keep using the current binding setup on the scheduler
  // side.
  // NOTE: we assume right now that all entry points have the same signature.
  // TODO(benvanik): replace when we have descriptor sets in the HAL IR.
  auto anyFuncOp = entryFuncOps.front();
  int binding = 0;
  for (auto inputType : llvm::enumerate(anyFuncOp.getType().getInputs())) {
    auto bindingName = "arg" + std::to_string(inputType.index());
    if (inputType.value().isa<TensorType>()) {
      interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interfaceLoc, bindingName,
          /*set=*/APInt(32, 0), /*binding=*/APInt(32, binding++),
          IREE::HAL::DescriptorType::StorageBuffer,
          IREE::HAL::MemoryAccessBitfield::Read);
    } else {
      emitError(interfaceLoc)
          << "unsupported argument " << inputType.index() << " type "
          << inputType.value()
          << "; requires tensors or simple primitive values (i32, etc)";
      return llvm::None;
    }
  }
  for (auto outputType : llvm::enumerate(anyFuncOp.getType().getResults())) {
    auto bindingName = "ret" + std::to_string(outputType.index());
    if (outputType.value().isa<TensorType>()) {
      interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interfaceLoc, bindingName,
          /*set=*/APInt(32, 0), /*binding=*/APInt(32, binding++),
          IREE::HAL::DescriptorType::StorageBuffer,
          IREE::HAL::MemoryAccessBitfield::DiscardWrite);
    } else {
      emitError(interfaceLoc)
          << "unsupported result " << outputType.index() << " type "
          << outputType.value() << "; requires tensor types";
      return llvm::None;
    }
  }

  return interfaceOp;
}

// Creates a new entry function that uses the hal.interface bindings to marshal
// IO to the original entry function.
static Optional<FuncOp> createDispatchEntryThunk(
    FuncOp sourceFuncOp, IREE::HAL::InterfaceOp interfaceOp) {
  // Functions take all I/O through the interface API.
  auto sourceFuncType = sourceFuncOp.getType();
  auto thunkFuncType = FunctionType::get({}, {}, sourceFuncOp.getContext());
  auto thunkFuncOp = FuncOp::create(sourceFuncOp.getLoc(),
                                    sourceFuncOp.getName(), thunkFuncType);
  SymbolTable::setSymbolVisibility(thunkFuncOp,
                                   SymbolTable::Visibility::Public);
  sourceFuncOp.setName((sourceFuncOp.getName() + "_impl").str());
  SymbolTable::setSymbolVisibility(sourceFuncOp,
                                   SymbolTable::Visibility::Private);
  sourceFuncOp.getParentRegion()->getBlocks().front().push_front(thunkFuncOp);

  // For now we only support tensor types, so bindings are in order.
  // In the future we will want to provide N:M mappings (as well as the
  // information to compute offsets).
  int binding = 0;
  auto bindingOps = llvm::to_vector<4>(
      interfaceOp.getBlock().getOps<IREE::HAL::InterfaceBindingOp>());

  // Pull all arguments from the bindings.
  auto* thunkEntryBlock = thunkFuncOp.addEntryBlock();
  OpBuilder thunkEntryBuilder(thunkEntryBlock);
  auto zeroOffset = thunkEntryBuilder.createOrFold<mlir::ConstantOp>(
      thunkFuncOp.getLoc(), thunkEntryBuilder.getI32IntegerAttr(0));
  SmallVector<Value, 4> operands;
  for (auto inputType : sourceFuncType.getInputs()) {
    if (inputType.isa<TensorType>()) {
      auto bindingOp = bindingOps[binding++];
      auto loadOp = thunkEntryBuilder.create<IREE::HAL::InterfaceLoadTensorOp>(
          thunkFuncOp.getLoc(), inputType,
          thunkEntryBuilder.getSymbolRefAttr(
              interfaceOp.sym_name(),
              {thunkEntryBuilder.getSymbolRefAttr(bindingOp)}),
          zeroOffset);
      operands.push_back(loadOp.getResult());
    } else {
      sourceFuncOp.emitError() << "function argument type " << inputType
                               << " is not valid for interface I/O";
      return llvm::None;
    }
  }

  // Call the original entry function.
  auto callOp = thunkEntryBuilder.create<mlir::CallOp>(thunkFuncOp.getLoc(),
                                                       sourceFuncOp, operands);

  // Push all results to the bindings.
  for (auto result : callOp.getResults()) {
    auto bindingOp = bindingOps[binding++];
    thunkEntryBuilder.create<IREE::HAL::InterfaceStoreTensorOp>(
        thunkFuncOp.getLoc(), result,
        thunkEntryBuilder.getSymbolRefAttr(
            interfaceOp.sym_name(),
            {thunkEntryBuilder.getSymbolRefAttr(bindingOp)}),
        zeroOffset);
  }
  thunkEntryBuilder.create<mlir::ReturnOp>(thunkFuncOp.getLoc());

  return thunkFuncOp;
}

// Adds the entry point ops with assigned ordinals for each entry function.
// The entry points will all use the provided |interfaceOp|.
static LogicalResult declareEntryPointOps(IREE::Flow::ExecutableOp sourceOp,
                                          IREE::HAL::ExecutableOp targetOp,
                                          IREE::HAL::InterfaceOp interfaceOp) {
  // Insert interface bindings into the flow module so that symbol references
  // work. This hacks around our isolated module handling used by the legacy
  // backend translation API needing the source in a specific state.
  auto inlinedInterfaceOp = interfaceOp.clone();
  SymbolTable::setSymbolVisibility(inlinedInterfaceOp,
                                   SymbolTable::Visibility::Private);
  sourceOp.getInnerModule().push_back(inlinedInterfaceOp);

  OpBuilder builder(targetOp.getContext());
  builder.setInsertionPointAfter(interfaceOp);
  int nextOrdinal = 0;
  for (auto& op : sourceOp.getBlock()) {
    if (auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
      // Hardwire workgroup size to 1,1,1 by default. Backends can override.
      auto sourceFuncOp = sourceOp.getInnerModule().lookupSymbol<FuncOp>(
          dispatchEntryOp.function_ref());
      auto workGroupSizeAttr = DenseIntElementsAttr::get(
          VectorType::get(3, builder.getIntegerType(32)), {1, 1, 1});
      auto thunkFuncOp = createDispatchEntryThunk(sourceFuncOp, interfaceOp);
      if (!thunkFuncOp.hasValue()) {
        return failure();
      }
      dispatchEntryOp.setAttr("function_ref",
                              builder.getSymbolRefAttr(thunkFuncOp.getValue()));

      builder.create<IREE::HAL::ExecutableEntryPointOp>(
          dispatchEntryOp.getLoc(),
          builder.getStringAttr(thunkFuncOp->getName()),
          builder.getI32IntegerAttr(nextOrdinal++), workGroupSizeAttr,
          builder.getSymbolRefAttr(interfaceOp),
          TypeAttr::get(sourceFuncOp.getType()));
    }
  }
  return success();
}

class MaterializeInterfacesPass : public ModulePass<MaterializeInterfacesPass> {
 public:
  MaterializeInterfacesPass()
      : executableOptions_(getExecutableTargetOptionsFromFlags()) {}
  explicit MaterializeInterfacesPass(ExecutableTargetOptions executableOptions)
      : executableOptions_(executableOptions) {}

  void runOnModule() override {
    // Processes all executables within the input module and produce the output
    // HAL ops. We should ensure all deduping is performed prior to this when
    // it's easier to diff IR and where we still have the flow context.
    auto executableOps =
        llvm::to_vector<32>(getModule().getOps<IREE::Flow::ExecutableOp>());
    for (auto sourceOp : executableOps) {
      // Create the op that will contain the translated executables.
      OpBuilder builder(getModule().getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto targetOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());

      // Add IO ops to define the bindings and how parameters are passed.
      auto interfaceOp = declareInterfaceIO(sourceOp, targetOp);
      if (!interfaceOp.hasValue()) {
        return signalPassFailure();
      }

      // Annotate the entry points.
      // TODO(benvanik): allow entry points to use different interfaces.
      if (failed(declareEntryPointOps(sourceOp, targetOp,
                                      interfaceOp.getValue()))) {
        return signalPassFailure();
      }

      // Move the flow.executable into a source op to keep it for later
      // transformation.
      // TODO(benvanik): remove the need for this by doing all interface related
      // things here. The legacy utils currently require the original flow ops
      // to extract their attributes.
      OpBuilder targetBuilder(&targetOp.getBlock().back());
      auto sourceContainerOp =
          targetBuilder.create<IREE::HAL::ExecutableSourceOp>(
              sourceOp.getLoc());
      OpBuilder containerBuilder(&sourceContainerOp.getBlock().back());
      auto sourceModuleOp =
          containerBuilder.create<ModuleOp>(sourceOp.getLoc());
      sourceOp.getOperation()->moveBefore(&sourceModuleOp.getBody()->front());
    }
  }

 private:
  ExecutableTargetOptions executableOptions_;
};

std::unique_ptr<OpPassBase<ModuleOp>> createMaterializeInterfacesPass(
    ExecutableTargetOptions executableOptions) {
  return std::make_unique<MaterializeInterfacesPass>(
      executableOptions);  // NOLINT
}

static PassRegistration<MaterializeInterfacesPass> pass(
    "iree-hal-materialize-interfaces",
    "Materializes hal.executable ops from flow.executable ops");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
