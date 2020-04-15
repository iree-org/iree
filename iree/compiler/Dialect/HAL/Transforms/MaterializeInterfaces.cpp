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
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
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
  for (auto &op : sourceOp.getBlock()) {
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
  int pushConstantCount = 0;
  for (auto inputType : llvm::enumerate(anyFuncOp.getType().getInputs())) {
    auto bindingName = "arg" + std::to_string(inputType.index());
    if (inputType.value().isa<TensorType>()) {
      interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interfaceLoc, bindingName,
          /*set=*/APInt(32, 0), /*binding=*/APInt(32, binding++),
          IREE::HAL::DescriptorType::StorageBuffer,
          IREE::HAL::MemoryAccessBitfield::Read);
    } else if (auto indexType = inputType.value().dyn_cast<IndexType>()) {
      ++pushConstantCount;
    } else if (auto integerType = inputType.value().dyn_cast<IntegerType>()) {
      if (integerType.getIntOrFloatBitWidth() != 32) {
        emitError(interfaceLoc)
            << "unsupported argument " << inputType.index() << " bit depth "
            << integerType.getIntOrFloatBitWidth() << " (" << integerType
            << "); only 32-bit values are supported right now";
        return llvm::None;
      }
      ++pushConstantCount;
    } else {
      emitError(interfaceLoc)
          << "unsupported interface function argument " << inputType.index()
          << " type " << inputType.value()
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

  if (pushConstantCount > 0) {
    interfaceOp.setAttr("push_constants",
                        interfaceBuilder.getI32IntegerAttr(pushConstantCount));
  }

  return interfaceOp;
}

// Creates a new entry function that uses the hal.interface bindings to marshal
// IO to the original entry function.
// Invariants:
//   - The thunk function generates loads for entries in the InterfaceOp
//     based on category:
//       1. Push constants
//       2. Bindings
//     Within a category, the order follows the order within the interface.
//     Such an ordering can be useful for downstream code generation because
//     it can often be necessary to reference primitives in the materialization
//     of binding-based loads (i.e. for size calculations, etc). For any
//     stronger guarnatees or inter-load ordering constraints, downstream
//     code generation must explicitly take non-determinism of argument
//     ordering into account.
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
  auto *thunkEntryBlock = thunkFuncOp.addEntryBlock();
  OpBuilder thunkEntryBuilder = OpBuilder::atBlockEnd(thunkEntryBlock);
  Operation *firstNonConstOp = nullptr;
  auto positionForNonConst = [&]() {
    thunkEntryBuilder.setInsertionPointToEnd(thunkEntryBlock);
  };
  auto positionForConst = [&]() {
    if (firstNonConstOp) {
      thunkEntryBuilder.setInsertionPoint(firstNonConstOp);
    } else {
      thunkEntryBuilder.setInsertionPointToEnd(thunkEntryBlock);
    }
  };

  // Create load ops, first for push constants with binding based loads after.
  auto zeroOffset = thunkEntryBuilder.createOrFold<mlir::ConstantIndexOp>(
      thunkFuncOp.getLoc(), 0);
  SmallVector<Value, 4> operands;
  int pushConstantOffset = 0;
  for (auto inputType : sourceFuncType.getInputs()) {
    if (inputType.isa<TensorType>()) {
      positionForNonConst();
      auto bindingOp = bindingOps[binding++];
      auto loadOp = thunkEntryBuilder.create<IREE::HAL::InterfaceLoadTensorOp>(
          thunkFuncOp.getLoc(), inputType,
          thunkEntryBuilder.getSymbolRefAttr(
              interfaceOp.sym_name(),
              {thunkEntryBuilder.getSymbolRefAttr(bindingOp)}),
          zeroOffset);
      operands.push_back(loadOp.getResult());
      firstNonConstOp = loadOp;
    } else if (inputType.isa<IndexType>() || inputType.isa<IntegerType>()) {
      positionForConst();
      auto loadOp =
          thunkEntryBuilder.create<IREE::HAL::InterfaceLoadConstantOp>(
              thunkFuncOp.getLoc(), inputType, APInt(64, pushConstantOffset));
      operands.push_back(loadOp.getResult());
      ++pushConstantOffset;
    } else {
      sourceFuncOp.emitError() << "function argument type " << inputType
                               << " is not valid for interface I/O";
      return llvm::None;
    }
  }
  thunkEntryBuilder.setInsertionPointToEnd(thunkEntryBlock);

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
  for (auto &op : sourceOp.getBlock()) {
    if (auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
      auto sourceFuncOp = sourceOp.getInnerModule().lookupSymbol<FuncOp>(
          dispatchEntryOp.function_ref());
      auto thunkFuncOp = createDispatchEntryThunk(sourceFuncOp, interfaceOp);
      if (!thunkFuncOp.hasValue()) {
        return failure();
      }
      dispatchEntryOp.setAttr("function_ref",
                              builder.getSymbolRefAttr(thunkFuncOp.getValue()));

      builder.create<IREE::HAL::ExecutableEntryPointOp>(
          dispatchEntryOp.getLoc(),
          builder.getStringAttr(thunkFuncOp->getName()),
          builder.getI32IntegerAttr(nextOrdinal++),
          builder.getSymbolRefAttr(interfaceOp),
          TypeAttr::get(sourceFuncOp.getType()));
    }
  }
  return success();
}

// Creates zero or more hal.executable.target ops for each target backend.
// The source op will contain the flow.executable contents and any attributes
// the backend wants to carry along during transformation.
static LogicalResult constructTargetOps(TargetOptions targetOptions,
                                        IREE::Flow::ExecutableOp sourceOp,
                                        IREE::HAL::ExecutableOp executableOp) {
  // The user has specified what targets they want as a set of patterns. This
  // matches against those patterns so vulkan-* may match vulkan-v1.1 and
  // vulkan-v1.2.
  auto targetBackends = matchTargetBackends(targetOptions.targets);
  if (targetBackends.empty()) {
    auto diagnostic = sourceOp.emitError();
    diagnostic
        << "no target backends available for executable translation; ensure "
        << "they are linked in and the target options are properly "
        << "specified. requested = [ ";
    for (const auto &target : targetOptions.targets) {
      diagnostic << "'" << target << "' ";
    }
    diagnostic << "], available = [ ";
    for (const auto &target : getRegisteredTargetBackends()) {
      diagnostic << "'" << target << "' ";
    }
    diagnostic << "]";
    return diagnostic;
  }

  // Materialize all of the hal.executable.target ops for all backends we are
  // targeting. Note that each backend may create zero or more target ops.
  for (auto &targetBackend : targetBackends) {
    targetBackend->constructTargetOps(sourceOp, executableOp);
  }

  // Ensure that at least one target op got created. If it didn't that means
  // the executable cannot be translated and it's better to fail now.
  if (executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>().empty()) {
    auto diagnostic = sourceOp.emitError();
    diagnostic
        << "no target backend was able to handle this executable; tried = [ ";
    for (const auto &target : targetOptions.targets) {
      diagnostic << "'" << target << "' ";
    }
    diagnostic << "]";
    return diagnostic;
  }

  return success();
}

class MaterializeInterfacesPass
    : public PassWrapper<MaterializeInterfacesPass, OperationPass<ModuleOp>> {
 public:
  MaterializeInterfacesPass() : targetOptions_(getTargetOptionsFromFlags()) {}
  explicit MaterializeInterfacesPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void runOnOperation() override {
    // Processes all executables within the input module and produce the output
    // HAL ops. We should ensure all deduping is performed prior to this when
    // it's easier to diff IR and where we still have the flow context.
    auto sourceOps =
        llvm::to_vector<32>(getOperation().getOps<IREE::Flow::ExecutableOp>());
    for (auto sourceOp : sourceOps) {
      // Create the op that will contain the translated executables.
      OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto exectuableOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());
      SymbolTable::setSymbolVisibility(exectuableOp,
                                       SymbolTable::Visibility::Private);

      // Add IO ops to define the bindings and how parameters are passed.
      auto interfaceOp = declareInterfaceIO(sourceOp, exectuableOp);
      if (!interfaceOp.hasValue()) {
        return signalPassFailure();
      }

      // Annotate the entry points.
      // TODO(benvanik): allow entry points to use different interfaces.
      if (failed(declareEntryPointOps(sourceOp, exectuableOp,
                                      interfaceOp.getValue()))) {
        return signalPassFailure();
      }

      // Embed the hal.executable.target ops for each source.
      if (failed(constructTargetOps(targetOptions_, sourceOp, exectuableOp))) {
        return signalPassFailure();
      }

      sourceOp.erase();
    }
  }

 private:
  TargetOptions targetOptions_;
};

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeInterfacesPass(
    TargetOptions targetOptions) {
  return std::make_unique<MaterializeInterfacesPass>(targetOptions);  // NOLINT
}

static PassRegistration<MaterializeInterfacesPass> pass(
    "iree-hal-materialize-interfaces",
    "Materializes hal.executable ops from flow.executable ops");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
