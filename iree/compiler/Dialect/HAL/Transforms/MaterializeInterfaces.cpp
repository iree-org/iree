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

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
  int nextBindingOrdinal = 0;
  int pushConstantCount = 0;
  for (auto inputType : llvm::enumerate(anyFuncOp.getType().getInputs())) {
    if (inputType.value().isa<TensorType>()) {
      int bindingOrdinal = nextBindingOrdinal++;
      auto bindingName = "arg" + std::to_string(inputType.index());
      interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interfaceLoc, bindingName, /*set=*/0, /*binding=*/bindingOrdinal,
          IREE::HAL::DescriptorType::StorageBuffer,
          IREE::HAL::MemoryAccessBitfield::Read);
    } else if (auto tensorType =
                   inputType.value()
                       .dyn_cast<IREE::Flow::DispatchTensorType>()) {
      StringRef prefix;
      IREE::HAL::MemoryAccessBitfield memoryAccess =
          IREE::HAL::MemoryAccessBitfield::None;
      switch (tensorType.getAccess()) {
        case IREE::Flow::TensorAccess::ReadOnly:
          prefix = "ro";
          memoryAccess = IREE::HAL::MemoryAccessBitfield::Read;
          break;
        case IREE::Flow::TensorAccess::ReadWrite:
          prefix = "rw";
          memoryAccess = IREE::HAL::MemoryAccessBitfield::Read |
                         IREE::HAL::MemoryAccessBitfield::Write;
          break;
        case IREE::Flow::TensorAccess::WriteOnly:
          prefix = "wo";
          memoryAccess = IREE::HAL::MemoryAccessBitfield::DiscardWrite;
          break;
      }
      int bindingOrdinal = nextBindingOrdinal++;
      std::string bindingName =
          std::string(prefix) + std::to_string(bindingOrdinal);
      interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interfaceLoc, bindingName, /*set=*/0, /*binding=*/bindingOrdinal,
          IREE::HAL::DescriptorType::StorageBuffer, memoryAccess);
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
    int bindingOrdinal = nextBindingOrdinal++;
    auto bindingName = "ret" + std::to_string(outputType.index());
    if (outputType.value().isa<TensorType>()) {
      interfaceBuilder.create<IREE::HAL::InterfaceBindingOp>(
          interfaceLoc, bindingName, /*set=*/0, /*binding=*/bindingOrdinal,
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
    interfaceOp->setAttr("push_constants",
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
//     stronger guarantees or inter-load ordering constraints, downstream
//     code generation must explicitly take non-determinism of argument
//     ordering into account.
static Optional<FuncOp> createDispatchEntryThunk(
    FuncOp sourceFuncOp, IREE::HAL::InterfaceOp interfaceOp,
    IREE::HAL::ExecutableTargetOp targetOp) {
  // Clone the source FuncOp into the target then manipulate it into a
  // dispatch entry thunk.
  auto clonedFuncOp = sourceFuncOp.clone();
  targetOp.getInnerModule().push_back(clonedFuncOp);

  // Functions take all I/O through the interface API.
  auto sourceFuncType = clonedFuncOp.getType();
  auto thunkFuncType = FunctionType::get(clonedFuncOp.getContext(), {}, {});
  auto thunkFuncOp = FuncOp::create(clonedFuncOp.getLoc(),
                                    clonedFuncOp.getName(), thunkFuncType);
  clonedFuncOp.setName((clonedFuncOp.getName() + "_impl").str());
  clonedFuncOp.setPrivate();
  clonedFuncOp->getParentRegion()->getBlocks().front().push_front(thunkFuncOp);

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
    if (auto sourceType = inputType.dyn_cast<TensorType>()) {
      positionForNonConst();
      auto bindingOp = bindingOps[binding++];
      auto targetType = convertTensorTypeToABIType(sourceType);
      auto loadOp = thunkEntryBuilder.create<IREE::HAL::InterfaceLoadTensorOp>(
          thunkFuncOp.getLoc(), targetType,
          thunkEntryBuilder.getSymbolRefAttr(
              interfaceOp.sym_name(),
              {thunkEntryBuilder.getSymbolRefAttr(bindingOp)}),
          zeroOffset);
      Value abiValue =
          convertABITensorType(thunkFuncOp.getLoc(), loadOp.getResult(),
                               sourceType, thunkEntryBuilder);
      if (!abiValue) {
        clonedFuncOp.emitError()
            << "function argument type " << inputType
            << " cannot be converted to a HAL ABI type " << targetType;
        return llvm::None;
      }
      operands.push_back(abiValue);
      firstNonConstOp = loadOp;
    } else if (inputType.isa<IndexType>() || inputType.isa<IntegerType>()) {
      positionForConst();
      auto loadOp =
          thunkEntryBuilder.create<IREE::HAL::InterfaceLoadConstantOp>(
              thunkFuncOp.getLoc(), inputType, APInt(64, pushConstantOffset));
      operands.push_back(loadOp.getResult());
      ++pushConstantOffset;
    } else {
      clonedFuncOp.emitError() << "function argument type " << inputType
                               << " is not valid for interface I/O";
      return llvm::None;
    }
  }
  thunkEntryBuilder.setInsertionPointToEnd(thunkEntryBlock);

  // Call the original entry function.
  auto callOp = thunkEntryBuilder.create<mlir::CallOp>(thunkFuncOp.getLoc(),
                                                       clonedFuncOp, operands);

  // Push all results to the bindings.
  for (auto resultTypeValue :
       llvm::zip(sourceFuncType.getResults(), callOp.getResults())) {
    auto sourceType = std::get<0>(resultTypeValue).cast<TensorType>();
    auto targetType = convertTensorTypeToABIType(sourceType);
    Value resultValue = std::get<1>(resultTypeValue);
    Value abiValue = convertABITensorType(thunkFuncOp.getLoc(), resultValue,
                                          targetType, thunkEntryBuilder);
    if (!abiValue) {
      clonedFuncOp.emitError()
          << "function result type " << resultValue.getType()
          << " cannot be converted from HAL ABI type " << targetType;
      return llvm::None;
    }
    auto bindingOp = bindingOps[binding++];
    thunkEntryBuilder.create<IREE::HAL::InterfaceStoreTensorOp>(
        thunkFuncOp.getLoc(), abiValue,
        thunkEntryBuilder.getSymbolRefAttr(
            interfaceOp.sym_name(),
            {thunkEntryBuilder.getSymbolRefAttr(bindingOp)}),
        zeroOffset);
  }
  thunkEntryBuilder.create<mlir::ReturnOp>(thunkFuncOp.getLoc());

  return thunkFuncOp;
}

// Converts a tile dispatch entry function in the Flow dialect to an
// argumentless function referencing the hal.interface symbols directly.
// The contents of the function are unchanged beyond the argument marshaling
// inserted meaning that flow ops will remain for the target backends to
// convert as needed.
static void convertTiledEntryFuncToSymbols(
    IREE::Flow::DispatchEntryOp entryOp, FuncOp sourceFuncOp,
    IREE::HAL::InterfaceOp interfaceOp,
    IREE::HAL::ExecutableTargetOp targetOp) {
  // Clone the source FuncOp into the target then manipulate it into a
  // dispatch entry thunk.
  auto clonedFuncOp = sourceFuncOp.clone();
  targetOp.getInnerModule().push_back(clonedFuncOp);

  // Strip all arguments as functions take all I/O through the interface API.
  clonedFuncOp.setType(FunctionType::get(clonedFuncOp.getContext(), {}, {}));

  auto *entryBlock = &clonedFuncOp.front();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // For now we only support 1:1 tensor types, so bindings are in order.
  // In the future we will want to provide N:M mappings (as well as the
  // information to compute offsets).
  auto bindingOps = llvm::to_vector<4>(
      interfaceOp.getBlock().getOps<IREE::HAL::InterfaceBindingOp>());

  // We also don't offset things today but will as soon as the ringbuffer lands.
  auto zeroOffset =
      entryBuilder.createOrFold<mlir::ConstantIndexOp>(entryOp.getLoc(), 0);

  // As we only have inputs and this is just an indirection we insert everything
  // at the top of the entry block and let replaceAllUses handle the rest.
  int bindingOrdinal = 0;
  int pushConstantOffset = 0;
  for (BlockArgument arg : entryBlock->getArguments()) {
    if (auto tensorType =
            arg.getType().dyn_cast<IREE::Flow::DispatchTensorType>()) {
      auto bindingOp = bindingOps[bindingOrdinal++];
      auto bindingSymRefAttr = entryBuilder.getSymbolRefAttr(
          interfaceOp.sym_name(), {entryBuilder.getSymbolRefAttr(bindingOp)});
      auto subspanOp =
          entryBuilder.create<IREE::HAL::InterfaceBindingSubspanOp>(
              entryOp.getLoc(), arg.getType(), bindingSymRefAttr,
              /*byte_offset=*/zeroOffset,
              /*byte_length=*/Value{});
      arg.replaceAllUsesWith(subspanOp);
    } else {
      auto loadOp = entryBuilder.create<IREE::HAL::InterfaceLoadConstantOp>(
          entryOp.getLoc(), arg.getType(), APInt(64, pushConstantOffset++));
      arg.replaceAllUsesWith(loadOp);
    }
  }

  // Remove all of the arguments now that we've turned them into symbol
  // accesses and replaced their uses.
  while (entryBlock->getNumArguments() > 0) {
    entryBlock->eraseArgument(0);
  }
}

// Adds the entry point ops with assigned ordinals for each entry function.
// The entry points will all use the provided |interfaceOp|.
static LogicalResult declareEntryPointOps(
    IREE::Flow::ExecutableOp sourceExecutableOp,
    IREE::HAL::ExecutableOp targetExecutableOp,
    IREE::HAL::InterfaceOp interfaceOp) {
  auto targetOps =
      targetExecutableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>();
  for (auto targetOp : targetOps) {
    OpBuilder builder(&targetOp.getBlock().front());

    // For each Flow entry point, create a HAL entry point and dispatch thunk.
    int nextOrdinal = 0;
    for (auto &op : sourceExecutableOp.getBlock()) {
      if (auto dispatchEntryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
        auto sourceFuncOp =
            sourceExecutableOp.getInnerModule().lookupSymbol<FuncOp>(
                dispatchEntryOp.function_ref());

        // If this is a new-style tiled dispatch entry then we don't need a
        // thunk function and directly map from flow->hal. If it's a
        // legacy-style dispatch entry then we need to create a thunk function
        // to handle marshaling IO.
        if (dispatchEntryOp.workgroup_rank().hasValue()) {
          convertTiledEntryFuncToSymbols(dispatchEntryOp, sourceFuncOp,
                                         interfaceOp, targetOp);
        } else {
          auto thunkFuncOp =
              createDispatchEntryThunk(sourceFuncOp, interfaceOp, targetOp);
          if (!thunkFuncOp.hasValue()) {
            return failure();
          }
          dispatchEntryOp->setAttr(
              "function_ref", builder.getSymbolRefAttr(thunkFuncOp.getValue()));
        }

        builder.create<IREE::HAL::ExecutableEntryPointOp>(
            dispatchEntryOp.getLoc(),
            builder.getStringAttr(dispatchEntryOp.function_ref()),
            builder.getI32IntegerAttr(nextOrdinal++),
            builder.getSymbolRefAttr(interfaceOp),
            TypeAttr::get(sourceFuncOp.getType()), ArrayAttr{});
      }
    }

    // Copy interface bindings into the target module so symbol references work.
    auto inlinedInterfaceOp = interfaceOp.clone();
    inlinedInterfaceOp.setPrivate();
    targetOp.getInnerModule().push_back(inlinedInterfaceOp);
  }
  return success();
}

// Creates zero or more hal.executable.target ops for each target backend.
// The source op will contain the flow.executable contents and any attributes
// the backend wants to carry along during transformation.
static LogicalResult declareTargetOps(TargetOptions targetOptions,
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
    targetBackend->declareTargetOps(sourceOp, executableOp);
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

namespace {

template <typename SrcOp, typename DstOp>
class ConverterDispatchWorkgroupInfoPattern final
    : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getResult().getType(),
                                       op.dimensionAttr());
    return success();
  }
};

}  // namespace

class MaterializeInterfacesPass
    : public PassWrapper<MaterializeInterfacesPass, OperationPass<ModuleOp>> {
 public:
  explicit MaterializeInterfacesPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();

    auto targetBackends = matchTargetBackends(targetOptions_.targets);
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

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
      auto executableOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());
      executableOp.setVisibility(sourceOp.getVisibility());

      // Add IO ops to define the bindings and how parameters are passed.
      auto interfaceOp = declareInterfaceIO(sourceOp, executableOp);
      if (!interfaceOp.hasValue()) {
        return signalPassFailure();
      }

      // Embed the hal.executable.target ops for each source.
      if (failed(declareTargetOps(targetOptions_, sourceOp, executableOp))) {
        return signalPassFailure();
      }

      // Annotate the entry points.
      // TODO(benvanik): allow entry points to use different interfaces.
      if (failed(declareEntryPointOps(sourceOp, executableOp,
                                      interfaceOp.getValue()))) {
        return signalPassFailure();
      }

      // Convert interface-related flow.dispatch.* ops to their hal.* versions.
      OwningRewritePatternList patterns(&getContext());
      patterns.insert<ConverterDispatchWorkgroupInfoPattern<
                          IREE::Flow::DispatchWorkgroupIDOp,
                          IREE::HAL::InterfaceWorkgroupIDOp>,
                      ConverterDispatchWorkgroupInfoPattern<
                          IREE::Flow::DispatchWorkgroupCountOp,
                          IREE::HAL::InterfaceWorkgroupCountOp>,
                      ConverterDispatchWorkgroupInfoPattern<
                          IREE::Flow::DispatchWorkgroupSizeOp,
                          IREE::HAL::InterfaceWorkgroupSizeOp>>(
          executableOp.getContext());
      if (failed(applyPatternsAndFoldGreedily(executableOp,
                                              std::move(patterns)))) {
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
    "Materializes hal.executable ops from flow.executable ops", [] {
      auto options = getTargetOptionsFromFlags();
      return std::make_unique<MaterializeInterfacesPass>(options);
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
