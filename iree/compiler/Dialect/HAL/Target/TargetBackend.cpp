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

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

#include <algorithm>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

TargetOptions getTargetOptionsFromFlags() {
  static llvm::cl::OptionCategory halTargetOptionsCategory(
      "IREE HAL executable target options");

  // This function is called as part of registering the pass
  // TranslateExecutablesPass. Pass registery is also staticly initialized,
  // so targetBackendsFlags needs to be here to be initialized first.
  static llvm::cl::list<std::string> *targetBackendsFlag =
      new llvm::cl::list<std::string>{
          "iree-hal-target-backends",
          llvm::cl::desc("Target backends for executable compilation"),
          llvm::cl::ZeroOrMore, llvm::cl::cat(halTargetOptionsCategory)};

  TargetOptions targetOptions;
  targetOptions.targets = *targetBackendsFlag;
  return targetOptions;
}

// static
bool TargetBackend::matchPattern(StringRef value, StringRef pattern) {
  size_t nextCharIndex = pattern.find_first_of("*?");
  if (nextCharIndex == std::string::npos) {
    return value == pattern;
  } else if (nextCharIndex > 0) {
    if (value.substr(0, nextCharIndex) != pattern.substr(0, nextCharIndex)) {
      return false;
    }
    value = value.substr(nextCharIndex);
    pattern = pattern.substr(nextCharIndex);
  }
  if (value.empty() && pattern.empty()) {
    return true;
  }
  char patternChar = pattern[0];
  if (patternChar == '*' && pattern.size() > 1 && value.empty()) {
    return false;
  } else if (patternChar == '*' && pattern.size() == 1) {
    return true;
  } else if (patternChar == '?' || value[0] == patternChar) {
    return matchPattern(value.substr(1), pattern.substr(1));
  } else if (patternChar == '*') {
    return matchPattern(value, pattern.substr(1)) ||
           matchPattern(value.substr(1), pattern);
  }
  return false;
}

// static
BufferConstraintsAttr TargetBackend::makeDefaultBufferConstraints(
    MLIRContext *context) {
  // Picked to represent what we kind of want on CPU today.
  uint64_t maxAllocationSize = 1 * 1024 * 1024 * 1024ull;
  uint64_t minBufferOffsetAlignment = 16ull;
  uint64_t maxBufferRange = 1 * 1024 * 1024 * 1024ull;
  uint64_t minBufferRangeAlignment = 16ull;
  Builder b(context);
  return BufferConstraintsAttr::get(b.getIndexAttr(maxAllocationSize),
                                    b.getIndexAttr(minBufferOffsetAlignment),
                                    b.getIndexAttr(maxBufferRange),
                                    b.getIndexAttr(minBufferRangeAlignment));
}

BufferConstraintsAttr TargetBackend::queryBufferConstraints(
    MLIRContext *context) {
  return makeDefaultBufferConstraints(context);
}

void TargetBackend::declareTargetOps(IREE::Flow::ExecutableOp sourceOp,
                                     IREE::HAL::ExecutableOp executableOp) {
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  auto targetContainerOp = targetBuilder.create<IREE::HAL::ExecutableTargetOp>(
      sourceOp.getLoc(), name(), filter_pattern());
  OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
  containerBuilder.create<ModuleOp>(sourceOp.getLoc());
}

// Destructively merges |sourceModuleOp| into |targetModuleOp|.
// |targetSymbolTable| is updated with the new symbols.
static void mergeModuleInto(Operation *sourceModuleOp,
                            Operation *targetModuleOp,
                            DenseMap<StringRef, Operation *> &targetSymbolMap) {
  auto &sourceBlock = sourceModuleOp->getRegion(0).front();
  auto &targetBlock = targetModuleOp->getRegion(0).front();
  auto allOps = llvm::to_vector<8>(
      llvm::map_range(sourceBlock, [&](Operation &op) { return &op; }));
  for (auto &op : allOps) {
    if (op->hasTrait<OpTrait::IsTerminator>()) continue;
    if (auto symbolInterface = dyn_cast<SymbolOpInterface>(op)) {
      if (targetSymbolMap.count(symbolInterface.getName())) {
        // TODO(scotttodd): compare ops to ensure we aren't copying different
        // things with the same name.
        continue;
      }
      targetSymbolMap[symbolInterface.getName()] = op;
    }
    op->moveBefore(&targetBlock.back());
  }

  // Now that we're done cloning its ops, delete the original target op.
  sourceModuleOp->erase();
}

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
static void replaceEntryPointUses(
    mlir::ModuleOp moduleOp,
    const DenseMap<Attribute, Attribute> &replacements) {
  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    funcOp.walk([&](IREE::HAL::CommandBufferDispatchSymbolOp dispatchOp) {
      auto it = replacements.find(dispatchOp.entry_point());
      if (it != replacements.end()) {
        dispatchOp.entry_pointAttr(it->second.cast<SymbolRefAttr>());
      }
    });
  }
}

LogicalResult TargetBackend::linkExecutablesInto(
    mlir::ModuleOp moduleOp,
    ArrayRef<IREE::HAL::ExecutableOp> sourceExecutableOps,
    IREE::HAL::ExecutableOp linkedExecutableOp,
    IREE::HAL::ExecutableTargetOp linkedTargetOp,
    std::function<Operation *(mlir::ModuleOp moduleOp)> getInnerModuleFn,
    OpBuilder &builder) {
  llvm::SmallVector<IREE::HAL::InterfaceOp, 4> linkedInterfaceOps;
  int nextEntryPointOrdinal = 0;
  DenseMap<StringRef, Operation *> symbolMap;
  DenseMap<Attribute, Attribute> entryPointRefReplacements;
  auto linkedExecutableBuilder =
      OpBuilder::atBlockBegin(linkedExecutableOp.getBody());
  auto linkedTargetBuilder = OpBuilder::atBlockBegin(linkedTargetOp.getBody());
  for (auto sourceExecutableOp : sourceExecutableOps) {
    auto targetOps = llvm::to_vector<4>(
        sourceExecutableOp.getOps<IREE::HAL::ExecutableTargetOp>());
    for (auto targetOp : targetOps) {
      // Only process targets matching our pattern.
      if (!matchPattern(targetOp.target_backend_filter(), filter_pattern())) {
        continue;
      }

      // Clone entry point ops and queue remapping ordinals and updating
      // symbol refs.
      for (auto entryPointOp :
           targetOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
        // Lookup the interface used by this entry point.
        auto sourceInterfaceOp =
            SymbolTable::lookupNearestSymbolFrom<IREE::HAL::InterfaceOp>(
                sourceExecutableOp, entryPointOp.interfaceAttr());
        assert(sourceInterfaceOp && "cannot find source interface");
        IREE::HAL::InterfaceOp linkedInterfaceOp;
        for (auto interfaceOp : linkedInterfaceOps) {
          if (interfaceOp.isEquivalentTo(sourceInterfaceOp)) {
            linkedInterfaceOp = interfaceOp;
            break;
          }
        }
        if (!linkedInterfaceOp) {
          linkedInterfaceOp = dyn_cast<IREE::HAL::InterfaceOp>(
              linkedExecutableBuilder.clone(*sourceInterfaceOp));
          linkedInterfaceOp.setName(
              llvm::formatv("legacy_io_{0}", linkedInterfaceOps.size()).str());
          linkedInterfaceOps.push_back(linkedInterfaceOp);
        }

        auto newEntryPointOp =
            linkedTargetBuilder.create<IREE::HAL::ExecutableEntryPointOp>(
                entryPointOp.getLoc(), entryPointOp.sym_nameAttr(),
                builder.getI32IntegerAttr(nextEntryPointOrdinal++),
                builder.getSymbolRefAttr(linkedInterfaceOp.getName()),
                entryPointOp.signatureAttr(), ArrayAttr{});

        // Add to replacement table for fixing up dispatch calls referencing
        // this entry point.
        auto oldSymbolRefAttr =
            builder.getSymbolRefAttr(sourceExecutableOp.getName(),
                                     {builder.getSymbolRefAttr(targetOp),
                                      builder.getSymbolRefAttr(entryPointOp)});
        auto newSymbolRefAttr = builder.getSymbolRefAttr(
            linkedExecutableOp.getName(),
            {builder.getSymbolRefAttr(linkedTargetOp),
             builder.getSymbolRefAttr(newEntryPointOp)});
        entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      }

      // Merge the existing module into the new linked module op.
      auto sourceModuleOp = getInnerModuleFn(targetOp.getInnerModule());
      auto linkedModuleOp = getInnerModuleFn(linkedTargetOp.getInnerModule());
      mergeModuleInto(sourceModuleOp, linkedModuleOp, symbolMap);

      targetOp.erase();
    }

    if (sourceExecutableOp.getOps<IREE::HAL::ExecutableTargetOp>().empty()) {
      sourceExecutableOp.erase();
    }
  }

  // Update references to @executable::@target::@entry symbols.
  replaceEntryPointUses(moduleOp, entryPointRefReplacements);

  // Remove if we didn't add anything.
  if (linkedTargetOp.getOps<IREE::HAL::ExecutableEntryPointOp>().empty()) {
    linkedTargetOp.erase();
    linkedExecutableOp.erase();
  }

  return success();
}

std::array<Value, 3> TargetBackend::calculateDispatchWorkgroupSize(
    Location loc, IREE::HAL::ExecutableOp executableOp,
    IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
    OpBuilder &builder) {
  // When no workgroup size is specified we just assume [1,1,1].
  // This yields a workgroup count that models the extents of the workload.
  return {
      builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
      builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
      builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
  };
}

static std::array<Value, 3> calculateDispatchWorkgroupCountFromRegion(
    Location loc, IREE::HAL::ExecutableEntryPointOp entryPointOp,
    ValueRange workload, OpBuilder &builder) {
  Block *body = entryPointOp.getBlock();
  if (workload.size() != body->getNumArguments()) {
    ::mlir::emitError(loc,
                      "mismatch in number of workload values provided during "
                      "lowering to HAL and provided");
    return {
        builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
        builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
        builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
    };
  }
  BlockAndValueMapping bvm;
  for (auto args : llvm::zip(workload, body->getArguments())) {
    bvm.map(std::get<1>(args), std::get<0>(args));
  }
  for (Operation &op : body->without_terminator()) {
    builder.clone(op, bvm);
  }
  auto returnOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
  // Verifier of EntryPointOp checks that the return has 3 values.
  SmallVector<Value, 4> count = llvm::to_vector<4>(llvm::map_range(
      returnOp.operands(), [&bvm](Value v) { return bvm.lookup(v); }));
  return {count[0], count[1], count[2]};
}

std::array<Value, 3> TargetBackend::calculateDispatchWorkgroupCount(
    Location loc, IREE::HAL::ExecutableOp executableOp,
    IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
    OpBuilder &builder) {
  Region *region = entryPointOp.getBody();
  if (region) {
    return calculateDispatchWorkgroupCountFromRegion(loc, entryPointOp,
                                                     workload, builder);
  }
  auto workgroupSize = calculateDispatchWorkgroupSize(
      loc, executableOp, entryPointOp, workload, builder);
  return calculateDispatchWorkgroupCount(loc, workload, workgroupSize, builder);
}

std::array<Value, 3> TargetBackend::calculateDispatchWorkgroupCount(
    Location loc, ValueRange workload,
    const std::array<Value, 3> &workgroupSize, OpBuilder &builder) {
  std::array<Value, 3> result;

  auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
  if (workload.size() <= 3) {
    // 1-D to 3-D are easy (pad 2 to 0 dimensions) and divide by workgroup size.
    for (int i = 0; i < 3; ++i) {
      // Round up: (workload[i] + workgroup_size - 1) / workgroup_size;
      Value workloadI = i < workload.size() ? workload[i] : constantOne;
      workloadI = builder.createOrFold<mlir::SubIOp>(
          loc,
          builder.createOrFold<mlir::AddIOp>(loc, workloadI, workgroupSize[i]),
          constantOne);
      result[i] = builder.createOrFold<UnsignedDivIOp>(loc, workloadI,
                                                       workgroupSize[i]);
    }
  } else {
    // TODO(#4140): remapping of N-D to 3-D: this is not how you do this!
    Value flatWorkload = constantOne;
    for (auto workloadI : workload) {
      flatWorkload = builder.createOrFold<MulIOp>(loc, flatWorkload, workloadI);
    }
    for (int i = 0; i < 3; ++i) {
      // Round up: (workload[i] + workgroup_size - 1) / workgroup_size;
      auto rounded = builder.createOrFold<mlir::SubIOp>(
          loc,
          builder.createOrFold<mlir::AddIOp>(loc, flatWorkload,
                                             workgroupSize[i]),
          constantOne);
      auto workgroupCountI = builder.createOrFold<mlir::UnsignedDivIOp>(
          loc, rounded, workgroupSize[i]);
      result[i] = workgroupCountI;

      // Multiply back out and subtract from invocations.
      flatWorkload = builder.createOrFold<SubIOp>(
          loc, flatWorkload,
          builder.createOrFold<MulIOp>(loc, workgroupCountI, rounded));
    }
  }

  return result;
}

LogicalResult TargetBackend::recordDispatch(
    Location loc, DispatchState dispatchState,
    DeviceSwitchRewriter &switchRewriter) {
  SmallVector<Value, 4> regionArgs;
  regionArgs.push_back(dispatchState.commandBuffer);
  for (auto dim : dispatchState.workgroupCount) {
    regionArgs.push_back(dim);
  }
  auto *region = switchRewriter.addConditionRegion(
      IREE::HAL::DeviceMatchIDAttr::get(filter_pattern(), loc.getContext()),
      regionArgs);
  auto &entryBlock = region->front();
  auto commandBuffer = entryBlock.getArgument(0);
  SmallVector<Value, 3> originalWorkgroupCount;
  for (int i = 0; i < dispatchState.workgroupCount.size(); ++i) {
    originalWorkgroupCount.push_back(entryBlock.getArgument(1 + i));
  }

  auto builder = OpBuilder::atBlockBegin(&entryBlock);
  auto remappedWorkgroupCount = calculateDispatchWorkgroupCount(
      loc, dispatchState.executableOp, dispatchState.entryPointOp,
      originalWorkgroupCount, builder);
  builder.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
      loc, commandBuffer, dispatchState.entryPointOp, remappedWorkgroupCount[0],
      remappedWorkgroupCount[1], remappedWorkgroupCount[2]);

  builder.create<IREE::HAL::ReturnOp>(loc);
  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
