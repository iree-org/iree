// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  // TranslateExecutableVariantsPass. Pass registery is also staticly
  // initialized, so targetBackendsFlags needs to be here to be initialized
  // first.
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

// Renames |op| within |moduleOp| with a new name that is unique within both
// |moduleOp| and |optionalSymbolTable| (if one is provided).
static void renameWithDisambiguatedName(
    Operation *op, Operation *moduleOp,
    DenseMap<StringRef, Operation *> &targetSymbolMap,
    SymbolTable *optionalSymbolTable) {
  StringRef originalName = SymbolTable::getSymbolName(op);

  // Iteratively try suffixes until we find one that isn't used.
  std::string disambiguatedName;
  int uniqueingCounter = 0;
  do {
    disambiguatedName =
        llvm::formatv("{0}_{1}", originalName, uniqueingCounter++).str();
  } while (
      targetSymbolMap.lookup(disambiguatedName) ||
      (optionalSymbolTable && optionalSymbolTable->lookup(disambiguatedName)));

  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUsers(symbolTable, moduleOp);
  symbolUsers.replaceAllUsesWith(op, disambiguatedName);
  SymbolTable::setSymbolName(op, disambiguatedName);
}

void TargetBackend::declareVariantOps(IREE::Flow::ExecutableOp sourceOp,
                                      IREE::HAL::ExecutableOp executableOp) {
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  auto targetContainerOp = targetBuilder.create<IREE::HAL::ExecutableVariantOp>(
      sourceOp.getLoc(), name(), filter_pattern());
  OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
  containerBuilder.create<ModuleOp>(sourceOp.getLoc());
}

// Destructively merges |sourceModuleOp| into |targetModuleOp|.
// |targetSymbolMap| is updated with the new symbols.
//
// If a private symbol in |sourceModuleOp| conflicts with another symbol
// (public or private) tracked in |targetSymbolMap|, it will be renamed.
//
// Fails if a public symbol in |sourceModuleOp| conflicts with another public
// symbol tracked in |targetSymbolMap|.
static LogicalResult mergeModuleInto(
    Operation *sourceModuleOp, Operation *targetModuleOp,
    DenseMap<StringRef, Operation *> &targetSymbolMap) {
  auto &sourceBlock = sourceModuleOp->getRegion(0).front();
  auto &targetBlock = targetModuleOp->getRegion(0).front();
  SymbolTable sourceSymbolTable(sourceModuleOp);
  auto allOps = llvm::to_vector<8>(
      llvm::map_range(sourceBlock, [&](Operation &op) { return &op; }));

  for (auto &op : allOps) {
    if (op->hasTrait<OpTrait::IsTerminator>()) continue;
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
      auto symbolName = symbolOp.getName();

      // Resolve symbol name conflicts.
      if (auto targetOp = targetSymbolMap[symbolName]) {
        if (symbolOp.getVisibility() == SymbolTable::Visibility::Private) {
          // Private symbols can be safely folded into duplicates or renamed.
          if (OperationEquivalence::isEquivalentTo(targetOp, op)) {
            // Optimization: skip over duplicate private symbols.
            // We could let CSE do this later, but we may as well check here.
            continue;
          } else {
            // Preserve the op but give it a unique name.
            renameWithDisambiguatedName(op, sourceModuleOp, targetSymbolMap,
                                        &sourceSymbolTable);
          }
        } else {
          // The source symbol has 'nested' or 'public' visibility.
          if (SymbolTable::getSymbolVisibility(targetOp) !=
              SymbolTable::Visibility::Private) {
            // Oops! Both symbols are public and we can't safely rename either.
            // If you hit this with ops that you think are safe to rename, mark
            // them private.
            //
            // Note: we could also skip linking between executables with
            // conflicting symbol names. We think such conflicts will be better
            // fixed in other ways, so we'll emit an error until we find a case
            // where that isn't true.
            return op->emitError()
                   << "multiple public symbols with the name: " << symbolName;
          } else {
            // Keep the original name for our new op, rename the target op.
            renameWithDisambiguatedName(targetOp, targetModuleOp,
                                        targetSymbolMap,
                                        /*optionalSymbolTable=*/nullptr);
          }
        }
      }
      targetSymbolMap[SymbolTable::getSymbolName(op)] = op;
    }
    if (!targetBlock.empty() &&
        targetBlock.back().hasTrait<OpTrait::IsTerminator>()) {
      op->moveBefore(&targetBlock.back());
    } else {
      op->moveBefore(&targetBlock, targetBlock.end());
    }
  }

  // Now that we're done cloning its ops, delete the original target op.
  sourceModuleOp->erase();

  return success();
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
    IREE::HAL::ExecutableVariantOp linkedTargetOp,
    std::function<Operation *(mlir::ModuleOp moduleOp)> getInnerModuleFn,
    OpBuilder &builder) {
  llvm::SmallVector<IREE::HAL::InterfaceOp, 4> linkedInterfaceOps;
  int nextEntryPointOrdinal = 0;
  DenseMap<StringRef, Operation *> targetSymbolMap;
  DenseMap<Attribute, Attribute> entryPointRefReplacements;

  auto linkedExecutableBuilder =
      OpBuilder::atBlockBegin(linkedExecutableOp.getBody());
  auto linkedTargetBuilder = OpBuilder::atBlockBegin(linkedTargetOp.getBody());
  auto linkedModuleOp = getInnerModuleFn(linkedTargetOp.getInnerModule());

  // Iterate over all source executable ops, linking as many as we can.
  for (auto sourceExecutableOp : sourceExecutableOps) {
    auto variantOps = llvm::to_vector<4>(
        sourceExecutableOp.getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      // Only process targets matching our pattern.
      if (!matchPattern(variantOp.target_backend_filter(), filter_pattern())) {
        continue;
      }

      // Clone entry point ops and queue remapping ordinals and updating
      // symbol refs.
      for (auto entryPointOp :
           variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
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
              llvm::formatv("io_{0}", linkedInterfaceOps.size()).str());
          linkedInterfaceOps.push_back(linkedInterfaceOp);
        }

        auto newEntryPointOp =
            linkedTargetBuilder.create<IREE::HAL::ExecutableEntryPointOp>(
                entryPointOp.getLoc(), entryPointOp.sym_nameAttr(),
                builder.getIndexAttr(nextEntryPointOrdinal++),
                builder.getSymbolRefAttr(linkedInterfaceOp.getName()),
                ArrayAttr{}, IntegerAttr{});

        // Add to replacement table for fixing up dispatch calls referencing
        // this entry point.
        auto oldSymbolRefAttr =
            builder.getSymbolRefAttr(sourceExecutableOp.getName(),
                                     {builder.getSymbolRefAttr(variantOp),
                                      builder.getSymbolRefAttr(entryPointOp)});
        auto newSymbolRefAttr = builder.getSymbolRefAttr(
            linkedExecutableOp.getName(),
            {builder.getSymbolRefAttr(linkedTargetOp),
             builder.getSymbolRefAttr(newEntryPointOp)});
        entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      }

      // Merge the existing module into the new linked module op.
      auto sourceModuleOp = getInnerModuleFn(variantOp.getInnerModule());
      if (failed(mergeModuleInto(sourceModuleOp, linkedModuleOp,
                                 targetSymbolMap))) {
        return failure();
      }

      variantOp.erase();
    }

    if (sourceExecutableOp.getOps<IREE::HAL::ExecutableVariantOp>().empty()) {
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
  BlockAndValueMapping bvm;
  for (auto args : llvm::enumerate(workload)) {
    bvm.map(body->getArgument(args.index()), args.value());
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
  auto entryPointSymRef = builder.getSymbolRefAttr(
      dispatchState.executableOp.getName(),
      {builder.getSymbolRefAttr(dispatchState.entryPointOp->getParentOp()),
       builder.getSymbolRefAttr(dispatchState.entryPointOp)});
  auto remappedWorkgroupCount = calculateDispatchWorkgroupCount(
      loc, dispatchState.executableOp, dispatchState.entryPointOp,
      originalWorkgroupCount, builder);
  builder.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
      loc, commandBuffer, entryPointSymRef, remappedWorkgroupCount[0],
      remappedWorkgroupCount[1], remappedWorkgroupCount[2]);

  builder.create<IREE::HAL::ReturnOp>(loc);
  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
