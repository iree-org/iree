// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-fuse-dispatch-bindings"

namespace mlir::iree_compiler::IREE::Stream {
namespace {

//===----------------------------------------------------------------------===//
// Fusion
//===----------------------------------------------------------------------===//

// NOTE: invalid once the dispatch is mutated.
struct BindingRange {
  BindingRange() = default;
  BindingRange(IREE::Stream::CmdDispatchOp dispatchOp, unsigned idx)
      : idx(idx), access(llvm::cast<IREE::Stream::ResourceAccessBitfieldAttr>(
                             dispatchOp.getResourceAccesses()[idx])
                             .getValue()),
        resource(dispatchOp.getResources()[idx]),
        resourceSize(dispatchOp.getResourceSizes()[idx]),
        offset(dispatchOp.getResourceOffsets()[idx]),
        length(dispatchOp.getResourceLengths()[idx]) {}

  unsigned idx = 0;
  IREE::Stream::ResourceAccessBitfield access =
      IREE::Stream::ResourceAccessBitfield::None;
  Value resource;
  Value resourceSize;
  Value offset;
  Value length;
};

struct Binding {
  // All resource ranges bound to the binding across the entire program.
  mutable SmallVector<BindingRange> ranges;
  // Ranges for a specific dispatch site.
  mutable DenseMap<Operation *, SmallVector<BindingRange>> sites;

  // One bit per binding that alias each other.
  llvm::BitVector correlationMap;

  // An access bitfield with a union of all range accesses.
  IREE::Stream::ResourceAccessBitfield derivedAccess =
      IREE::Stream::ResourceAccessBitfield::None;
};

// Builds a set of fused bindings based on dispatches.
// Each dispatch may have a unique binding set and we conservatively fuse only
// those we can prove are the same. We could in the future introduce new entry
// points if we had minor divergence in order to gain more fusion in the common
// cases.
static SmallVector<Binding>
findCorrelatedBindings(unsigned bindingCount,
                       ArrayRef<IREE::Stream::CmdDispatchOp> dispatchOps,
                       bool aliasMutableBindings) {
  // For each dispatch build equivalence classes indicating which bindings are
  // from the same base resource. Note that not all dispatches will have the
  // same duplicate bindings (though we hope they do!).
  SmallVector<llvm::EquivalenceClasses<unsigned>> ecs;
  ecs.reserve(dispatchOps.size());
  for (auto dispatchOp : dispatchOps) {
    llvm::EquivalenceClasses<unsigned> ec;
    DenseMap<Value, unsigned> leaders;
    for (auto [idx, resource, resourceAccessAttr] : llvm::enumerate(
             dispatchOp.getResources(), dispatchOp.getResourceAccesses())) {
      // If the resource is mutable and we were told not to alias mutable
      // bindings we always put the resource into its own class.
      auto resourceAccess =
          llvm::cast<IREE::Stream::ResourceAccessBitfieldAttr>(
              resourceAccessAttr);
      if (!aliasMutableBindings &&
          bitEnumContainsAll(resourceAccess.getValue(),
                             IREE::Stream::ResourceAccessBitfield::Write)) {
        ec.insert(idx);
        leaders.insert(std::make_pair(resource, idx));
        continue;
      }

      // Find or create a class for equivalent aliasable resource bindings.
      auto ecIt = leaders.find(resource);
      if (ecIt == leaders.end()) {
        // New unique value.
        ec.insert(idx);
        leaders.insert(std::make_pair(resource, idx));
      } else {
        // Found existing; union with leader.
        ec.unionSets(ecIt->second, idx);
      }
    }
    ecs.push_back(std::move(ec));
  }

  // For each binding produce a bitmap indicating aliasing bindings.
  // This allows us to quickly see for any given binding which ones we know are
  // consistently correlated across all dispatches.
  SmallVector<llvm::BitVector> bindingCorrelationMap;
  bindingCorrelationMap.resize(bindingCount);
  llvm::BitVector tempBits(bindingCount, /*t=*/false);
  for (unsigned i = 0; i < bindingCount; ++i) {
    // Set bits to 1 when they share a set with binding i.
    // We do this by starting with all equivalent and then ANDing away
    // divergences.
    llvm::BitVector bits(bindingCount, /*t=*/true);
    for (auto &ec : ecs) {
      tempBits.reset();
      for (auto mit = ec.findLeader(i); mit != ec.member_end(); ++mit) {
        tempBits.set(*mit);
      }
      bits &= tempBits;
    }
    bindingCorrelationMap[i] = std::move(bits);
  }

  LLVM_DEBUG({
    for (unsigned i = 0; i < bindingCount; ++i) {
      llvm::dbgs() << "binding " << i << " correlation: ";
      llvm::interleaveComma(bindingCorrelationMap[i].set_bits(), llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  SmallVector<Binding> bindings;
  llvm::BitVector handledBindings(bindingCount, /*t=*/false);
  for (unsigned i = 0; i < bindingCount; ++i) {
    // Ignore bindings we've already covered earlier during iteration.
    if (handledBindings.test(i))
      continue;

    // Build new binding.
    Binding binding;
    binding.correlationMap = bindingCorrelationMap[i];
    for (unsigned j : bindingCorrelationMap[i].set_bits()) {
      assert(!handledBindings.test(j));
      handledBindings.set(j);
      for (auto dispatchOp : dispatchOps) {
        auto range = BindingRange(dispatchOp, j);
        binding.ranges.push_back(range);
        binding.sites[dispatchOp].push_back(range);
        binding.derivedAccess = binding.derivedAccess | range.access;
      }
    }
    bindings.push_back(binding);
  }
  return bindings;
}

// Updates an executable function to use the new bindings.
static void updateExecutableSignature(IREE::Stream::ExecutableOp executableOp,
                                      IREE::Stream::ExecutableExportOp exportOp,
                                      mlir::func::FuncOp funcOp,
                                      ArrayRef<Binding> bindings) {
  auto &entryBlock = funcOp.front();

  // Gather old bindings (in order).
  SmallVector<BlockArgument> oldBindingArgs;
  for (auto arg : entryBlock.getArguments()) {
    if (llvm::isa<IREE::Stream::BindingType>(arg.getType())) {
      oldBindingArgs.push_back(arg);
    }
  }

  // Insert new binding args before the old ones (because that's easier).
  // Since we need to do live replacement of the old arg values we can't
  // erase them yet.
  SmallVector<BlockArgument> newBindingArgs;
  auto bindingType = IREE::Stream::BindingType::get(funcOp.getContext());
  auto offsetType = IndexType::get(funcOp.getContext());
  for (auto &binding : bindings) {
    SmallVector<Location> locs;
    for (unsigned oldIdx : binding.correlationMap.set_bits()) {
      locs.push_back(oldBindingArgs[oldIdx].getLoc());
    }
    auto loc = FusedLoc::get(funcOp.getContext(), locs);
    auto bindingArg =
        entryBlock.insertArgument(newBindingArgs.size(), bindingType, loc);
    newBindingArgs.push_back(bindingArg);
  }

  // Replace uses of the old args with the new args and update the ranges.
  unsigned offsetIdx = newBindingArgs.back().getArgNumber() + 1;
  for (auto binding : llvm::enumerate(bindings)) {
    auto newBindingArg = newBindingArgs[binding.index()];
    for (unsigned oldIdx : binding.value().correlationMap.set_bits()) {
      auto oldBindingArg = oldBindingArgs[oldIdx];
      auto offsetArg = entryBlock.insertArgument(offsetIdx++, offsetType,
                                                 newBindingArg.getLoc());
      for (auto &use : llvm::make_early_inc_range(oldBindingArg.getUses())) {
        if (auto subspanOp =
                dyn_cast<IREE::Stream::BindingSubspanOp>(use.getOwner())) {
          OpBuilder builder(subspanOp);
          Value offsetSum = offsetArg;
          if (!mlir::matchPattern(subspanOp.getByteOffset(), m_Zero())) {
            offsetSum = builder.createOrFold<arith::AddIOp>(
                newBindingArg.getLoc(), subspanOp.getByteOffset(), offsetSum);
          }
          subspanOp.getByteOffsetMutable().assign(offsetSum);
        }
        use.set(newBindingArg);
      }
    }
  }

  // Erase old binding arguments (they should all be unused).
  entryBlock.eraseArguments([&](BlockArgument arg) {
    return llvm::is_contained(oldBindingArgs, arg);
  });

  // Be lazy with updating the signature by just reading back what we did.
  funcOp.setType(FunctionType::get(funcOp.getContext(),
                                   entryBlock.getArgumentTypes(), {}));
}

// Memoization of constant 0 (that we insert a lot) with special handling for
// insertion outside of the parent stream.cmd.execute region.
struct MemoizedCmdZeros {
  DenseMap<Operation *, Value> parentZeros;
  Value getForOp(Operation *op) {
    auto parentOp = op->getParentOfType<IREE::Stream::CmdExecuteOp>();
    auto it = parentZeros.find(parentOp);
    if (it != parentZeros.end()) {
      return it->second;
    }
    auto zero =
        OpBuilder(parentOp).create<arith::ConstantIndexOp>(op->getLoc(), 0);
    parentZeros[parentOp] = zero;
    return zero;
  }
};

// Updates each stream.cmd.dispatch site to use the new binding scheme.
static void updateDispatchSite(IREE::Stream::CmdDispatchOp dispatchOp,
                               ArrayRef<Binding> bindings,
                               MemoizedCmdZeros &memoizedZeros) {
  auto zero = memoizedZeros.getForOp(dispatchOp);

  // Compute the new binding set with any additional operands we may insert to
  // track offsets.
  SmallVector<Value> newResources;
  SmallVector<Value> newResourceSizes;
  SmallVector<Value> newOffsets;
  SmallVector<Value> newLengths;
  SmallVector<Attribute> newAccesses;
  SmallVector<Value> newOperands;
  for (auto &binding : bindings) {
    auto &ranges = binding.sites[dispatchOp];

    // New binding resource is uniform across all the ranges.
    // Note that a fused readonly and writeonly will become a readwrite.
    auto anyRange = ranges.front();
    newResources.push_back(anyRange.resource);
    newResourceSizes.push_back(anyRange.resourceSize);
    newAccesses.push_back(IREE::Stream::ResourceAccessBitfieldAttr::get(
        dispatchOp.getContext(), binding.derivedAccess));

    // Add operands for each old offset.
    // We could be more selective about what we add but doing it like this and
    // relying on dispatch site specialization allows us to reuse that pass to
    // better deduplicate and inline values.
    for (auto &range : ranges) {
      newOperands.push_back(range.offset);
    }

    // New binding has full resource range. We could use min/max to get a
    // tighter range but (today) we don't have a need for that.
    newOffsets.push_back(zero);
    newLengths.push_back(anyRange.resourceSize);
  }

  // Add the original operands that we push to the end.
  newOperands.append(dispatchOp.getUniformOperands().begin(),
                     dispatchOp.getUniformOperands().end());

  // Replace the old dispatch op with a new one.
  OpBuilder builder(dispatchOp);
  auto newOp = builder.create<IREE::Stream::CmdDispatchOp>(
      dispatchOp.getLoc(), dispatchOp.getWorkload(),
      dispatchOp.getEntryPointsAttr(), newOperands, newResources,
      newResourceSizes, newOffsets, newLengths,
      builder.getArrayAttr(newAccesses));
  (void)newOp;
  LLVM_DEBUG({
    llvm::dbgs() << "updated dispatch:\n";
    newOp.dump();
  });
  dispatchOp.erase();
}

// Fuses bindings on an |exportOp| based on all |dispatchOps| invoking it.
static void
fuseDispatchBindings(IREE::Stream::ExecutableOp executableOp,
                     IREE::Stream::ExecutableExportOp exportOp,
                     ArrayRef<IREE::Stream::CmdDispatchOp> dispatchOps,
                     MemoizedCmdZeros &memoizedZeros) {
  if (dispatchOps.empty())
    return; // no-op if no dispatches
  auto anyDispatchOp = dispatchOps.front();
  unsigned bindingCount = anyDispatchOp.getResources().size();

  auto configAttr = IREE::Stream::ResourceConfigAttr::lookup(exportOp);
  bool aliasMutableBindings = configAttr.getAliasMutableBindings();

  LLVM_DEBUG({
    AsmState asmState(executableOp->getParentOp());
    llvm::dbgs() << "---- fuseDispatchBindings(@" << executableOp.getSymName()
                 << "::" << exportOp.getSymName() << ") ----\n";
    llvm::dbgs() << "using dispatches:\n";
    for (auto dispatchOp : dispatchOps) {
      dispatchOp.print(llvm::dbgs(), asmState);
      llvm::dbgs() << "\n";
    }
  });

  // Analysis to find which bindings we can fuse together based on dispatches.
  auto bindings =
      findCorrelatedBindings(bindingCount, dispatchOps, aliasMutableBindings);

  // TODO(benvanik): canonicalize bindings and bail early here. Today this
  // rebasing will widen access modes and pass in the offset across the bindings
  // such that they will often be redundant later on during descriptor updating
  // and allow us to elide some updates. A real canonicalization pass should do
  // this instead as well as reordering the bindings.
  // if (bindings.size() == bindingCount) {
  //   LLVM_DEBUG(llvm::dbgs() << " (no change)\n");
  //   return;
  // }
  LLVM_DEBUG({
    if (bindings.size() == bindingCount) {
      llvm::dbgs()
          << " (no change, but rebasing to 0 and changing access mode)\n";
    }
  });

  LLVM_DEBUG({
    AsmState asmState(executableOp->getParentOp());
    llvm::dbgs() << "updated binding set:\n";
    for (auto binding : llvm::enumerate(bindings)) {
      llvm::dbgs() << " binding " << binding.index() << ":\n";
      for (auto &range : binding.value().ranges) {
        llvm::dbgs() << "  src " << range.idx << ": ";
        range.resource.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "[";
        range.offset.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << " for ";
        range.length.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "] : ";
        range.resource.getType().print(llvm::dbgs());
        llvm::dbgs() << "{";
        range.resourceSize.printAsOperand(llvm::dbgs(), asmState);
        llvm::dbgs() << "}\n";
      }
    }
  });

  // TODO(benvanik): some special handling for finding least-common-denominator
  // or base values for constants passed in. We can end up with a lot of
  // subranges into transient or constant resources that are all relatively
  // correlated:
  //   operand[0]: @storage0: offset 100
  //   operand[1]: @storage0: offset 200
  //   operand[2]: @storage0: offset 300
  // ->
  //   operand[0]: @storage0: offset +0
  //   operand[1]: @storage0: offset +100
  //   operand[2]: @storage0: offset +200
  // Identifying these and using those relative offsets would make it cheaper to
  // inline the values into the executables as they are not dispatch site
  // specific. We'd do this by going per range and finding uniformly constant
  // values. See the old MaterializeInterfaces.cpp pass for an earlier
  // implementation that was special-cased for constant buffers only: here we
  // can do it for everything.

  // Update the executable function to use the new bindings.
  auto funcOp = exportOp.lookupFunctionRef();
  assert(funcOp && "entry func not found");
  updateExecutableSignature(executableOp, exportOp, funcOp, bindings);

  // Update each dispatch site to pass the new bindings and operands.
  // NOTE: this invalidates the bindings data structure!
  for (auto dispatchOp : dispatchOps) {
    updateDispatchSite(dispatchOp, bindings, memoizedZeros);
  }
  bindings.clear(); // invalidated above
}

//===----------------------------------------------------------------------===//
// -iree-stream-fuse-dispatch-bindings
//===----------------------------------------------------------------------===//

class FuseDispatchBindingsPass
    : public FuseDispatchBindingsBase<FuseDispatchBindingsPass> {
public:
  FuseDispatchBindingsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
  }

  // TODO(benvanik): preserve the information we are eliding by inserting
  // appropriate memory ops. On devices that require prefetching and other
  // nasty things we want to pass along as fine-grained of information as
  // possible. For example, if we fused bindings addressing ranges 0-32 and
  // 2000000-2000032 into one doing 0-2000032 we've lost the ability to tell
  // the target what we need in memory. A majority of hardware in existence has
  // an MMU or a flat uniformly accessible address space and doesn't care but
  // existing ML accelerators are ... what they are. The important part is that
  // we have the information we need here and can find ways of plumbing it down
  // as we find ourselves caring.
  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    // Find all dispatches and bucket by their target entry point.
    DenseMap<Operation *, SmallVector<IREE::Stream::CmdDispatchOp>>
        entryDispatchMap;
    getOperation()->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
      dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
        auto exportOp =
            symbolTable.lookupNearestSymbolFrom(dispatchOp, entryPointAttr);
        entryDispatchMap[exportOp].push_back(dispatchOp);
      });
    });

    // Perform fusion for each executable entry point using all known dispatches
    // as source material.
    MemoizedCmdZeros memoizedZeros;
    for (auto executableOp :
         getOperation().getBodyRegion().getOps<IREE::Stream::ExecutableOp>()) {
      if (!executableOp.getInnerModule())
        continue;
      for (auto exportOp :
           executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
        fuseDispatchBindings(executableOp, exportOp, entryDispatchMap[exportOp],
                             memoizedZeros);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createFuseDispatchBindingsPass() {
  return std::make_unique<FuseDispatchBindingsPass>();
}

} // namespace mlir::iree_compiler::IREE::Stream
