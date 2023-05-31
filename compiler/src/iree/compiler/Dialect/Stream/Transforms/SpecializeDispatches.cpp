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
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-specialize-dispatches"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Per-dispatchable export optimization
//===----------------------------------------------------------------------===//

struct ConstantSet {
  // Type of the set (index, i32, etc).
  Type type;
  // Locations of all constants that went into the table.
  SetVector<Location> locs;
  // Operand index -> all values from dispatch sites.
  SmallVector<std::pair<unsigned, SmallVector<TypedAttr>>> values;
};

struct ConstantTable {
  // Operands that are covered by the constant table and can be removed.
  llvm::BitVector coveredOperands;
  // One set of constants per type.
  SmallVector<ConstantSet> sets;
};

// Builds a constant table composed of unique per-dispatch constant values.
// Each dispatch gets a row in the table that can be selected based on the
// dispatch ordinal.
static ConstantTable buildConstantTable(
    mlir::func::FuncOp funcOp,
    SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps) {
  auto anyDispatchOp = dispatchOps.front();
  unsigned operandCount = anyDispatchOp.getUniformOperands().size();

  // Find which operands are uniformly constants across all usages.
  llvm::BitVector constantOperandMap(operandCount, /*t=*/true);
  for (auto dispatchOp : dispatchOps) {
    for (unsigned idx = 0; idx < operandCount; ++idx) {
      if (!constantOperandMap.test(idx)) continue;
      auto value = dispatchOp.getUniformOperands()[idx];
      Attribute constantValue;
      if (!matchPattern(value, m_Constant(&constantValue))) {
        // Non-constant breaks the operand constant uniformity.
        constantOperandMap.reset(idx);
        continue;
      }
    }
  }
  if (constantOperandMap.none()) {
    // Early-exit if no-op.
    return {};
  }

  // Build constant sets for each type.
  // Note that we need to ensure we build them in a deterministic order so we
  // keep track of the order in which we build the sets per type.
  DenseMap<Type, ConstantSet> typeSets;
  SmallVector<Type> typeOrder;
  for (unsigned idx = 0; idx < operandCount; ++idx) {
    if (!constantOperandMap.test(idx)) continue;
    auto operandType = anyDispatchOp.getUniformOperands()[idx].getType();
    auto &set = typeSets[operandType];
    if (!set.type) {
      set.type = operandType;
      typeOrder.push_back(operandType);
    }
    SmallVector<TypedAttr> values;
    for (auto dispatchOp : dispatchOps) {
      auto operand = dispatchOp.getUniformOperands()[idx];
      TypedAttr constantValue;
      matchPattern(operand, m_Constant(&constantValue));
      values.push_back(constantValue);
      set.locs.insert(operand.getLoc());
    }
    set.values.push_back(std::make_pair(idx, values));
  }

  ConstantTable constantTable;
  constantTable.coveredOperands = constantOperandMap;
  llvm::append_range(
      constantTable.sets,
      llvm::map_range(typeOrder, [&](Type type) { return typeSets[type]; }));
  return constantTable;
}

// Builds a tensor<SITExOPERANDxTYPE> constant attribute.
// This should probably be vector<> but that dialect has some issues with
// expressing basic multi-dimension loads :/
static TypedAttr buildConstantSetAttr(ConstantSet &set, OpBuilder &builder) {
  // TODO(benvanik): better definition of variable-width integers across HAL.
  // HACK: we can't handle index types in a few of the codegen backends (vulkan
  // at least); we convert index -> i32 here but we should probably have a
  // specific "IREE HAL ABI size" type we use instead.
  auto storageType = set.type;
  if (set.type.isIndex()) {
    storageType = builder.getI32Type();
  }

  // Need to invert operand->sites to site->operands.
  int64_t siteCount = static_cast<int64_t>(set.values.front().second.size());
  int64_t valueCount = static_cast<int64_t>(set.values.size());
  SmallVector<Attribute> flattenedAttrs;
  flattenedAttrs.reserve(siteCount * valueCount);
  for (int64_t site = 0; site < siteCount; ++site) {
    for (int64_t value = 0; value < valueCount; ++value) {
      auto valueAttr = set.values[value].second[site];
      if (storageType != valueAttr.getType()) {
        valueAttr = IntegerAttr::get(
            storageType, llvm::cast<IntegerAttr>(valueAttr).getInt());
      }
      flattenedAttrs.push_back(valueAttr);
    }
  }
  auto tensorType = RankedTensorType::get({siteCount, valueCount}, storageType);
  return DenseElementsAttr::get(tensorType, flattenedAttrs);
}

// Inserts a constant table into the given |funcOp| and adds an argument that
// selects which row is used to provide the constant argument values.
// Arguments covered by the constant table are removed from the function.
//
// Note that this trades off increased executable size for decreased runtime
// overhead and less utilization of the limited push constant resources.
// The other benefit is that once the table is inlined there is more potential
// for optimization as all possible constants are known. We may need something
// more sophisticated than just the vector.extracts to make this analysis nice
// though.
//
// This produces constant tables with rows for each site and then extracts the
// densely packed argument from the row:
//   %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<SITExARGxindex>
//   %1 = tensor.extract %0[SITE, ARG]: tensor<SITExARGxindex>
//
// TODO(benvanik): maybe a dedicated lookup table op to make further combining
// easier to do in a backend-generic way.
static void insertConstantTableLookup(mlir::func::FuncOp funcOp,
                                      ConstantTable &constantTable) {
  auto &entryBlock = funcOp.front();
  auto operandToArgMap =
      IREE::Stream::CmdDispatchOp::makeOperandToArgMap(funcOp);

  // Insert site identifier argument (populated by
  // insertDispatchSiteIdentifiers). This is how we know which dispatch is
  // calling us and which table row we need to select.
  auto builder = OpBuilder::atBlockBegin(&entryBlock);
  auto siteId = entryBlock.addArgument(builder.getIndexType(), funcOp.getLoc());

  IndexSet indexSet(funcOp.getLoc(), builder);

  // Build the constant lookup table tensors, one per type.
  SmallVector<Value> tableTensors;
  for (auto &set : constantTable.sets) {
    auto tableAttr = buildConstantSetAttr(set, builder);
    auto tableTensor = builder.create<arith::ConstantOp>(
        builder.getFusedLoc(set.locs.takeVector()), tableAttr);
    tableTensors.push_back(tableTensor);
  }

  // TODO(benvanik): invert this loop so that we preserve argument order.

  // Replace the arguments with lookups into the lookup table tensors.
  for (auto [set, tableTensor] :
       llvm::zip_equal(constantTable.sets, tableTensors)) {
    for (auto operandValues : llvm::enumerate(set.values)) {
      unsigned operandIdx = operandValues.value().first;
      unsigned argIdx = operandToArgMap[operandIdx];
      auto arg = entryBlock.getArgument(argIdx);
      auto extractedValue = builder
                                .create<tensor::ExtractOp>(
                                    arg.getLoc(), tableTensor,
                                    ValueRange{
                                        siteId,
                                        indexSet.get(operandValues.index()),
                                    })
                                .getResult();
      if (extractedValue.getType() != arg.getType()) {
        extractedValue = builder.create<arith::IndexCastOp>(
            arg.getLoc(), arg.getType(), extractedValue);
      }
      arg.replaceAllUsesWith(extractedValue);
    }
  }

  // Fixup function signature.
  llvm::BitVector deadArgMap(funcOp.getNumArguments() + 1);
  for (auto operandIdx : constantTable.coveredOperands.set_bits()) {
    unsigned argIdx = operandToArgMap[operandIdx];
    deadArgMap.set(argIdx);
  }
  funcOp.setType(funcOp.getTypeWithoutArgsAndResults(deadArgMap, {}));
  funcOp.setType(funcOp.getTypeWithArgsAndResults(
      {funcOp.getNumArguments()}, {builder.getIndexType()}, {}, {}));
  entryBlock.eraseArguments(
      [&](BlockArgument arg) { return deadArgMap.test(arg.getArgNumber()); });
}

// Memoization of constants that we insert a lot with special handling for
// insertion outside of the parent stream.cmd.execute region.
struct MemoizedCmdConstants {
  DenseMap<Operation *, DenseMap<int64_t, Value>> parentIndexValues;
  Value getIndexForOp(int64_t value, Operation *op) {
    auto parentOp = op->getParentOfType<IREE::Stream::CmdExecuteOp>();
    auto &parentMap = parentIndexValues[parentOp];
    auto it = parentMap.find(value);
    if (it != parentMap.end()) {
      return it->second;
    }
    auto constantValue =
        OpBuilder(parentOp)
            .create<arith::ConstantIndexOp>(op->getLoc(), value)
            .getResult();
    parentMap.insert({value, constantValue});
    return constantValue;
  }
};

// Inserts a site-unique identifier at each dispatch op that corresponds to its
// row in the constant table. Operands covered by the constant table are removed
// from the dispatch site.
//
// Example:
//   stream.cmd.dispatch @foo(%c100, %c200 : index, index)
//   stream.cmd.dispatch @foo(%c101, %c201 : index, index)
// ->
//   stream.cmd.dispatch @foo(%c0 : index)
//   stream.cmd.dispatch @foo(%c1 : index)
static void insertDispatchSiteIdentifiers(
    SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps,
    ConstantTable &constantTable, MemoizedCmdConstants &memoizedConstants) {
  for (auto it : llvm::enumerate(dispatchOps)) {
    auto dispatchOp = it.value();

    // Remove operands that are covered by the constant table.
    for (int i = constantTable.coveredOperands.size() - 1; i >= 0; --i) {
      if (constantTable.coveredOperands.test(i)) {
        dispatchOp.getUniformOperandsMutable().erase(i);
      }
    }

    // Add the site-unique identifier.
    auto siteId = memoizedConstants.getIndexForOp(it.index(), dispatchOp);
    dispatchOp.getUniformOperandsMutable().append({siteId});
  }
}

// Specializes each dispatchable function based on the way it is dispatched.
// The goal is to reduce the number of operands we pass in dynamically at
// runtime as to stay under device limitations (push constant count in Vulkan
// or the argument buffer size in CUDA) and reduce our overheads (fastest value
// to pass in is the one you don't).
//
// Since we've already deduplicated things we (in theory) don't have to worry
// about introducing divergence. There's potential for later deduping to happen
// while performing a second round of specialization per-backend.
static void specializeDispatches(
    IREE::Stream::ExecutableOp executableOp,
    IREE::Stream::ExecutableExportOp exportOp,
    SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps,
    MemoizedCmdConstants &memoizedConstants) {
  if (dispatchOps.empty()) return;  // no-op if no dispatches

  auto funcOp = exportOp.lookupFunctionRef();

  // Build a constant table for unique per-dispatch constant values.
  auto constantTable = buildConstantTable(funcOp, dispatchOps);
  if (constantTable.coveredOperands.none()) return;

  LLVM_DEBUG({
    AsmState asmState(executableOp->getParentOp());
    llvm::dbgs() << "---- specializeDispatches(@" << executableOp.getSymName()
                 << "::" << exportOp.getSymName() << ") ----\n";
    llvm::dbgs() << "constant table from " << dispatchOps.size()
                 << " dispatches:\n";
    for (auto &set : constantTable.sets) {
      llvm::dbgs() << "  type: ";
      set.type.print(llvm::dbgs());
      llvm::dbgs() << "\n";
      for (auto &operandValues : set.values) {
        llvm::dbgs() << "    operand " << operandValues.first << ":\n      ";
        llvm::interleave(operandValues.second, llvm::dbgs(), "\n      ");
        llvm::dbgs() << "\n";
      }
    }
  });

  // Inline that constant table into the dispatch function and look up the
  // contants to use based on a parameterized input. All unneeded operands
  // are removed.
  insertConstantTableLookup(funcOp, constantTable);

  // Update each dispatch site to remove the constant operands and insert a new
  // per-site identifier passed to the dispatch function.
  insertDispatchSiteIdentifiers(dispatchOps, constantTable, memoizedConstants);
}

//===----------------------------------------------------------------------===//
// -iree-stream-specialize-dispatches
//===----------------------------------------------------------------------===//

class SpecializeDispatchesPass
    : public SpecializeDispatchesBase<SpecializeDispatchesPass> {
 public:
  SpecializeDispatchesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
  }

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

    // Optimize each dispatchable function and its dispatch sites.
    MemoizedCmdConstants memoizedConstants;
    for (auto executableOp :
         getOperation().getBodyRegion().getOps<IREE::Stream::ExecutableOp>()) {
      for (auto exportOp :
           executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
        specializeDispatches(executableOp, exportOp, entryDispatchMap[exportOp],
                             memoizedConstants);
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSpecializeDispatchesPass() {
  return std::make_unique<SpecializeDispatchesPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
