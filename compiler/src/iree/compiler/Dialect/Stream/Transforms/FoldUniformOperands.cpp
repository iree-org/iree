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

#define DEBUG_TYPE "iree-stream-fold-uniform-operands"

namespace mlir::iree_compiler::IREE::Stream {
namespace {

//===----------------------------------------------------------------------===//
// Per-dispatchable export optimization
//===----------------------------------------------------------------------===//

// Deduplicates operands that have the same value at all dispatch sites.
// This will deduplicate dynamic values as well.
//
// Example:
//   stream.cmd.dispatch @foo(%0, %1, %0 : index, index, index)
//  ->
//   stream.cmd.dispatch @foo(%0, %1 : index, index)
// + deduped arguments in the executable
static void
deduplicateOperands(mlir::func::FuncOp funcOp,
                    SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps) {
  auto &entryBlock = funcOp.front();
  auto anyDispatchOp = dispatchOps.front();
  unsigned operandCount = anyDispatchOp.getUniformOperands().size();

  // Build a map of operand indices to its base duplicate for each dispatch
  // site. Base/non-duplicated values will be identity.
  // Example: (%a, %b, %a, %b) -> (0, 1, 0, 1)
  static const int kUnassigned = -1;
  SmallVector<SmallVector<unsigned>> dupeIndexMaps(dispatchOps.size());
  for (auto dispatchOp : llvm::enumerate(dispatchOps)) {
    auto &dupeIndexMap = dupeIndexMaps[dispatchOp.index()];
    dupeIndexMap.resize(operandCount, kUnassigned);
    auto operands = dispatchOp.value().getUniformOperands();
    for (unsigned i = 0; i < operands.size(); ++i) {
      for (unsigned j = 0; j < i; ++j) {
        if (operands[j] == operands[i]) {
          dupeIndexMap[i] = j;
          break;
        }
      }
    }
  }

  // Per-operand now find which are consistently duplicated.
  llvm::BitVector sameValues(operandCount);
  llvm::BitVector deadOperandsMap(operandCount);
  auto uniformDupeIndexMap =
      llvm::to_vector(llvm::seq(0u, operandCount)); // old -> new
  for (unsigned idx = 0; idx < operandCount; ++idx) {
    if (deadOperandsMap.test(idx))
      continue;
    // Each bit represents an operand that duplicates the operand at idx.
    // We walk all the sites and AND their masks together to get the safe
    // set of duplicate operands.
    // Example for %0: (%a, %b, %a) -> b001
    // Example for %1: (%a, %b, %a) -> b000
    sameValues.set(); // note reused
    for (auto &dupeIndexMap : dupeIndexMaps) {
      for (unsigned i = 0; i < operandCount; ++i) {
        if (i == idx || dupeIndexMap[i] != idx) {
          sameValues.reset(i);
        }
      }
    }
    if (sameValues.none()) {
      uniformDupeIndexMap[idx] = idx;
      continue;
    }
    deadOperandsMap |= sameValues;
    uniformDupeIndexMap[idx] = idx;
    for (auto dupeIdx : sameValues.set_bits()) {
      uniformDupeIndexMap[dupeIdx] = idx;
    }
  }
  if (deadOperandsMap.none()) {
    // No-op.
    return;
  }

  // Build a map of old duplicate arguments to their base arguments.
  auto argReplacementMap =
      llvm::to_vector(llvm::seq(0u, funcOp.getNumArguments())); // old -> new
  auto operandToArgMap =
      IREE::Stream::CmdDispatchOp::makeOperandToArgMap(funcOp);
  for (auto dupe : llvm::enumerate(uniformDupeIndexMap)) {
    unsigned deadIdx = operandToArgMap[dupe.index()];
    unsigned liveIdx = operandToArgMap[dupe.value()];
    argReplacementMap[deadIdx] = liveIdx;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "deduplicateOperands for " << funcOp.getSymName() << "\n";
    llvm::dbgs() << "  dead operands: ";
    llvm::interleaveComma(deadOperandsMap.set_bits(), llvm::dbgs());
    llvm::dbgs() << "\n";
    for (auto replacement : llvm::enumerate(argReplacementMap)) {
      if (replacement.index() == replacement.value())
        continue;
      llvm::dbgs() << "  %arg" << replacement.index() << " -> %arg"
                   << replacement.value() << "\n";
    }
  });

  // Replace uses of the duplicate arguments with their base arguments.
  llvm::BitVector deadArgMap(funcOp.getNumArguments());
  for (auto replacement : llvm::enumerate(argReplacementMap)) {
    unsigned deadIdx = replacement.index();
    unsigned liveIdx = replacement.value();
    if (deadIdx == liveIdx)
      continue;
    deadArgMap.set(deadIdx);
    entryBlock.getArgument(deadIdx).replaceAllUsesWith(
        entryBlock.getArgument(liveIdx));
  }

  // Update each dispatch site to remove duplicates.
  SmallVector<unsigned> deadOperands;
  for (auto idx : deadOperandsMap.set_bits())
    deadOperands.push_back(idx);
  for (auto dispatchOp : dispatchOps) {
    for (auto idx : llvm::reverse(deadOperands)) {
      dispatchOp.getUniformOperandsMutable().erase(idx);
    }
  }

  // Update the function signature.
  // Lame we need two data structures to do this.
  funcOp.setType(funcOp.getTypeWithoutArgsAndResults(deadArgMap, {}));
  entryBlock.eraseArguments(
      [&](BlockArgument arg) { return deadArgMap.test(arg.getArgNumber()); });
}

// Inlines constant values passed in at dispatch sites that are uniform across
// all sites. These may be shape dimensions, resource offsets/sizes, or
// user-provided values that folded to constants.
//
// Example:
//   stream.cmd.dispatch @foo(%c1, %c100 : index, index)
//   stream.cmd.dispatch @foo(%c1, %c101 : index, index)
// ->
//   stream.cmd.dispatch @foo(%c100 : index)
//   stream.cmd.dispatch @foo(%c101 : index)
// + inlined %c1 in the executable
static void
inlineUniformConstants(mlir::func::FuncOp funcOp,
                       SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps) {
  auto &entryBlock = funcOp.front();
  auto anyDispatchOp = dispatchOps.front();
  unsigned operandCount = anyDispatchOp.getUniformOperands().size();

  // Find uniform constant values for each operand across all usages.
  SmallVector<std::optional<APInt>> operandValues(operandCount);
  SmallVector<SmallVector<Location>> operandLocs(operandCount);
  llvm::BitVector uniformOperandMap(operandCount, /*t=*/true);
  for (auto dispatchOp : dispatchOps) {
    for (unsigned idx = 0; idx < operandCount; ++idx) {
      if (!uniformOperandMap.test(idx))
        continue;
      auto value = dispatchOp.getUniformOperands()[idx];
      APInt intValue;
      if (!matchPattern(value, m_ConstantInt(&intValue))) {
        // Non-constant breaks the operand uniformity.
        uniformOperandMap.reset(idx);
        continue;
      }
      if (!operandValues[idx].has_value()) {
        // First constant seen for this operand.
        operandValues[idx] = intValue;
      } else {
        // Ensure uniform constant value with previous occurrances.
        if (operandValues[idx].value() != intValue) {
          uniformOperandMap.reset(idx);
          continue;
        }
      }
      operandLocs[idx].push_back(value.getLoc());
    }
  }
  if (uniformOperandMap.none()) {
    // Early-exit if no-op.
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "inlineUniformConstants for " << funcOp.getSymName()
                 << "\n";
    for (unsigned i = 0; i < operandValues.size(); ++i) {
      if (!operandValues[i].has_value())
        continue;
      llvm::dbgs() << "  operand " << i << " = " << operandValues[i].value()
                   << "\n";
    }
  });

  auto operandToArgMap =
      IREE::Stream::CmdDispatchOp::makeOperandToArgMap(funcOp);

  // Replace uses of the uniform arguments with a constant value.
  llvm::BitVector deadArgMap(funcOp.getNumArguments());
  auto builder = OpBuilder::atBlockBegin(&entryBlock);
  for (auto operandIdx : uniformOperandMap.set_bits()) {
    unsigned argIdx = operandToArgMap[operandIdx];
    auto arg = entryBlock.getArgument(argIdx);
    deadArgMap.set(argIdx);
    auto constantOp = builder.create<arith::ConstantOp>(
        builder.getFusedLoc(operandLocs[operandIdx]),
        builder.getIntegerAttr(arg.getType(),
                               operandValues[operandIdx].value()));
    arg.replaceAllUsesWith(constantOp);
  }

  // Update each dispatch site to remove duplicates.
  SmallVector<unsigned> deadOperands;
  for (auto idx : uniformOperandMap.set_bits())
    deadOperands.push_back(idx);
  for (auto dispatchOp : dispatchOps) {
    for (auto idx : llvm::reverse(deadOperands)) {
      dispatchOp.getUniformOperandsMutable().erase(idx);
    }
  }

  // Fixup function signature.
  funcOp.setType(funcOp.getTypeWithoutArgsAndResults(deadArgMap, {}));
  entryBlock.eraseArguments(
      [&](BlockArgument arg) { return deadArgMap.test(arg.getArgNumber()); });
}

//===----------------------------------------------------------------------===//
// -iree-stream-specialize-dispatches
//===----------------------------------------------------------------------===//

class FoldUniformOperandsPass
    : public FoldUniformOperandsBase<FoldUniformOperandsPass> {
public:
  FoldUniformOperandsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
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

    // Optimize each dispatch op.
    for (auto executableOp :
         getOperation().getBodyRegion().getOps<IREE::Stream::ExecutableOp>()) {
      if (!executableOp.getInnerModule())
        continue;
      for (auto exportOp :
           executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
        auto &dispatchOps = entryDispatchMap[exportOp];
        if (dispatchOps.empty())
          continue; // no-op if no dispatches

        auto funcOp = exportOp.lookupFunctionRef();

        // Deduplicate operands that are correlated at all dispatch sites.
        // We do this first so that we know all constants passed in are unique
        // per dispatch site.
        deduplicateOperands(funcOp, dispatchOps);

        // Inline constants that have the same value at all sites.
        inlineUniformConstants(funcOp, dispatchOps);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldUniformOperandsPass() {
  return std::make_unique<FoldUniformOperandsPass>();
}

} // namespace mlir::iree_compiler::IREE::Stream
