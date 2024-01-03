// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

class DeduplicateRodataPass
    : public PassWrapper<DeduplicateRodataPass,
                         OperationPass<IREE::VM::ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "iree-vm-deduplicate-rodata";
  }

  StringRef getDescription() const override {
    return "Deduplicates vm.rodata ops in the module.";
  }

  using RodataKey = std::tuple<StringRef, Attribute>;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather all rodata ops with the same value.
    DenseMap<RodataKey, SmallVector<IREE::VM::RodataOp>> bucketedOps;
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      if (rodataOp.getOrdinal().has_value()) {
        rodataOp.emitError() << "rodata op already has an ordinal assigned; "
                                "cannot perform deduplication";
        return signalPassFailure();
      }
      RodataKey key = std::make_tuple(rodataOp.getMimeType().value_or(""),
                                      rodataOp.getValue());
      auto &bucketOps = bucketedOps[key];
      bucketOps.push_back(rodataOp);
    }

    DenseMap<SymbolRefAttr, SymbolRefAttr> replacements;
    for (auto bucketKV : bucketedOps) {
      auto &bucketOps = bucketKV.second;

      // Compute the fused location and required alignment based on all rodata
      // ops that we will be deduplicating.
      SmallVector<Location> locs;
      uint64_t alignment = 0;
      for (auto rodataOp : bucketOps) {
        locs.push_back(rodataOp.getLoc());
        alignment = std::max(alignment, rodataOp.getAlignment().value_or(0));
      }
      auto fusedLoc = FusedLoc::get(moduleOp.getContext(), locs);

      // Update the base op that all others will be duplicated into.
      auto baseOp = bucketOps.front();
      bucketOps.erase(bucketOps.begin());
      baseOp->setLoc(fusedLoc);
      if (alignment != 0) {
        baseOp.setAlignmentAttr(IntegerAttr::get(
            IntegerType::get(moduleOp.getContext(), 64), APInt(64, alignment)));
      }

      // Point all duplicates at the base op.
      auto baseName = FlatSymbolRefAttr::get(baseOp.getNameAttr());
      for (auto duplicateOp : bucketOps) {
        replacements.insert(std::make_pair(
            FlatSymbolRefAttr::get(duplicateOp.getSymNameAttr()), baseName));
        duplicateOp.erase();
      }
    }

    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](SymbolRefAttr attr) -> std::pair<Attribute, WalkResult> {
          auto replacement = replacements.find(attr);
          if (replacement != replacements.end())
            return {replacement->getSecond(), WalkResult::skip()};
          return {attr, WalkResult::skip()};
        });
    moduleOp.walk([&](Operation *op) { replacer.replaceElementsIn(op); });
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createDeduplicateRodataPass() {
  return std::make_unique<DeduplicateRodataPass>();
}

static PassRegistration<DeduplicateRodataPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
