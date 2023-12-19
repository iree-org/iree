// Copyright 2023 The IREE Authors
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

// Replaces a vm.rodata.table op with two vm.rodata.inline ops, one for the
// indexing table, and the second for the padded data.
template <typename IntTy>
static void reifyRodataTable(RewriterBase &rewriter,
                             IREE::VM::RodataTableOp tableOp) {
  SmallVector<IntTy> table;
  SmallVector<Attribute> dataAttrs;
  size_t dataSize = 0;
  size_t dataAlignment =
      tableOp.getDataAlignment() ? *tableOp.getDataAlignment() : 1;
  for (auto value : tableOp.getDataArray().getValue()) {
    auto serializableAttr =
        llvm::cast<IREE::Util::SerializableAttrInterface>(value);
    size_t storageSize = serializableAttr.getStorageSize();
    dataAttrs.push_back(value);

    // Pad to the (byte) data alignment.
    size_t padding =
        (dataAlignment - storageSize % dataAlignment) % dataAlignment;
    if (padding) {
      SmallVector<int8_t> zeros(padding, 0);
      VectorType paddingType = VectorType::get({static_cast<int64_t>(padding)},
                                               rewriter.getIntegerType(8));
      dataAttrs.push_back(rewriter.getZeroAttr(paddingType));
    }

    // The running data size is the offset of the current value.
    table.push_back(dataSize);
    // The table specifies the (unpadded) storage size for this element.
    table.push_back(storageSize);

    // Increment the total storage size by the (padded) storage size.
    dataSize += storageSize + padding;
  }

  auto refType =
      IREE::VM::RefType::get(rewriter.getType<IREE::VM::BufferType>());
  IREE::VM::RodataInlineOp tableRodata;
  if constexpr (std::is_same<IntTy, int32_t>()) {
    tableRodata = rewriter.create<IREE::VM::RodataInlineOp>(
        tableOp.getLoc(), refType, rewriter.getI32VectorAttr(table));
  } else {
    tableRodata = rewriter.create<IREE::VM::RodataInlineOp>(
        tableOp.getLoc(), refType, rewriter.getI64VectorAttr(table));
  }
  if (auto tableNameAttr = tableOp.getTableNameAttr()) {
    tableRodata.setNameAttr(tableNameAttr);
  }

  auto dataRodata = rewriter.create<IREE::VM::RodataInlineOp>(
      tableOp.getLoc(), refType,
      IREE::Util::CompositeAttr::get(rewriter.getContext(), dataAttrs));
  if (auto dataNameAttr = tableOp.getDataNameAttr()) {
    dataRodata.setNameAttr(dataNameAttr);
  }

  if (auto alignmentAttr = tableOp.getAlignmentAttr()) {
    tableRodata.setAlignmentAttr(alignmentAttr);
    dataRodata.setAlignmentAttr(alignmentAttr);
  }
  if (auto mimeTypeAttr = tableOp.getMimeTypeAttr()) {
    tableRodata.setMimeTypeAttr(mimeTypeAttr);
    dataRodata.setMimeTypeAttr(mimeTypeAttr);
  }
  rewriter.replaceOp(tableOp, {tableRodata, dataRodata});
}

class ReifyRodataTablesPass
    : public PassWrapper<ReifyRodataTablesPass,
                         OperationPass<IREE::VM::ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override {
    return "iree-vm-reify-rodata-tables";
  }

  StringRef getDescription() const override {
    return "Converts vm.rodata.table into two rodata, one for the flat data and"
           "the other for a newly constructed table for the element subviews.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Walk all of the rodata table ops and convert to rodata.inline
    IRRewriter rewriter(moduleOp.getContext());
    moduleOp.walk([&](IREE::VM::RodataTableOp tableOp) {
      rewriter.setInsertionPoint(tableOp);
      size_t tableBitwidth = tableOp.getTableType().getIntOrFloatBitWidth();
      if (tableBitwidth == 32) {
        reifyRodataTable<int32_t>(rewriter, tableOp);
      } else if (tableBitwidth == 64) {
        reifyRodataTable<int64_t>(rewriter, tableOp);
      } else {
        llvm_unreachable("Invalid table bit width");
      }
    });
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createReifyRodataTablesPass() {
  return std::make_unique<ReifyRodataTablesPass>();
}

static PassRegistration<ReifyRodataTablesPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
