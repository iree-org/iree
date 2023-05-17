// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct InstrumentMemoryAccessesPass
    : InstrumentMemoryAccessesBase<InstrumentMemoryAccessesPass> {
  void runOnOperation() override {
    // Lookup the root instrumentation op. If not present it means the dispatch
    // is not instrumented and we can skip it.
    IREE::HAL::InstrumentWorkgroupOp instrumentOp;
    getOperation().walk([&](IREE::HAL::InstrumentWorkgroupOp op) {
      instrumentOp = op;
      return WalkResult::interrupt();
    });
    if (!instrumentOp) {
      // Not instrumented.
      return;
    }

    auto buffer = instrumentOp.getBuffer();
    auto workgroupKey = instrumentOp.getWorkgroupKey();
    getOperation()->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<memref::LoadOp>([&](auto loadOp) {
            OpBuilder builder(loadOp);
            builder.setInsertionPointAfter(loadOp);
            auto instrumentOp =
                builder.create<IREE::HAL::InstrumentMemoryLoadOp>(
                    loadOp.getLoc(), loadOp.getResult().getType(), buffer,
                    workgroupKey, loadOp.getResult(), loadOp.getMemRef(),
                    loadOp.getIndices());
            loadOp.getResult().replaceAllUsesExcept(instrumentOp.getResult(),
                                                    instrumentOp);
          })
          .Case<memref::StoreOp>([&](auto storeOp) {
            OpBuilder builder(storeOp);
            auto instrumentOp =
                builder.create<IREE::HAL::InstrumentMemoryStoreOp>(
                    storeOp.getLoc(), storeOp.getValueToStore().getType(),
                    buffer, workgroupKey, storeOp.getValueToStore(),
                    storeOp.getMemRef(), storeOp.getIndices());
            storeOp.getValueMutable().assign(instrumentOp.getResult());
          })
          .Case<vector::LoadOp>([&](auto loadOp) {
            OpBuilder builder(loadOp);
            builder.setInsertionPointAfter(loadOp);
            auto instrumentOp =
                builder.create<IREE::HAL::InstrumentMemoryLoadOp>(
                    loadOp.getLoc(), loadOp.getVectorType(), buffer,
                    workgroupKey, loadOp.getResult(), loadOp.getBase(),
                    loadOp.getIndices());
            loadOp.getResult().replaceAllUsesExcept(instrumentOp.getResult(),
                                                    instrumentOp);
          })
          .Case<vector::StoreOp>([&](auto storeOp) {
            OpBuilder builder(storeOp);
            auto instrumentOp =
                builder.create<IREE::HAL::InstrumentMemoryStoreOp>(
                    storeOp.getLoc(), storeOp.getVectorType(), buffer,
                    workgroupKey, storeOp.getValueToStore(), storeOp.getBase(),
                    storeOp.getIndices());
            storeOp.getValueToStoreMutable().assign(instrumentOp.getResult());
          })
          .Default([&](Operation *) {});
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createInstrumentMemoryAccessesPass() {
  return std::make_unique<InstrumentMemoryAccessesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
