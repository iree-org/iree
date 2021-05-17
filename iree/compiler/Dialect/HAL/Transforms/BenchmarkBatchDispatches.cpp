// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

// A pass converting the IREE flow dialect into the IREE HAL dialect.
class BenchmarkBatchDispatchesPass
    : public PassWrapper<BenchmarkBatchDispatchesPass, OperationPass<FuncOp>> {
 public:
  explicit BenchmarkBatchDispatchesPass(unsigned repeatCount)
      : repeatCount_(repeatCount) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<HALDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    FuncOp f = getOperation();
    SmallVector<HAL::CommandBufferDispatchOp> ops;
    f.walk([&](HAL::CommandBufferDispatchOp op) { ops.push_back(op); });

    for (auto op : ops) {
      OpBuilder builder(op);
      for (unsigned i = 1; i < repeatCount_; ++i) {
        builder.clone(*op.getOperation());
        // Add a barrier after each clone. If the original dispatch has a small
        // problem size, simply duplicating without barrier will increase the
        // number of subgroups and thus "help" filling the GPU. In the end we
        // will have an over optimistic result. Inserting barriers avoids that,
        // but it assumes that the command buffer has a linear dispatch
        // structure.
        builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
            op.getLoc(), op.command_buffer(),
            IREE::HAL::ExecutionStageBitfield::CommandRetire |
                IREE::HAL::ExecutionStageBitfield::Dispatch,
            IREE::HAL::ExecutionStageBitfield::CommandIssue |
                IREE::HAL::ExecutionStageBitfield::Dispatch,
            IREE::HAL::ExecutionBarrierFlagBitfield::None);
      }
    }
  }

 private:
  unsigned repeatCount_;
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createBenchmarkBatchDispatchesPass(
    unsigned repeatCount) {
  return std::make_unique<BenchmarkBatchDispatchesPass>(repeatCount);
}

static PassRegistration<BenchmarkBatchDispatchesPass> pass(
    "test-iree-hal-benchmark-batch-dispatches-2-times",
    "Test pass used for benchmarking batch dispatches analysis",
    [] { return std::make_unique<BenchmarkBatchDispatchesPass>(2); });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
