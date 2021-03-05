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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

static llvm::cl::opt<int> repeatDispatchNum{
    "iree-hal-repeat-dispatch-num",
    llvm::cl::desc("The number of times to repeat dispatches."),
    llvm::cl::init(1),
};

// A pass converting the IREE flow dialect into the IREE HAL dialect.
class RepeatDispatchesPass
    : public PassWrapper<RepeatDispatchesPass, OperationPass<FuncOp>> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<HALDialect, StandardOpsDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (repeatDispatchNum == 1) return;
    FuncOp f = getOperation();
    f.walk([&](HAL::CommandBufferDispatchOp op) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      Value zero = builder.create<ConstantIndexOp>(loc, 0);
      Value one = builder.create<ConstantIndexOp>(loc, 1);
      Value ub = builder.create<ConstantIndexOp>(loc, repeatDispatchNum);
      auto forOp = builder.create<scf::ForOp>(
          op.getLoc(), zero, ub, one, ValueRange{},
          [&op](OpBuilder& b, Location l, Value v, ValueRange vr) {
            b.clone(*(op.getOperation()));
            b.create<scf::YieldOp>(l, ValueRange{});
          });
      op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createRepeatDispatchesPass() {
  return std::make_unique<RepeatDispatchesPass>();  // NOLINT
}

static PassRegistration<RepeatDispatchesPass> pass("iree-hal-repeat-dispatches",
                                                   "Repeat all the dispatches");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
