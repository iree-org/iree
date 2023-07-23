// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./Utils.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct WorkGroupSwizzlePass
    : public WorkGroupSwizzleBase<WorkGroupSwizzlePass> {
  WorkGroupSwizzlePass(unsigned swizzleLogTile)
      : swizzleLogTile(swizzleLogTile) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    swizzleLogTile = logTile;
    return success();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    (void)swizzleWorkgroupsInFunc(funcOp, swizzleLogTile);
  }

private:
  unsigned swizzleLogTile;
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createWorkGroupSwizzle(unsigned swizzleLogTile) {
  return std::make_unique<WorkGroupSwizzlePass>(swizzleLogTile);
}

} // namespace iree_compiler
} // namespace mlir
