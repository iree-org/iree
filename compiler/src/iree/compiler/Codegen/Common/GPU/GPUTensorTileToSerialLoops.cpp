// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-tensor-tile-to-serial-loops"

namespace mlir::iree_compiler {

namespace {
struct GPUTensorTensorTileToSerialLoopsPass final
    : public GPUTensorTileToSerialLoopsBase<
          GPUTensorTensorTileToSerialLoopsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    // Tile reductions based on the annotated tiling configuration.
    if (failed(tileReductionToSerialLoops(getOperation(),
                                          /*fuseInputProducer=*/true))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createGPUTensorTileToSerialLoops() {
  return std::make_unique<GPUTensorTensorTileToSerialLoopsPass>();
}

} // namespace mlir::iree_compiler
