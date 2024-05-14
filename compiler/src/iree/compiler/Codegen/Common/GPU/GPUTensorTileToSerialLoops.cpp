// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUTENSORTILETOSERIALLOOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUTensorTileToSerialLoopsPass final
    : impl::GPUTensorTileToSerialLoopsPassBase<GPUTensorTileToSerialLoopsPass> {
  void runOnOperation() override {
    // Tile reductions based on the annotated tiling configuration.
    if (failed(tileReductionToSerialLoops(getOperation(),
                                          /*fuseInputProducer=*/true))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
