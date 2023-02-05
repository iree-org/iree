// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#define DEBUG_TYPE "iree-gpu-coalasced-access"

namespace mlir {
namespace iree_compiler {

namespace {

/// Finds stride-1 access on genericOp's inputs and set
/// DescriptorFlags::Streaming flag. This data will not need caching since it is
/// read by cache-line and once. perfectly coalasced data.
void setStreamingDataflag(linalg::GenericOp genericOp) {
  auto inputs = genericOp.getInputs();
  for (auto [idx, indexingMap] :
       llvm::enumerate(genericOp.getIndexingMapsArray())) {
    if (inputs.size() <= idx) break;
    int numRes = indexingMap.getNumResults() - 1;
    // Traverse readonly inputs if their indexing is stride-1 mark streaming
    // flag. Check the last dimension, for example (..., dn) -> (..., dn)
    if (indexingMap.getNumDims() >= numRes &&
        indexingMap.getDimPosition(numRes) == numRes) {
      if (auto loadOp =
              inputs[idx].getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
        if (auto subspanOp =
                loadOp.getSource()
                    .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>()) {
          auto flags = IREE::HAL::DescriptorFlags::Streaming;
          if (subspanOp.getDescriptorFlags().has_value())
            flags = flags | subspanOp.getDescriptorFlags().value();
          subspanOp.setDescriptorFlags(flags);
        }
      }
    }
  }
}

struct GPUCoalascedAccessPass
    : public GPUCoalascedAccessBase<GPUCoalascedAccessPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp->walk(
        [&](linalg::GenericOp genericOp) { setStreamingDataflag(genericOp); });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUCoalascedAccessPass() {
  return std::make_unique<GPUCoalascedAccessPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
