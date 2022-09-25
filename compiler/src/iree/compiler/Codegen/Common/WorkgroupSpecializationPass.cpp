// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== WorkgroupSpecializationPass.cpp ----===//
//
// This pass specializes the workgroup distribution loops with the tile sizes.
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-workgroup-specialization"

namespace mlir {
namespace iree_compiler {

namespace {
struct WorkgroupSpecializationPass
    : public WorkgroupSpecializationBase<WorkgroupSpecializationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void WorkgroupSpecializationPass::runOnOperation() {
  auto funcOp = getOperation();

  SmallVector<Operation *> computeOps;
  SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
  if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
    funcOp.emitOpError("failed to get compute ops in dispatch");
    return signalPassFailure();
  }
  if (!tiledLoops.empty()) {
    // The entry point already has distribution to workgroups. Do nothing.
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createWorkgroupSpecializationPass() {
  return std::make_unique<WorkgroupSpecializationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
