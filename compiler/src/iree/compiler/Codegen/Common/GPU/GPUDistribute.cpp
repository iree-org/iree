// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute"

namespace mlir::iree_compiler {

static constexpr int64_t kCudaWarpSize = 32;

/// Hoist allocations to the top of the loop if they have no dependencies.
static void hoistAlloc(mlir::FunctionOpInterface funcOp) {
  SmallVector<memref::AllocOp> allocs;
  funcOp.walk([&](memref::AllocOp alloc) {
    if (alloc.getOperands().empty())
      allocs.push_back(alloc);
  });
  for (memref::AllocOp alloc : allocs) {
    alloc->moveBefore(&(*funcOp.getBlocks().begin()),
                      funcOp.getBlocks().begin()->begin());
  }
}

// struct HoistAllocFromScfIf final
//     : OpRewritePattern<memref::AllocOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(memref::AllocOp allocOp,
//                                 PatternRewriter &rewriter) const override {
//     auto scfIfOp = allocOp->getParentOfType<scf::IfOp>();
//     if (!scfIfOp) {
//       return failure();
//     }
//     if (allocOp.getResult().getUses().empty()) {
//       return failure();
//     }
//     Location loc = scfIfOp.getLoc();
//     llvm::dbgs() << "alloc: " << allocOp << "\n";
//     llvm::dbgs() << "allocOp.getType(): " << allocOp.getType() << "\n";
//     auto newAlloc = rewriter.create<memref::AllocOp>(loc, allocOp.getType());
//     llvm::dbgs() << "newAlloc: " << newAlloc << "\n";
//     llvm::dbgs() << "newAlloc.getResult(): " << newAlloc.getResult() << "\n";
//     rewriter.replaceOp(allocOp, newAlloc->getResults());
//     llvm::dbgs() << "parent: " << *(newAlloc->getParentOp()) << "\n";
//     return success();
//   }
// };

namespace {
struct GPUDistributePass : public GPUDistributeBase<GPUDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp))
      return;

    auto workgroupSize = llvm::map_to_vector(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });

    // TODO: Thread through subgroup size everywhere.
    std::optional<llvm::APInt> maybeSubgroupSize =
        getEntryPoint(funcOp)->getSubgroupSize();
    // TODO: Don't hard code kCudaWarpSize here.
    int64_t subgroupSize =
        maybeSubgroupSize ? maybeSubgroupSize->getSExtValue() : kCudaWarpSize;

    IRRewriter rewriter(funcOp->getContext());
    rewriter.setInsertionPointToStart(&funcOp.front());
    DiagnosedSilenceableFailure result =
        mlir::transform::gpu::mapNestedForallToThreadsImpl(
            rewriter, std::nullopt, funcOp, workgroupSize, subgroupSize, false);
    if (!result.succeeded())
      return signalPassFailure();
    hoistAlloc(funcOp);
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUDistribute() {
  return std::make_unique<GPUDistributePass>();
}

} // namespace mlir::iree_compiler
