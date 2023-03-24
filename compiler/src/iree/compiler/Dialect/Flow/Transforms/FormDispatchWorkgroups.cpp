// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-form-dispatch-workgroups"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Dispatch workgroups formation
//===----------------------------------------------------------------------===//

/// Traverses DispatchRegionOp in the function and rewrites them as
/// DispatchWorkgroupsOp
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>>
createDispatchWorkgroups(mlir::TensorDimTrackingRewriter &rewriter,
                         FunctionOpInterface funcOp,
                         DominanceInfo const &dominanceInfo,
                         bool generateWorkloadRegion) {
  SmallVector<Flow::DispatchRegionOp> regionOps;
  funcOp.walk([&](Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  SmallVector<Flow::DispatchWorkgroupsOp> result;
  for (auto regionOp : regionOps) {
    auto maybeWorkgroupOp =
        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) return failure();
    result.push_back(*maybeWorkgroupOp);
  }
  return result;
}

/// Return `true` if the given op is contained in DispatchWorkgroupsOp or in a
/// DispatchRegionOp.
static bool isInDispatchRegion(Operation *op) {
  return op->getParentOfType<Flow::DispatchWorkgroupsOp>() ||
         op->getParentOfType<Flow::DispatchRegionOp>();
}

/// Wrap a single op in a DispatchWorkgroupsOp. When generateWorkloadRegion is
/// true, `workload_count` region is generated for dispatch.region
static FailureOr<Flow::DispatchWorkgroupsOp> wrapInWorkgroupsOp(
    mlir::TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Compute workload.
  std::optional<Flow::WorkloadBuilder> workloadBuilder = std::nullopt;
  if (generateWorkloadRegion) {
    auto maybeBuilder =
        iree_compiler::IREE::Flow::getWorkloadBuilder(rewriter, op);
    if (failed(maybeBuilder)) return failure();
    workloadBuilder = *maybeBuilder;
  }

  // Simplify tensor::DimOps.
  SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
  if (failed(iree_compiler::IREE::Flow::simplifyDimOps(
          rewriter, rewriter.getTensorDimOps())))
    return failure();

  // Wrap operation.
  auto regionOp = Flow::wrapOpInDispatchRegion(rewriter, op, workloadBuilder);
  if (failed(regionOp)) return failure();
  if (failed(cloneProducersToRegion(rewriter, *regionOp))) return failure();
  auto workgroupsOp = Flow::rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
      *regionOp, rewriter);
  if (failed(workgroupsOp)) return failure();
  return *workgroupsOp;
}

/// Wrap all given ops in a DispatchWorkgroupsOp.
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> wrapInWorkgroupsOp(
    mlir::TensorDimTrackingRewriter &rewriter, SmallVector<Operation *> rootOps,
    bool generateWorkloadRegion) {
  SmallVector<Flow::DispatchWorkgroupsOp> result;
  for (Operation *rootOp : rootOps) {
    auto workgroupsOp =
        wrapInWorkgroupsOp(rewriter, rootOp, generateWorkloadRegion);
    if (failed(workgroupsOp)) return failure();
    result.push_back(*workgroupsOp);
  }
  return result;
}

/// Wrap all ops of the given types that are direct children of the given op
/// in DispatchWorkgroupsOps.
template <typename... OpTys>
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> wrapInWorkgroupsOp(
    mlir::TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Find ops of type OpTys.
  SmallVector<Operation *> rootOps;
  for (Region &r : op->getRegions())
    for (Block &b : r.getBlocks())
      for (Operation &op : b)
        if (isa<OpTys...>(&op)) rootOps.push_back(&op);

  // Wrap ops in DispatchWorkgroupsOps.
  return wrapInWorkgroupsOp(rewriter, rootOps, generateWorkloadRegion);
}

/// Rewrite top-level InsertSliceOps to FlowUpdateOps or wrap them in a
/// dispatch region.
LogicalResult convertInsertSliceOps(
    mlir::TensorDimTrackingRewriter &rewriter, mlir::FunctionOpInterface funcOp,
    SmallVector<Flow::DispatchWorkgroupsOp> &workgroupsOps,
    bool generateWorkloadRegion) {
  // Find eligible InsertSliceOps.
  SmallVector<tensor::InsertSliceOp> insertSliceOps;
  funcOp.walk([&](tensor::InsertSliceOp op) {
    if (!isInDispatchRegion(op)) insertSliceOps.push_back(op);
  });

  // Rewrite InsertSliceOps to FlowUpdateOps.
  SmallVector<Operation *> remainingInsertSliceOps;
  for (tensor::InsertSliceOp insertSliceOp : insertSliceOps) {
    if (failed(convertInsertSliceOpToFlowUpdateOp(rewriter, insertSliceOp))) {
      remainingInsertSliceOps.push_back(insertSliceOp);
    }
  }

  // Create a DispatchWorkgroupsOp for every remaining InsertSliceOp.
  FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingInsertSliceOps,
                         generateWorkloadRegion);
  if (failed(newWorkgroupsOps)) return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return success();
}

/// Rewrite top-level ExtractSliceOps to FlowSliceOps or wrap them in a
/// dispatch region.
LogicalResult convertExtractSliceOps(
    mlir::TensorDimTrackingRewriter &rewriter, mlir::FunctionOpInterface funcOp,
    SmallVector<Flow::DispatchWorkgroupsOp> &workgroupsOps,
    bool generateWorkloadRegion) {
  // Find eligible ExtractSliceOps.
  SmallVector<tensor::ExtractSliceOp> extractSliceOps;
  funcOp.walk([&](tensor::ExtractSliceOp op) {
    if (!isInDispatchRegion(op)) extractSliceOps.push_back(op);
  });

  // Rewrite ExtractSliceOps to FlowSliceOps.
  SmallVector<Operation *> remainingExtractSliceOps;
  for (tensor::ExtractSliceOp extractSliceOp : extractSliceOps) {
    if (failed(convertExtractSliceOpToFlowSliceOp(rewriter, extractSliceOp))) {
      remainingExtractSliceOps.push_back(extractSliceOp);
    }
  }

  // Create a DispatchWorkgroupsOp for every remaining ExtractSliceOp.
  FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingExtractSliceOps,
                         generateWorkloadRegion);
  if (failed(newWorkgroupsOps)) return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return success();
}

namespace {
/// Pass declaration.
struct FormDispatchWorkgroupsPass
    : public FormDispatchWorkgroupsBase<FormDispatchWorkgroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }
  FormDispatchWorkgroupsPass(bool generateWorkloadRegion) {
    this->generateWorkloadRegion = generateWorkloadRegion;
  }
  FormDispatchWorkgroupsPass(const FormDispatchWorkgroupsPass &pass)
      : FormDispatchWorkgroupsPass(pass.generateWorkloadRegion) {}
  void runOnOperation() override;
};
}  // namespace

void FormDispatchWorkgroupsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  DominanceInfo const &dominanceInfo = getAnalysis<DominanceInfo>();
  mlir::TensorDimTrackingRewriter rewriter(funcOp);

  // Step 1: Create a DispatchWorkgroupsOp for every DispatchRegionOp.
  auto maybeWorkgroupsOps = createDispatchWorkgroups(
      rewriter, funcOp, dominanceInfo, generateWorkloadRegion);
  if (failed(maybeWorkgroupsOps)) return signalPassFailure();
  SmallVector<Flow::DispatchWorkgroupsOp> workgroupsOps = *maybeWorkgroupsOps;

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After forming of dispatch workgroups ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Step 2: Rewrite InsertSliceOps to FlowUpdateOps.
  if (failed(convertInsertSliceOps(rewriter, funcOp, workgroupsOps,
                                   generateWorkloadRegion))) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.insert_slice`");
    return signalPassFailure();
  }

  // Step 3: Rewrite ExtractSliceOps to FlowUpdateOps.
  if (failed(convertExtractSliceOps(rewriter, funcOp, workgroupsOps,
                                    generateWorkloadRegion))) {
    funcOp->emitOpError(
        "failed to create dispatch region for `tensor.extract_slice`");
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After other conversions ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // A few extra canonicalizations/lowerings.
  {
    RewritePatternSet convertToFlowPatterns(context);
    Flow::populateTensorToFlowConversionPatterns(context,
                                                 convertToFlowPatterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(
        convertToFlowPatterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(
        convertToFlowPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(convertToFlowPatterns)))) {
      funcOp->emitOpError("failed conversion to flow.tensor ops");
      return signalPassFailure();
    }

    // Finally fold `tensor.insert_slice/extract_slice` operations with
    // `flow.dispatch.tensor.load/store`.
    RewritePatternSet foldExtractInsertSliceOps(context);
    Flow::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
        foldExtractInsertSliceOps, context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(foldExtractInsertSliceOps)))) {
      funcOp->emitOpError(
          "failed to insert/extract_slice with "
          "flow.dispatch.tensor.load/store");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormDispatchWorkgroupsPass(bool generateWorkloadRegion) {
  return std::make_unique<FormDispatchWorkgroupsPass>(generateWorkloadRegion);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
