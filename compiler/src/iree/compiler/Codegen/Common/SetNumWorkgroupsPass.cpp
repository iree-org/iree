// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-set-num-workgroups"

static const unsigned kNumMaxParallelDims = 3;

namespace mlir {
namespace iree_compiler {

namespace {
/// Sets the hal.interace.workgroup.size operation to the constant value passed
/// in as `workloadPerWorkgroup`. The number of entries in
/// `workloadPerWorkgroup` is at least as much as the dimensionality of the
/// workgroup. It is assumed that the inner-most loop is mapped to the fastest
/// varying dimension in flow.dispatch.workgroup_size.
class SetWorkgroupSizePattern
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  SetWorkgroupSizePattern(MLIRContext *context,
                          ArrayRef<int64_t> workloadPerWorkgroupRef,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        workloadPerWorkgroup(llvm::to_vector<4>(
            workloadPerWorkgroupRef.size() > kNumMaxParallelDims
                ? workloadPerWorkgroupRef.take_front(kNumMaxParallelDims)
                : workloadPerWorkgroupRef)) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceWorkgroupSizeOp workgroupSizeOp,
      PatternRewriter &rewriter) const override {
    int64_t dim = workgroupSizeOp.dimension().getSExtValue();
    if (dim >= workloadPerWorkgroup.size()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
        workgroupSizeOp, workloadPerWorkgroup[dim]);
    return success();
  }

 private:
  SmallVector<int64_t, 4> workloadPerWorkgroup;
};

class SetNumWorkgroupsPass : public SetNumWorkgroupsBase<SetNumWorkgroupsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::HAL::HALDialect, linalg::LinalgDialect>();
  }

  SetNumWorkgroupsPass(ArrayRef<int64_t> ws = {})
      : workloadPerWorkgroup(ws.begin(), ws.end()) {}
  SetNumWorkgroupsPass(const SetNumWorkgroupsPass &pass)
      : workloadPerWorkgroup(pass.workloadPerWorkgroup) {}

  void runOnOperation() override;

 private:
  SmallVector<int64_t> workloadPerWorkgroup;
};
}  // namespace

LogicalResult setNumWorkgroupsImpl(IREE::HAL::ExecutableVariantOp variantOp,
                                   ArrayRef<int64_t> workloadPerWorkgroup) {
  MLIRContext *context = variantOp.getContext();
  ModuleOp module = variantOp.getInnerModule();

  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(module);
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;

    SmallVector<int64_t, 4> currWorkloadPerWorkgroup;

    // First check if there is a per-workgroup workload provided.
    if (!workloadPerWorkgroup.empty()) {
      currWorkloadPerWorkgroup.assign(workloadPerWorkgroup.begin(),
                                      workloadPerWorkgroup.end());
    } else if (IREE::Codegen::TranslationInfoAttr translationInfo =
                   getTranslationInfo(exportOp)) {
      currWorkloadPerWorkgroup = translationInfo.getWorkloadPerWorkgroupVals();
    }

    if (!currWorkloadPerWorkgroup.empty()) {
      // Fold hal.workgroup.size ops.
      RewritePatternSet patterns(funcOp.getContext());
      patterns.insert<SetWorkgroupSizePattern>(funcOp.getContext(),
                                               currWorkloadPerWorkgroup);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
        return failure();
    }

    // The workgroup count region might already be set by op-specific
    // configuration logic. If so, just return to avoid overwriting that.
    if (!exportOp.workgroup_count().empty()) continue;

    WorkgroupCountRegionBuilder regionBuilder;
    if (currWorkloadPerWorkgroup.empty()) {
      // If no workgroup size is specified, leave the workgroup size as is, just
      // set the number of workgroups to be 1, 1, 1 to have a single invocation.
      regionBuilder = [](OpBuilder &b, Location loc, Value device,
                         std::array<Value, 3> workload) {
        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        return std::array<Value, 3>{one, one, one};
      };
    } else {
      assert(currWorkloadPerWorkgroup.size() <= kNumMaxParallelDims &&
             "workloadPerWorkgroup size greater than max num parallel dims");
      regionBuilder = [&currWorkloadPerWorkgroup](
                          OpBuilder &b, Location loc, Value device,
                          std::array<Value, 3> workload) {
        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        std::array<Value, 3> returnValues = {one, one, one};
        for (auto ts : llvm::enumerate(currWorkloadPerWorkgroup)) {
          returnValues[ts.index()] = applyMapToValues(
              b, loc,
              AffineMap::get(0, 1,
                             b.getAffineSymbolExpr(0).ceilDiv(ts.value())),
              workload[ts.index()])[0];
        }
        return returnValues;
      };
    }

    OpBuilder builder(context);
    if (failed(defineWorkgroupCountRegion(builder, exportOp, regionBuilder))) {
      return failure();
    }
  }

  // Apply post distribution canonicalization passes.
  RewritePatternSet canonicalization(context);
  AffineMinOp::getCanonicalizationPatterns(canonicalization, context);
  populateAffineMinSCFCanonicalizationPattern(canonicalization);
  IREE::Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                           context);
  return applyPatternsAndFoldGreedily(module, std::move(canonicalization));
}

void SetNumWorkgroupsPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  if (failed(setNumWorkgroupsImpl(variantOp, workloadPerWorkgroup)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSetNumWorkgroupsPass(ArrayRef<int64_t> workgroupSize) {
  return std::make_unique<SetNumWorkgroupsPass>(workgroupSize);
}

}  // namespace iree_compiler
}  // namespace mlir
