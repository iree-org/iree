// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

static const unsigned kNumMaxParallelDims = 3;

namespace mlir {
namespace iree_compiler {

namespace {
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

void SetNumWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp module = variantOp.getInnerModule();

  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPoints =
      getAllEntryPoints(module);
  for (auto funcOp : module.getOps<FuncOp>()) {
    auto entryPointOp = entryPoints.lookup(funcOp.getName());
    if (!entryPointOp) continue;
    SmallVector<int64_t, 4> currWorkloadPerWorkgroup;

    // First check if there is a workload provided.
    if (!workloadPerWorkgroup.empty()) {
      currWorkloadPerWorkgroup.assign(workloadPerWorkgroup.begin(),
                                      workloadPerWorkgroup.end());
    } else if (IREE::HAL::TranslationInfo translationInfo =
                   getTranslationInfo(entryPointOp)) {
      if (ArrayAttr workloadPerWorkgroupAttr =
              translationInfo.workloadPerWorkgroup()) {
        currWorkloadPerWorkgroup = llvm::to_vector<4>(llvm::map_range(
            workloadPerWorkgroupAttr,
            [](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
      }
    }

    if (currWorkloadPerWorkgroup.empty()) {
      // If no workgroup size is specified, leave the workgroup size as is, just
      // set the number of workgroups to be 1, 1, 1 to have a single invocation.
      WorkgroupCountRegionBuilder regionBuilder =
          [](OpBuilder &b, Location loc,
             std::array<Value, 3> workload) -> std::array<Value, 3> {
        Value one = b.create<ConstantIndexOp>(loc, 1);
        return {one, one, one};
      };
      OpBuilder builder(context);
      for (auto funcOp : module.getOps<FuncOp>()) {
        if (failed(
                defineWorkgroupCountRegion(builder, funcOp, regionBuilder))) {
          return signalPassFailure();
        }
      }
    } else {
      if (failed(materializeStaticLaunchInformation(
              funcOp, currWorkloadPerWorkgroup))) {
        funcOp.emitError("failed to materialize constant workgroup size");
        return signalPassFailure();
      }
    }
  }

  // Apply post distribution canonicalization passes.
  OwningRewritePatternList canonicalization(context);
  AffineMinOp::getCanonicalizationPatterns(canonicalization, context);
  populateAffineMinSCFCanonicalizationPattern(canonicalization);
  IREE::Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                           context);
  (void)applyPatternsAndFoldGreedily(module, std::move(canonicalization));
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSetNumWorkgroupsPass(ArrayRef<int64_t> workgroupSize) {
  return std::make_unique<SetNumWorkgroupsPass>(workgroupSize);
}

}  // namespace iree_compiler
}  // namespace mlir
