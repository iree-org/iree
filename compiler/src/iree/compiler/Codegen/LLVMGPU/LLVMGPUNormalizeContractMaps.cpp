// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

static void normalizeContractionMaps(RewriterBase &rewriter,
                                     vector::ContractionOp contractOp) {
  SmallVector<int64_t> perm = {1, 0};
  MLIRContext *context = rewriter.getContext();
  AffineExpr m, n, k;
  bindDims(context, m, n, k);

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList(MapList{{m, k}, {k, n}, {m, n}}, context);

  Location loc = contractOp.getLoc();
  auto transposeAcc =
      rewriter.create<vector::TransposeOp>(loc, contractOp.getAcc(), perm);
  auto newContractOp = rewriter.create<vector::ContractionOp>(
      loc, contractOp.getRhs(), contractOp.getLhs(), transposeAcc,
      rewriter.getAffineMapArrayAttr(indexingMaps),
      contractOp.getIteratorTypes());

  rewriter.replaceOpWithNewOp<vector::TransposeOp>(contractOp, newContractOp,
                                                   perm);
}

struct LLVMGPUNormalizeContractMapsPass
    : public LLVMGPUNormalizeContractMapsBase<
          LLVMGPUNormalizeContractMapsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    auto scheduleAttr =
        func->getAttrOfType<IREE::GPU::MMAScheduleAttr>(scheduleAttrName);
    if (!scheduleAttr) {
      DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
      scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
          configDict.get(scheduleAttrName));
    }
    if (!scheduleAttr) {
      func.emitError() << "missing mma_schedule\n";
      return signalPassFailure();
    }

    SmallVector<vector::ContractionOp> targetOps;
    func->walk([&](vector::ContractionOp contractOp) {
      VectorContractOpInfo opInfo(contractOp);
      if (opInfo.getOpKind() == VectorContractOpInfo::OpKind::KM_NK_MN &&
          dyn_cast<VectorType>(contractOp.getAccType())) {
        targetOps.push_back(contractOp);
      }
    });

    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    for (auto contractOp : llvm::make_early_inc_range(targetOps)) {
      rewriter.setInsertionPoint(contractOp);
      normalizeContractionMaps(rewriter, contractOp);
    }

    // We at least did one flip, and would require to flip the schedule to match
    // new contract op.
    if (!targetOps.empty()) {
      auto newScheduleAttr = IREE::GPU::MMAScheduleAttr::get(
          context, scheduleAttr.getIntrinsic(),
          scheduleAttr.getSubgroupNCount(), scheduleAttr.getSubgroupMCount(),
          scheduleAttr.getSubgroupNTileCount(),
          scheduleAttr.getSubgroupMTileCount(),
          scheduleAttr.getSubgroupKTileCount());
      func->setAttr(scheduleAttrName, newScheduleAttr);
    }
  }
};
} // namespace
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUNormalizeContractMapsPass() {
  return std::make_unique<LLVMGPUNormalizeContractMapsPass>();
}

} // namespace mlir::iree_compiler
