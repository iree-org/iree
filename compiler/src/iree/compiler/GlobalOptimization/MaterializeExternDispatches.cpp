// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/Utils/CustomPatternApplicatorPassBase.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

/// Wrapper to build a hal.dispatch.extern op with the given arguments.
static Operation *
createDispatchExtern(PatternRewriter &rewriter, Operation *target,
                     ValueRange workload, TypeRange resultTypes,
                     ValueRange resultDims, ValueRange arguments,
                     ValueRange argumentDims, DenseI64ArrayAttr tiedOperands,
                     DictionaryAttr attrDict) {
  return rewriter.create<IREE::HAL::DispatchExternOp>(
      target->getLoc(), workload, resultTypes, resultDims, arguments,
      argumentDims, tiedOperands, attrDict);
}

/// Helper to emplace a block on the given hal.dispatch.extern op. This returns
/// the block arguments of the updated workgroup count region. Note that the
/// workgroup count region will not include the terminator and that is left up
/// to the user to properly populate.
static FailureOr<ValueRange>
emplaceExternWorkgroupCountRegion(PatternRewriter &rewriter, Operation *op) {
  auto externOp = dyn_cast<IREE::HAL::DispatchExternOp>(op);
  if (!externOp) {
    return failure();
  }

  Block *entryBlock = externOp.emplaceWorkgroupCountRegion(rewriter);

  /// Update the insertion point to the beginning of the block to enable
  /// contructing the workgroup count region.
  rewriter.setInsertionPointToStart(entryBlock);
  return ValueRange(entryBlock->getArguments());
}

static void
registerExternDispatchRewriteFunction(PDLPatternModule &pdlPatterns) {
  pdlPatterns.registerRewriteFunction("create_dispatch_extern",
                                      createDispatchExtern);
  pdlPatterns.registerRewriteFunction("emplace_extern_workgroup_count",
                                      emplaceExternWorkgroupCountRegion);
}

} // namespace

class MaterializeExternDispatchesPass
    : public iree_compiler::PatternApplicatorPassBase<
          MaterializeExternDispatchesPass,
          iree_compiler::GlobalOptimization::
              MaterializeExternDispatchesPassBase> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mlir::iree_compiler::IREE::HAL::HALDialect, arith::ArithDialect,
                pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }

  LogicalResult initializePatterns(MLIRContext *context,
                                   RewritePatternSet &tmpPatterns) {
    registerExternDispatchRewriteFunction(tmpPatterns.getPDLPatterns());
    for (auto fileName : this->pdlModuleFileNames) {
      if (failed(iree_compiler::detail::populatePDLModuleFromFileName(
              context, tmpPatterns, fileName))) {
        return failure();
      }
    }
    return success();
  }

  MaterializeExternDispatchesPass(ArrayRef<std::string> pdlModuleFileNames) {
    this->pdlModuleFileNames = pdlModuleFileNames;
  }
  MaterializeExternDispatchesPass(const MaterializeExternDispatchesPass &pass) =
      default;
};

std::unique_ptr<Pass> createMaterializeExternDispatchesPass(
    ArrayRef<std::string> pdlModuleFileNames) {
  return std::make_unique<MaterializeExternDispatchesPass>(pdlModuleFileNames);
}
} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
