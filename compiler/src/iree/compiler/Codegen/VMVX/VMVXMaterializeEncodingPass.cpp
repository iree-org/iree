// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/EncodingInfo.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/VMVX/EncodingInfo.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

using namespace IREE::LinalgExt;
using IREE::HAL::ExecutableTargetAttr;

namespace {

static MatmulTileParams chooseMatmulTileParamsGeneric() { return {8, 4, 8}; }

static MatmulTileParams chooseMicrokernelMatmulTileParams() {
  return {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};
}

static MatmulTileParams chooseMatmulTileParams(MatmulType type,
                                               ExecutableTargetAttr target) {
  if (hasMicrokernels(target)) {
    return chooseMicrokernelMatmulTileParams();
  }
  return chooseMatmulTileParamsGeneric();
}

static MaterializeEncodingValueFn getMaterializeEncodingValueFn(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (hasMicrokernels(targetAttr)) {
    return chooseDynamicEncodingInfoVMVXMicrokernels;
  }
  return {};
}

struct VMVXMaterializeEncodingPass
    : public VMVXMaterializeEncodingBase<VMVXMaterializeEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, AffineDialect, tensor::TensorDialect,
                IREE::Flow::FlowDialect, IREE::LinalgExt::IREELinalgExtDialect,
                IREE::VMVX::VMVXDialect>();
  }
  void runOnOperation() override;
};

}  // namespace

void VMVXMaterializeEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();
  RewritePatternSet materializeEncodingPattern(context);
  auto targetAttr = ExecutableTargetAttr::lookup(operation);
  MaterializeEncodingTypeConverter typeConverter(
      [targetAttr](
          RankedTensorType tensorType) -> FailureOr<MaterializeEncodingInfo> {
        Optional<TensorEncoding> encoding = getEncoding(tensorType);
        if (!encoding) return failure();

        auto matmulType = getMatmulType(*encoding);
        auto matmulOperandRole = getMatmulOperandRole(*encoding);
        if (!matmulType || !matmulOperandRole) {
          return failure();
        }
        MatmulTileParams tileParams =
            chooseMatmulTileParams(*matmulType, targetAttr);
        auto encodingInfo = chooseEncodingInfoForMatmul(
            *matmulType, *matmulOperandRole, tileParams);
        adjustTileSizesToNarrowStaticShape(encodingInfo, tensorType.getShape());
        return encodingInfo;
      });
  MaterializeEncodingConversionTarget target(*context);
  auto materializeEncodingValueFn = getMaterializeEncodingValueFn(targetAttr);
  populateMaterializeEncodingIntoPackUnPackPatterns(materializeEncodingPattern,
                                                    target, typeConverter,
                                                    materializeEncodingValueFn);

  if (failed(applyPartialConversion(operation, target,
                                    std::move(materializeEncodingPattern)))) {
    operation.emitOpError("materialization failed");
    return signalPassFailure();
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and resolve
  // dims ops.
  {
    RewritePatternSet patterns(context);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
      operation.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createVMVXMaterializeEncodingPass() {
  return std::make_unique<VMVXMaterializeEncodingPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
