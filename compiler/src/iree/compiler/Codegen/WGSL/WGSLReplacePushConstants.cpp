// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace {

// These must match what the runtime uses.
#define IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX 3
#define IREE_HAL_WEBGPU_PARAMS_BINDING_INDEX 0

static Value convertOpTypeFromI32(IREE::HAL::InterfaceConstantLoadOp loadOp,
                                  vector::ExtractElementOp extractElementOp) {
  OpBuilder builder(loadOp);

  auto loc = loadOp.getLoc();
  auto opType = loadOp.getType();

  // Index
  if (opType.isIndex()) {
    return builder.create<arith::IndexCastOp>(loc, opType, extractElementOp);
  }

  unsigned sourceBitWidth = 32;
  unsigned destBitWidth = opType.getIntOrFloatBitWidth();

  // AnySignlessInteger
  if (opType.isa<IntegerType>()) {
    if (sourceBitWidth > destBitWidth) {
      return builder.create<arith::TruncIOp>(loc, opType, extractElementOp);
    } else if (sourceBitWidth < destBitWidth) {
      return builder.create<arith::ExtUIOp>(loc, opType, extractElementOp);
    } else {
      return extractElementOp.getResult();
    }
  }

  // AnyFloat
  Value resizedValue = extractElementOp.getResult();
  if (sourceBitWidth > destBitWidth) {
    return builder.create<arith::TruncFOp>(loc, opType, extractElementOp);
  } else if (sourceBitWidth < destBitWidth) {
    return builder.create<arith::ExtFOp>(loc, opType, extractElementOp);
  }
  return builder.create<arith::BitcastOp>(loc, opType, resizedValue);
}

static void replaceConstantLoadOp(IREE::Flow::DispatchTensorLoadOp loadOp,
                                  IREE::HAL::InterfaceConstantLoadOp op) {
  OpBuilder builder(op);

  // tensor.extract -> vector<4xi32>
  uint64_t vec4Index = op.getIndex().getZExtValue() / 4;
  auto tensorOffsetValue =
      builder.createOrFold<arith::ConstantIndexOp>(op.getLoc(), vec4Index);
  auto tensorExtractOp = builder.createOrFold<tensor::ExtractOp>(
      op.getLoc(), loadOp, tensorOffsetValue);

  // vector<4xi32> -> i32
  uint64_t elementIndex = op.getIndex().getZExtValue() % 4;
  auto vectorOffsetValue =
      builder.createOrFold<arith::ConstantIndexOp>(op.getLoc(), elementIndex);
  auto vectorExtractElementOp = builder.create<vector::ExtractElementOp>(
      op.getLoc(), tensorExtractOp, vectorOffsetValue);

  // i32 -> original type
  auto convertedTypeResult = convertOpTypeFromI32(op, vectorExtractElementOp);
  op.replaceAllUsesWith(convertedTypeResult);

  op.erase();
}

class WGSLReplacePushConstantsPass
    : public WGSLReplacePushConstantsBase<WGSLReplacePushConstantsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                    IREE::Flow::FlowDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto loc = funcOp.getLoc();
    auto constantLoadOps =
        llvm::to_vector<4>(funcOp.getOps<IREE::HAL::InterfaceConstantLoadOp>());
    if (constantLoadOps.empty()) return;

    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(&funcOp.getBlocks().front());

    // Group all push constants into a single `hal.interface.binding.subspan`
    // and load from it once using `flow.dispatch.tensor.load`, then extract
    // individual push constants with `tensor.extract`.

    // Find the range of push constant indices (0 to some maximum).
    uint64_t maxConstantIndex = 0;
    // Inspect the alignment values. These are just hints, so if all are equal
    // then use the value, otherwise drop the alignment hint.
    SmallVector<uint64_t> alignmentValues;
    bool missingAlignmentValue = false;
    for (auto constantLoadOp : constantLoadOps) {
      maxConstantIndex =
          std::max(constantLoadOp.getIndex().getZExtValue(), maxConstantIndex);

      auto alignmentAttr = constantLoadOp.getAlignmentAttr();
      if (alignmentAttr) {
        uint64_t alignmentValue = alignmentAttr.getValue().getZExtValue();
        alignmentValues.push_back(alignmentValue);
      } else {
        missingAlignmentValue = true;
      }
    }
    mlir::IntegerAttr alignmentAttr = nullptr;
    // TODO(scotttodd): try llvm::all_equal with attrs directly
    if (!missingAlignmentValue && llvm::all_equal(alignmentValues)) {
      alignmentAttr = constantLoadOps[0].getAlignmentAttr();
    }

    // We could store into a tensor<Nxi32>, but vec4s are better supported, so
    // we'll use tensor<Nxvector<4xi32>> instead.
    // Compute how many vec4s to use, i.e.
    //   max index 0 -> 1 vec4
    //   max index 3 -> 1 vec4
    //   max index 4 -> 2 vec4s
    uint64_t numberOfVec4s = maxConstantIndex / 4 + 1;

    // hal.interface.binding.subspan ->
    // !flow.dispatch.tensor<readonly:tensor<Nxvector<4xi32>>>
    //   * Group all push constants into a single tensor<Nxvector<4xi32>>
    //   * If individual data types differ, they'll be bitcast when extracted
    auto v4i32Type = VectorType::get({4}, builder.getI32Type());
    auto dispatchTensorType = IREE::Flow::DispatchTensorType::get(
        IREE::Flow::TensorAccess::ReadOnly,
        {static_cast<int64_t>(numberOfVec4s)}, v4i32Type);
    SmallVector<Value> dynamicDims;
    // Note: we're ignoring all potential 'values' hints (if provided) on ops -
    // InterfaceBindingSubspanOp has no matching concept and we assume that any
    // analysis using the hint should have been performed by earlier passes.
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto subspanOp = builder.create<IREE::HAL::InterfaceBindingSubspanOp>(
        loc, dispatchTensorType,
        /*set=*/APInt(64, IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX),
        /*binding=*/APInt(64, IREE_HAL_WEBGPU_PARAMS_BINDING_INDEX),
        IREE::HAL::DescriptorType::UniformBuffer,
        /*byte_offset=*/zero, dynamicDims, alignmentAttr, nullptr);

    // flow.dispatch.tensor.load -> tensor<Nxvector<4xi32>>
    auto tensorType =
        RankedTensorType::get({(int64_t)numberOfVec4s}, v4i32Type);
    auto loadOp = builder.create<IREE::Flow::DispatchTensorLoadOp>(
        loc, tensorType, subspanOp, dynamicDims);

    // The grouped subspan and load are complete - now extract each constant.
    for (auto constantLoadOp : constantLoadOps) {
      replaceConstantLoadOp(loadOp, constantLoadOp);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createWGSLReplacePushConstantsPass() {
  return std::make_unique<WGSLReplacePushConstantsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
