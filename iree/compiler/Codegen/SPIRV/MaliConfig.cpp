// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MaliConfig.h - Mali CodeGen Configurations -------------------------===//
//
// This file contains CodeGen configurations for Mali GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  // The number of threads per workgroup along each dimension.
  std::array<int64_t, 3> workgroupSize;
};

//===----------------------------------------------------------------------===//
// Convolution
//===----------------------------------------------------------------------===//

static LogicalResult setOpConfig(linalg::Conv2DNhwcHwcfOp op) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  ArrayRef<int64_t> inputShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);

  if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic)) {
    return success();
  }

  bool isInputTilable = inputShape[3] % 4 == 0 || inputShape[3] < 4;
  if (!isInputTilable) return success();

  // A list of preferred tile sizes and workgroup sizes.
  // TODO(antiagainst): This is for Valhall now; need to consider other Mali
  // architectures.
  static const TileWorkgroupSizePair tileWorkgroupSizePairs[] = {
      {{4, 4, 16}, {4, 4, 1}},
      {{2, 2, 64}, {16, 1, 1}},
      {{4, 8, 8}, {2, 4, 2}},
      {{2, 2, 32}, {8, 2, 1}},
      {{1, 1, 32}, {8, 1, 1}}};

  for (const auto &pair : tileWorkgroupSizePairs) {
    const std::array<int64_t, 3> &tileSize = pair.tileSize;
    const std::array<int64_t, 3> &workgroupSize = pair.workgroupSize;

    bool isOutputTilable = (outputShape[0] == 1) &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    TileSizesListType tileSizes;
    // Workgroup level.
    tileSizes.push_back({/*batch=*/0, /*output_height=*/tileSize[0],
                         /*output_width=*/tileSize[1],
                         /*output_channel=*/tileSize[2]});

    // No tiling at the subgroup level given that we don't use subgroup
    // level syncrhonization or shared memory.
    tileSizes.emplace_back();
    // Invocation level.
    tileSizes.push_back({/*batch=*/0,
                         /*output_height=*/tileSize[0] / workgroupSize[2],
                         /*output_width=*/tileSize[1] / workgroupSize[1],
                         /*output_channel=*/tileSize[2] / workgroupSize[0]});

    auto funcOp = op->getParentOfType<FuncOp>();
    if (failed(setOpConfigAndEntryPointFnTranslation(
            funcOp, op, tileSizes, {}, pipeline, workgroupSize))) {
      return failure();
    }

    return defineConvWorkgroupCountRegion(
        op, llvm::makeArrayRef(outputShape).drop_front(), tileSize);
  }
  return success();
}

static LogicalResult setOpConfig(linalg::DepthwiseConv2DNhwOp op) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  ArrayRef<int64_t> inputShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);

  if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic)) {
    return success();
  }

  // A list of preferred tile sizes and workgroup sizes.
  // TODO(antiagainst): This is for Valhall now; need to consider other Mali
  // architectures.
  static const TileWorkgroupSizePair tileWorkgroupSizePairs[] = {
      {{2, 2, 32}, {8, 2, 2}},
      {{1, 4, 16}, {4, 4, 1}},
      {{1, 1, 64}, {16, 1, 1}},
      {{4, 4, 8}, {2, 4, 2}},
  };

  for (const auto &pair : tileWorkgroupSizePairs) {
    const std::array<int64_t, 3> &tileSize = pair.tileSize;
    const std::array<int64_t, 3> &workgroupSize = pair.workgroupSize;

    bool isOutputTilable = outputShape[0] == 1 &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    TileSizesListType tileSizes;
    // Workgroup level.
    tileSizes.push_back({/*batch=*/0,
                         /*output_height=*/tileSize[0],
                         /*output_width=*/tileSize[1],
                         /*output_channel=*/tileSize[2]});

    // No tiling at the subgroup level given that we don't use subgroup
    // level syncrhonization  or shared memory.
    tileSizes.emplace_back();
    tileSizes.push_back({/*batch=*/0,
                         /*output_height=*/tileSize[0] / workgroupSize[2],
                         /*output_width=*/tileSize[1] / workgroupSize[1],
                         /*output_channel=*/tileSize[2] / workgroupSize[0]});

    auto funcOp = op->getParentOfType<FuncOp>();
    if (failed(setOpConfigAndEntryPointFnTranslation(
            funcOp, op, tileSizes, {}, pipeline, workgroupSize))) {
      return failure();
    }

    return defineConvWorkgroupCountRegion(
        op, llvm::makeArrayRef(outputShape).drop_front(), tileSize);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setMaliCodeGenConfig(const spirv::TargetEnv &,
                                   Operation *rootOp) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([](auto op) {
        std::array<int64_t, 2> workgroupXY = {8, 2};
        std::array<int64_t, 3> threadMNK;
        auto inputType = op.inputs()[0].getType().template cast<ShapedType>();
        if (inputType.getElementType().isF16()) {
          threadMNK = {8, 8, 4};
        } else {
          threadMNK = {6, 4, 4};
        }
        return setMatmulOpConfig(op, workgroupXY, threadMNK);
      })
      .Case<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwOp>(
          [](auto op) { return setOpConfig(op); })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
