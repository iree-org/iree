// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUSELECTLOWERINGSTRATEGYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
/// Selects the lowering strategy for a hal.executable.variant operation.
class LLVMCPUSelectLoweringStrategyPass
    : public impl::LLVMCPUSelectLoweringStrategyPassBase<
          LLVMCPUSelectLoweringStrategyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::CPU::IREECPUDialect, IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
};
} // namespace

static bool isValidInterchange(ArrayRef<int64_t> interchange, int numLoops) {
  if (interchange.empty()) {
    return true;
  }
  return isPermutationVector(interchange) && interchange.size() == numLoops;
}

/// Verifies if the tile sizes from `loweringConfig` are valid for each level.
static LogicalResult verifyMultiTilingExpertPassPipelineConfig(
    Operation *op, IREE::CPU::LoweringConfigAttr loweringConfig) {

  auto interfaceOp = dyn_cast_or_null<TilingInterface>(op);
  if (!interfaceOp) {
    return success();
  }

  // Collects parallel loops.
  llvm::SmallDenseSet<unsigned> pLoopsSet;
  for (auto [index, iteratorType] :
       llvm::enumerate(interfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::parallel) {
      pLoopsSet.insert(index);
    }
  }

  for (unsigned i = 0, e = IREE::CPU::TilingLevel::MaxNumTileLevels; i < e;
       ++i) {
    if (!loweringConfig.hasTilingLevel(i)) {
      continue;
    }

    auto level = static_cast<IREE::CPU::TilingLevel>(i);
    auto tilingLevelAttr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(level));
    switch (level) {
    case IREE::CPU::TilingLevel::DistributionTiles:
    case IREE::CPU::TilingLevel::CacheParallelTiles:
    case IREE::CPU::TilingLevel::VectorCommonParallelTiles:
    case IREE::CPU::TilingLevel::VectorInnerParallelTiles: {
      for (auto [index, tileSize] :
           llvm::enumerate(tilingLevelAttr.getSizes())) {
        if (tileSize != 0 && !pLoopsSet.contains(index)) {
          return op->emitOpError(
                     "expected only parallel dims to be set in the ")
                 << IREE::CPU::getTilingLevelName(level)
                 << " tiling level, but tile size at index (" << index
                 << ") was also set";
        }
      }
      break;
    }
    case IREE::CPU::TilingLevel::CacheReductionTiles:
    case IREE::CPU::TilingLevel::VectorReductionTiles: {
      for (auto [index, tileSize] :
           llvm::enumerate(tilingLevelAttr.getSizes())) {
        if (tileSize != 0 && pLoopsSet.contains(index)) {
          return op->emitOpError(
                     "expected only reduction dims to be set in the ")
                 << IREE::CPU::getTilingLevelName(level)
                 << " tiling level, but tile size at index (" << index
                 << ") was also set";
        }
      }
      break;
    }
    case IREE::CPU::TilingLevel::MaxNumTileLevels:
    case IREE::CPU::TilingLevel::InvalidLevel:
      break;
    };

    ArrayRef<int64_t> interchange = tilingLevelAttr.getInterchange();
    size_t expectedSize = tilingLevelAttr.getSizes().size();
    if (!isValidInterchange(interchange, expectedSize)) {
      return op->emitOpError("expected [0, ")
             << expectedSize << ") to be set exactly once in interchange for "
             << IREE::CPU::getTilingLevelName(level) << " tiling level";
    }
  }

  return success();
}

/// Verifies that the given `loweringConfig` can decompose convolution ops to
/// lower dim ops. It requires {Distribution, VectorCommonParallel,
/// VectorReduction} tiling levels.
static LogicalResult verifyConvTileAndDecomposeExpertConfig(
    Operation *op, IREE::CPU::LoweringConfigAttr loweringConfig) {
  if (!isa<linalg::ConvolutionOpInterface>(op)) {
    return success();
  }

  auto getTileSizeAtIndex = [](ArrayRef<int64_t> sizes,
                               ArrayRef<bool> scalableFlags,
                               unsigned index) -> std::pair<int64_t, bool> {
    return std::make_pair(sizes[index],
                          index < scalableFlags.size() && scalableFlags[index]);
  };

  SmallVector<IREE::CPU::TilingLevel> requiredLevels = {
      IREE::CPU::DistributionTiles, IREE::CPU::VectorCommonParallelTiles,
      IREE::CPU::VectorReductionTiles};
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  SmallVector<int64_t> shapeAfterTiling = linalgOp.getStaticLoopRanges();
  for (auto level : requiredLevels) {
    if (!loweringConfig.hasTilingLevel(level)) {
      return op->emitOpError("expected ")
             << IREE::CPU::getTilingLevelName(level) << " is set";
    }
    auto tilingLevelAttr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(level));
    for (size_t i = 0, e = tilingLevelAttr.getSizes().size(); i < e; ++i) {
      auto [size, scalableFlag] = getTileSizeAtIndex(
          tilingLevelAttr.getSizes(), tilingLevelAttr.getScalableFlags(), i);
      if (scalableFlag) {
        shapeAfterTiling[i] = ShapedType::kDynamic;
        continue;
      }
      if (size == 1) {
        shapeAfterTiling[i] = 1;
        continue;
      }
      if (ShapedType::isDynamicShape(shapeAfterTiling[i]) ||
          ShapedType::isDynamic(size) || size == 0) {
        continue;
      }
      if (shapeAfterTiling[i] % size != 0) {
        shapeAfterTiling[i] = ShapedType::kDynamic;
      } else {
        shapeAfterTiling[i] = size;
      }
    }
  }

  int64_t khSize, kwSize, ohSize, owSize;
  auto isSizeExtracted =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp,
                linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
                linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
                linalg::PoolingNhwcMinUnsignedOp>([&](auto) {
            // shape: N, OH, OW, OC, KH, KW, (IC)
            khSize = shapeAfterTiling[4];
            kwSize = shapeAfterTiling[5];
            ohSize = shapeAfterTiling[1];
            owSize = shapeAfterTiling[2];
            return success();
          })
          .Case<linalg::Conv2DNchwFchwOp>([&](auto) {
            // shape: N, OC, OH, OW, (IC), KH, KW
            khSize = shapeAfterTiling[5];
            kwSize = shapeAfterTiling[6];
            ohSize = shapeAfterTiling[2];
            owSize = shapeAfterTiling[3];
            return success();
          })
          .Case<linalg::PoolingNchwSumOp, linalg::PoolingNchwMaxOp>([&](auto) {
            // shape: N, OC, OH, OW, KH, KW
            khSize = shapeAfterTiling[4];
            kwSize = shapeAfterTiling[5];
            ohSize = shapeAfterTiling[2];
            owSize = shapeAfterTiling[3];
            return success();
          })
          .Default([&](auto) { return failure(); });
  if (failed(isSizeExtracted)) {
    return op->emitOpError("unsupported conv types");
  }

  bool removeH = (khSize == 1 && ohSize == 1);
  bool removeW = (kwSize == 1 && owSize == 1);
  if (!removeH && !removeW) {
    return op->emitOpError("can't decompose the conv op");
  }

  return success();
}

/// Verify that valid configuration is set for all ops within the funcOp.
template <typename F>
static LogicalResult verifyLoweringConfiguration(FunctionOpInterface funcOp,
                                                 F verificationFn) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (isa<IREE::LinalgExt::CustomOp>(op)) {
      return WalkResult::advance();
    }
    auto loweringConfig = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
    if (!loweringConfig)
      return WalkResult::advance();
    return verificationFn(op, loweringConfig);
  });
  return failure(walkResult.wasInterrupted());
}

void LLVMCPUSelectLoweringStrategyPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    // Set the strategy with default heuristics.
    if (failed(initCPULaunchConfig(funcOp))) {
      funcOp.emitOpError("failed to set lowering configuration");
      return signalPassFailure();
    }

    auto translationInfo = getTranslationInfo(funcOp);
    if (!translationInfo) {
      continue;
    }

    // Verify the configuration.
    LogicalResult verificationStatus = success();
    switch (translationInfo.getDispatchLoweringPassPipeline()) {
    case IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert:
      verificationStatus = verifyLoweringConfiguration(
          funcOp, verifyMultiTilingExpertPassPipelineConfig);
      break;
    case IREE::Codegen::DispatchLoweringPassPipeline::
        CPUConvTileAndDecomposeExpert:
      verificationStatus = verifyLoweringConfiguration(
          funcOp, verifyConvTileAndDecomposeExpertConfig);
      break;
    default:
      break;
    }
    if (failed(verificationStatus)) {
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler
