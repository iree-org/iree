// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TILESIZESELECTIONPATTERN_H_
#define IREE_COMPILER_CODEGEN_COMMON_TILESIZESELECTIONPATTERN_H_

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

struct ScalableTileSize {
  int64_t size;
  bool scalable;
};

using ScalableTileSizes = SmallVector<ScalableTileSize>;

struct TileSizeConfig {
  ScalableTileSizes distributedTileSizes;
  ScalableTileSizes cacheTileSizes;
  ScalableTileSizes vectorTileSizes;

  TileSizeConfig(ScalableTileSizes _distributedTileSizes,
                 ScalableTileSizes _cacheTileSizes,
                 ScalableTileSizes _vectorTileSizes)
      : distributedTileSizes(_distributedTileSizes),
        cacheTileSizes(_cacheTileSizes),
        vectorTileSizes(_vectorTileSizes) {}
};

struct TileSizeAndPipelineConfig {
  TileSizeConfig rootConfig;
  IREE::Codegen::DispatchLoweringPassPipeline pipeline;
};

class TileSizeSelectionPattern {
 public:
  virtual ~TileSizeSelectionPattern() = default;

  virtual FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, Operation *op) const {
    assert(false && "need to implement");
    return failure();
  }
};

template <typename T>
class OpTileSizeSelectionPattern : public TileSizeSelectionPattern {
 public:
  virtual ~OpTileSizeSelectionPattern() = default;

  virtual FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, T op) const {
    assert(false && "need to implement");
  }

  FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, Operation *op) const final override {
    if (T typed_op = llvm::dyn_cast<T>(op)) {
      return matchAndConfig(funcOp, typed_op);
    }
    return failure();
  }
};

class ContractionOpTileSizeSelectionPattern : public TileSizeSelectionPattern {
 public:
  virtual ~ContractionOpTileSizeSelectionPattern() = default;

  virtual FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, linalg::ContractionOpInterface op) const {
    assert(false && "need to implement");
  }

  FailureOr<TileSizeAndPipelineConfig> matchAndConfig(
      FunctionOpInterface funcOp, Operation *op) const final override {
    auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
      return failure();
    }
    return matchAndConfig(funcOp, cast<linalg::ContractionOpInterface>(op));
  }

 protected:
  static bool isInnermostReduction(linalg::ContractionOpInterface op) {
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());

    SmallVector<unsigned> dims;
    linalgOp.getReductionDims(dims);
    // Only support exactly one reduction dim, and it is the innermost one.
    if (dims.size() != 1 || dims[0] != linalgOp.getNumLoops() - 1) {
      return false;
    }
    return true;
  }
};

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_CODEGEN_COMMON_TILESIZESELECTIONPATTERN_H_
