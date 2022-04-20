// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/TrackingRewriteDriver.h"

#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"

using namespace mlir;

LogicalResult mlir::applyPatternsTrackAndFoldGreedily(
    Operation *root, RewriteListener &listener,
    const FrozenRewritePatternSet &patterns, GreedyRewriteConfig config) {
  return applyPatternsAndFoldGreedily(root, patterns, config, &listener);
}
