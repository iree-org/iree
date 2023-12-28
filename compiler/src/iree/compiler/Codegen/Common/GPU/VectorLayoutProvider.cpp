// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/VectorLayoutProvider.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

/// The order in which the dimensions are considered after distribution.
const LayoutDimension simtLabels[] = {
    LayoutDimension::BATCHY, LayoutDimension::BATCHX, LayoutDimension::VECTORZ,
    LayoutDimension::VECTORY, LayoutDimension::VECTORX};

SmallVector<int64_t>
HigherDimLayoutProvider::getDistributedShape(TypedValue<VectorType> value) {
  auto layout = analysis.getLayout<LayoutAttr>(value);
  return layout.getSIMTVectorShape(simtLabels);
}

}; // namespace mlir::iree_compiler
