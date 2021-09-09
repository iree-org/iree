// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-tiled-op-interface"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.cpp.inc"

/// Converts an `OpFoldResult` to a `Value` by building a constant op if
/// if the `OpFoldResult` is an `IntegerAttr`.
static Value getValue(OpBuilder &builder, Location loc,
                      OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return builder.create<ConstantIndexOp>(loc,
                                           attr.cast<IntegerAttr>().getInt());
  }
  return valueOrAttr.get<Value>();
}

void registerTiledOpInterfaceExternalModels(DialectRegistry &registry) {
  LLVM_DEBUG({
    llvm::dbgs() << "Adding tiled op interface for tensor.insert_slice\n";
  });
  // TODO(ravishankarm): For now this is commented out since there are a lot of
  // upstream bugs exposed by this. Leaving the restructuring in place, but
  // avoiding the interface hook till those are addressed.
  //
  // registry.addOpInterface<tensor::InsertSliceOp,
  // InsertSliceTiledOpInterface>();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir
