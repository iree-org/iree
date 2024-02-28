// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

struct VectorTransforms {
  static const StringLiteral kVectorTransformMarker;
};
const StringLiteral VectorTransforms::kVectorTransformMarker =
    "__internal_vector_transform__";

StringRef getFusedMarker() { return "fused_numprocs_ge_numiters"; }

StringRef getWorkgroupKTiledMarker() { return "workgroup_k_tiled"; }

StringRef getWorkgroupL1TileMarker() { return "workgroup_l1_tile"; }

StringRef getWorkgroupMemoryMarker() { return "workgroup_memory"; }

StringRef getWorkgroupNumItemsGENumItersMarker() {
  return "workgroup_numprocs_ge_numiters";
}

StringRef getWorkgroupMemoryNumItemsGENumItersMarker() {
  return "workgroup_memory_numprocs_ge_numiters";
}

StringRef getCopyToWorkgroupMemoryMarker() {
  return "copy_to_workgroup_memory";
}

StringRef getTileReductionMarker() { return "tile_reduction"; }

StringRef getVectorizeMarker() { return "vectorize"; }

StringRef getDeleteMarker() { return "delete"; }

StringRef getMarkerOrNull(Operation *op) {
  StringAttr attr = op->getAttrOfType<StringAttr>(
      IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
  if (!attr)
    return "";
  return attr.getValue();
}

bool hasMarker(Operation *op, ArrayRef<StringRef> marker) {
  StringAttr attr = op->getAttrOfType<StringAttr>(
      IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
  return attr && (marker.empty() ||
                  llvm::any_of(marker, [&attr](StringRef markerValue) {
                    return attr.getValue() == markerValue;
                  }));
}

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker,
              StringAttr::get(op->getContext(), marker));
}

} // namespace mlir::iree_compiler
