// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler {

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
  if (f) {
    filters.push_back(f);
  }
}

LogicalResult LinalgTransformationFilter::checkAndNotify(RewriterBase &rewriter,
                                                         Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); })) {
    return failure();
  }

  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty() || matchByDefault) {
      return success();
    }

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction) {
    if (attr.getValue() == filter) {
      return success();
    }
  }

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void LinalgTransformationFilter::replaceLinalgTransformationFilter(
    RewriterBase &rewriter, Operation *op) const {
  if (replacement.has_value()) {
    op->setAttr(LinalgTransforms::kLinalgTransformMarker, replacement.value());
  } else {
    op->removeAttr(
        rewriter.getStringAttr(LinalgTransforms::kLinalgTransformMarker));
  }
}

bool LinalgTransformationFilter::hasReplacementFilter(Operation *op) const {
  if (!replacement) {
    return false;
  }
  auto attr = op->getAttr(LinalgTransforms::kLinalgTransformMarker)
                  .dyn_cast<StringAttr>();
  return attr && attr == *replacement;
}

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
  StringAttr attr =
      op->getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker);
  if (!attr)
    return "";
  return attr.getValue();
}

bool hasMarker(Operation *op, ArrayRef<StringRef> marker) {
  StringAttr attr =
      op->getAttrOfType<StringAttr>(LinalgTransforms::kLinalgTransformMarker);
  return attr && (marker.empty() ||
                  llvm::any_of(marker, [&attr](StringRef markerValue) {
                    return attr.getValue() == markerValue;
                  }));
}

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(LinalgTransforms::kLinalgTransformMarker,
              StringAttr::get(op->getContext(), marker));
}

} // namespace mlir::iree_compiler
