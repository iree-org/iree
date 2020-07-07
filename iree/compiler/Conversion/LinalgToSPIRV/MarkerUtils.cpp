// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

struct VectorTransforms {
  static const StringLiteral kVectorTransformMarker;
};
const StringLiteral VectorTransforms::kVectorTransformMarker =
    "__internal_vector_transform__";
/// Checks if the operation has the `marker` If `marker` is null string, checks
/// if any marker is set.
static bool checkMarkerValue(Operation *op, StringRef marker = "") {
  StringAttr attr = op->getAttrOfType<StringAttr>(
      linalg::LinalgTransforms::kLinalgTransformMarker);
  return attr && (marker.empty() || attr.getValue() == marker);
}

StringRef getWorkGroupMarker() { return "workgroup"; }

StringRef getWorkItemMarker() { return "workitem"; }

bool hasMarker(Operation *op, StringRef marker) {
  return checkMarkerValue(op, marker);
}

bool hasWorkGroupMarker(Operation *op) {
  return checkMarkerValue(op, getWorkGroupMarker());
}

bool hasWorkItemMarker(Operation *op) {
  return checkMarkerValue(op, getWorkItemMarker());
}

void setMarker(Operation *op, StringRef marker) {
  op->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker,
              StringAttr::get(marker, op->getContext()));
}

void setWorkGroupMarker(Operation *op) { setMarker(op, getWorkGroupMarker()); }

void setWorkItemMarker(Operation *op) { setMarker(op, getWorkItemMarker()); }
}  // namespace iree_compiler
}  // namespace mlir
