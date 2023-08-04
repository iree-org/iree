// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_MANAGEDELEMENTS_UTILS_H_
#define IREE_COMPILER_UTILS_MANAGEDELEMENTS_UTILS_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

namespace mlir::iree_compiler {

// It is common to need to transform the type/contents of an ElementsAttr.
// However each concrete type handles this differently, and there are policy
// ramifications to how to store them. This class encapsulates the switchiness
// involved and attempts to expose an interface that can at least be made
// efficient.
class ManagedElementsMapper {
public:
  ManagedElementsMapper(IREE::Util::ManagedElementsAttrInterface managed)
      : managed(managed), elements(managed.getElements()) {}

  // Maps IntegerType and FloatType element types to a new attribute with
  // altered element type and values.
  // If it is structurally impossible to do the mapping, asserts in debug
  // builds and returns nullptr in release builds.
  // For floating point types, the APInt is bitcasted from the raw FP data.
  // This is not a particularly efficient way to be transforming but is the
  // most generic.
  TypedAttr
  mapViaBitcastAPInt(Type newElementType,
                     std::function<void(APInt &value)> elementCallback);

  ShapedType getShapedType() {
    return llvm::cast<ShapedType>(elements.getType());
  }

private:
  IREE::Util::ManagedElementsAttrInterface managed;
  ElementsAttr elements;
};

TypedAttr ManagedElementsMapper::mapViaBitcastAPInt(
    Type newElementType, std::function<void(APInt &value)> elementCallback) {
  auto inType = getShapedType();
  auto newType = inType.cloneWith(inType.getShape(), newElementType);
  if (auto intAttr = llvm::dyn_cast<DenseIntElementsAttr>(elements)) {
    // DenseIntElementsAttr.
    SmallVector<APInt> newValues;
    if (intAttr.isSplat()) {
      newValues.push_back(intAttr.getValues<APInt>()[0]);
      elementCallback(newValues.back());
    } else {
      newValues.reserve(elements.getNumElements());
      for (auto value : elements.getValues<APInt>()) {
        elementCallback(value);
        newValues.push_back(value);
      }
    }
    return DenseElementsAttr::get(newType, newValues);
  } else if (auto floatAttr = llvm::dyn_cast<DenseFPElementsAttr>(elements)) {
    // TODO.
    // // DenseFPElementsAttr.
    // SmallVector<APInt> newValues;
    // if (intAttr.isSplat()) {
    //   newValues.push_back(intAttr.getValues<APInt>()[0]);
    //   elementCallback(newValues.back());
    // } else {
    //   newValues.reserve(elements.getNumElements());
    //   for (auto value : elements.getValues<APInt>()) {
    //     elementCallback(value);
    //     newValues.push_back(value);
    //   }
    // }
    // return DenseElementsAttr::get(newType, newValues);
  }

  assert(false && "unhandled ManagedElementMapper type");
  return {};
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_MANAGEDELEMENTS_UTILS_H_