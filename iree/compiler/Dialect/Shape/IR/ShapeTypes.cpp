// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace detail {

struct RankedShapeTypeStorage : public TypeStorage {
  struct KeyTy {
    KeyTy(ArrayRef<int64_t> dims) : dims(dims) {}
    bool operator==(const KeyTy &other) const {
      return dims.equals(other.dims);
    }
    unsigned getHashValue() const {
      return llvm::hash_combine_range(dims.begin(), dims.end());
    }
    ArrayRef<int64_t> dims;
  };

  RankedShapeTypeStorage(const KeyTy &key) : key(key) {}
  static RankedShapeTypeStorage *construct(TypeStorageAllocator &allocator,
                                           KeyTy key) {
    key.dims = allocator.copyInto(key.dims);
    return new (allocator.allocate<RankedShapeTypeStorage>())
        RankedShapeTypeStorage(key);
  }

  bool operator==(const KeyTy &otherKey) const { return key == otherKey; }
  static unsigned hashKey(const KeyTy &key) { return key.getHashValue(); }

  KeyTy key;
};

}  // namespace detail
}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

using namespace mlir;
using namespace mlir::iree_compiler::Shape;

//===----------------------------------------------------------------------===//
// RankedShapeType
//===----------------------------------------------------------------------===//

RankedShapeType RankedShapeType::get(ArrayRef<int64_t> dims,
                                     MLIRContext *context) {
  return Base::get(context, dims);
}

RankedShapeType RankedShapeType::getChecked(ArrayRef<int64_t> dims,
                                            Location loc) {
  return Base::getChecked(loc, loc.getContext(), dims);
}

RankedShapeType RankedShapeType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<int64_t> dims) {
  return Base::getChecked(emitError, context, dims);
}

RankedShapeType RankedShapeType::get(ShapedType shapedType) {
  return Base::get(shapedType.getContext(), shapedType.getShape());
}

LogicalResult RankedShapeType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> dims) {
  for (auto dim : dims) {
    if (dim < 0 && dim != -1) {
      return emitError() << "dims must be -1 for dynamic";
    }
  }
  return success();
}

int64_t RankedShapeType::getRank() const { return getImpl()->key.dims.size(); }

bool RankedShapeType::isFullyStatic() const {
  for (auto dim : getImpl()->key.dims) {
    if (dim < 0) return false;
  }
  return true;
}

ArrayRef<int64_t> RankedShapeType::getAllDims() const {
  return getImpl()->key.dims;
}

unsigned RankedShapeType::getNumDynamicDims() const {
  auto allDims = getAllDims();
  return std::count_if(allDims.begin(), allDims.end(),
                       [](int64_t dim) { return dim < 0; });
}

bool RankedShapeType::isDimDynamic(int allDimsIndex) const {
  assert(allDimsIndex >= 0 && allDimsIndex < getImpl()->key.dims.size());
  return getImpl()->key.dims[allDimsIndex] < 0;
}

int64_t RankedShapeType::getStaticDim(int allDimsIndex) const {
  assert(allDimsIndex >= 0 && allDimsIndex < getRank());
  auto dim = getAllDims()[allDimsIndex];
  assert(dim >= 0 && "getStaticDim() called on dynamic dimension");
  return dim;
}

//===----------------------------------------------------------------------===//
// ShapeDialect
//===----------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {
void ShapeDialect::registerTypes() { addTypes<Shape::RankedShapeType>(); }
}  // namespace iree_compiler
}  // namespace mlir
