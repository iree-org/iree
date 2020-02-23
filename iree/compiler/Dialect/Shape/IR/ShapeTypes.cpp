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

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"

#include "llvm/ADT/Twine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace detail {

struct RankedShapeTypeStorage : public TypeStorage {
  struct KeyTy {
    KeyTy(ArrayRef<int64_t> dims, Type dimType)
        : dims(dims), dimType(dimType) {}
    bool operator==(const KeyTy &other) const {
      return dimType == dimType && dims.equals(other.dims);
    }
    unsigned getHashValue() const {
      return llvm::hash_combine(
          dimType, llvm::hash_combine_range(dims.begin(), dims.end()));
    }
    ArrayRef<int64_t> dims;
    Type dimType;
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

RankedShapeType RankedShapeType::get(ArrayRef<int64_t> dims, Type dimType) {
  return Base::get(dimType.getContext(), IREE::Shape::TypeKind::RankedShape,
                   dims, dimType);
}

RankedShapeType RankedShapeType::getChecked(ArrayRef<int64_t> dims,
                                            Type dimType, Location loc) {
  return Base::getChecked(loc, IREE::Shape::TypeKind::RankedShape, dims,
                          dimType);
}

LogicalResult RankedShapeType::verifyConstructionInvariants(
    Location loc, ArrayRef<int64_t> dims, Type dimType) {
  for (auto dim : dims) {
    if (dim < 0 && dim != -1) {
      return emitError(loc, "dims must be -1 for dynamic");
    }
  }
  if (!dimType) {
    return emitError(loc, "RankedShapeType must have a dim type");
  }
  if (!dimType.isa<IntegerType>() && !dimType.isa<IndexType>()) {
    return emitError(loc,
                     "RankedShapeType must have an integral or index "
                     "dim type");
  }
  return success();
}

Type RankedShapeType::getDimType() const { return getImpl()->key.dimType; }

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
