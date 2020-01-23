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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace detail {

struct RankedShapeTypeStorage : public TypeStorage {
  struct KeyTy {
    KeyTy(ArrayRef<int64_t> offsetDims, Type dimType,
          VectorType dynamicDimsType)
        : offsetDims(offsetDims),
          dimType(dimType.cast<IntegerType>()),
          dynamicDimsType(dynamicDimsType) {}
    bool operator==(const KeyTy &other) const {
      return dimType == dimType && dynamicDimsType == other.dynamicDimsType &&
             offsetDims.equals(other.offsetDims);
    }
    unsigned getHashValue() const {
      return llvm::hash_combine(
          dimType, dynamicDimsType,
          llvm::hash_combine_range(offsetDims.begin(), offsetDims.end()));
    }
    ArrayRef<int64_t> offsetDims;
    IntegerType dimType;
    VectorType dynamicDimsType;
  };

  RankedShapeTypeStorage(const KeyTy &key) : key(key) {}
  static RankedShapeTypeStorage *construct(TypeStorageAllocator &allocator,
                                           KeyTy key) {
    key.offsetDims = allocator.copyInto(key.offsetDims);
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

static void computeOffsetDims(Type dimType, ArrayRef<int64_t> dims,
                              SmallVectorImpl<int64_t> &offsetDims,
                              VectorType &dynamicDimsType) {
  // Compute offset dims.
  offsetDims.resize(dims.size());
  int64_t dynamicOffset = 0;
  int64_t dynamicDimCount = 0;
  for (size_t i = 0, e = dims.size(); i < e; ++i) {
    auto dim = dims[i];
    if (dim >= 0) {
      // Static dim.
      offsetDims[i] = dim;
    } else {
      // Dynamic dim.
      offsetDims[i] = --dynamicOffset;
      dynamicDimCount += 1;
    }
  }

  // Dynamic dims type.
  if (dynamicDimCount == 0)
    dynamicDimsType = nullptr;
  else
    dynamicDimsType = VectorType::get({dynamicDimCount}, dimType);
}

RankedShapeType RankedShapeType::get(ArrayRef<int64_t> dims, Type dimType) {
  VectorType dynamicDimsType;
  SmallVector<int64_t, 7> offsetDims;
  computeOffsetDims(dimType, dims, offsetDims, dynamicDimsType);
  return Base::get(dimType.getContext(), IREE::Shape::TypeKind::RankedShape,
                   offsetDims, dimType, dynamicDimsType);
}

RankedShapeType RankedShapeType::getChecked(ArrayRef<int64_t> dims,
                                            Type dimType, Location loc) {
  VectorType dynamicDimsType;
  SmallVector<int64_t, 7> offsetDims;
  computeOffsetDims(dimType, dims, offsetDims, dynamicDimsType);
  return Base::getChecked(loc, dimType.getContext(),
                          IREE::Shape::TypeKind::RankedShape, offsetDims,
                          dimType, dynamicDimsType);
}

LogicalResult RankedShapeType::verifyConstructionInvariants(
    Optional<Location> loc, MLIRContext *context, ArrayRef<int64_t> dims,
    Type dimType, VectorType dynamicDimsType) {
  if (!dimType) {
    return emitOptionalError(loc, "RankedShapeType must have a dim type");
  }
  if (!dimType.isa<IntegerType>()) {
    return emitOptionalError(loc,
                             "RankedShapeType must have an integral dim type");
  }
  return success();
}

IntegerType RankedShapeType::getDimType() const {
  return getImpl()->key.dimType;
}

VectorType RankedShapeType::getDynamicDimsType() const {
  return getImpl()->key.dynamicDimsType;
}

int64_t RankedShapeType::getRank() const {
  return getImpl()->key.offsetDims.size();
}

bool RankedShapeType::isFullyStatic() const {
  for (auto dim : getImpl()->key.offsetDims) {
    if (dim < 0) return false;
  }
  return true;
}

void RankedShapeType::getAllDims(SmallVectorImpl<int64_t> &dims) {
  dims.clear();
  for (auto offsetDim : getImpl()->key.offsetDims) {
    if (offsetDim < 0)
      dims.push_back(-1);
    else
      dims.push_back(offsetDim);
  }
}

bool RankedShapeType::isDimDynamic(int allDimsIndex) {
  assert(allDimsIndex >= 0 && allDimsIndex < getImpl()->key.offsetDims.size());
  return getImpl()->key.offsetDims[allDimsIndex] < 0;
}

int64_t RankedShapeType::getStaticDim(int allDimsIndex) {
  assert(allDimsIndex >= 0 && allDimsIndex < getImpl()->key.offsetDims.size());
  auto dim = getImpl()->key.offsetDims[allDimsIndex];
  assert(dim >= 0 && "getStaticDim() called on dynamic dimension");
  return dim;
}

unsigned RankedShapeType::getDynamicDimIndex(int allDimsIndex) {
  assert(allDimsIndex >= 0 && allDimsIndex < getImpl()->key.offsetDims.size());
  auto dim = getImpl()->key.offsetDims[allDimsIndex];
  assert(dim < 0 && "getDynamicDimIndex() called on static dimension");
  return -(dim + 1);  // negative offset dim -1 == dynamic index 0
}
