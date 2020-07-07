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

#include "iree/compiler/Dialect/Sequence/IR/SequenceTypes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeSupport.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Sequence {
namespace detail {

struct SequenceTypeStorage : public TypeStorage {
  SequenceTypeStorage(Type targetType, unsigned subclassData = 0)
      : TypeStorage(subclassData), targetType(targetType) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == targetType; }

  static SequenceTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<SequenceTypeStorage>())
        SequenceTypeStorage(key);
  }

  Type targetType;
};

}  // namespace detail

SequenceType SequenceType::get(Type targetType) {
  assert(targetType && "sequence targetType required");
  return Base::get(targetType.getContext(), TypeKind::Sequence, targetType);
}

SequenceType SequenceType::getChecked(Type targetType, Location location) {
  if (!targetType) {
    emitError(location) << "null target type: " << targetType;
    return SequenceType();
  }
  return Base::getChecked(location, TypeKind::Sequence, targetType);
}

Type SequenceType::getTargetType() { return getImpl()->targetType; }

}  // namespace Sequence
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
