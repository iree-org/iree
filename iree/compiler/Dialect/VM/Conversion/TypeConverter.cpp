// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

VMTypeConverter::VMTypeConverter() {
  // All ref_ptr types are passed through unmodified.
  addConversion([](IREE::VM::RefType type) { return type; });
  // Wrap ref types.
  addConversion([](Type type) -> Optional<Type> {
    if (IREE::VM::RefType::isCompatible(type)) {
      return IREE::VM::RefType::get(type);
    }
    return llvm::None;
  });
  // Convert integer types.
  addConversion([](IntegerType integerType) -> Optional<Type> {
    if (integerType.isSignlessInteger(32)) {
      return integerType;
    } else if (integerType.isInteger(1)) {
      // Promote i1 -> i32.
      return IntegerType::get(32, integerType.getContext());
    } else if (integerType.isSignlessInteger()) {
      // TODO(benvanik): ensure we want to do this and not something that forces
      // materialization of trunc/ext.
      return IntegerType::get(32, integerType.getContext());
    }
    return llvm::None;
  });
  // All ref_ptr types are passed through unmodified.
  addConversion([this](IREE::PtrType type) -> Type {
    // Recursively handle pointer target types (we want to convert ptr<index> to
    // ptr<i32>, for example).
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return Type();
    }
    return IREE::PtrType::get(targetType);
  });
  // Convert ranked shape types.
  addConversion(
      [](Shape::RankedShapeType rankedShape, SmallVectorImpl<Type> &results) {
        for (int i = 0; i < rankedShape.getRank(); ++i) {
          if (rankedShape.isDimDynamic(i)) {
            results.push_back(rankedShape.getDimType());
          }
        }
        return success();
      });
}

Operation *VMTypeConverter::materializeConversion(PatternRewriter &rewriter,
                                                  Type resultType,
                                                  ArrayRef<Value> inputs,
                                                  Location loc) {
  // TODO(b/145876978): materialize conversion when this is called.
  llvm_unreachable("unhandled materialization");
  return nullptr;
}

}  // namespace iree_compiler
}  // namespace mlir
