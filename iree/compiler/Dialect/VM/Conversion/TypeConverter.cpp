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
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/StandardTypes.h"

#define DEBUG_TYPE "iree-vm"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

TypeConverter::TypeConverter(TargetOptions targetOptions)
    : targetOptions_(targetOptions) {
  // All ref_ptr types are passed through unmodified.
  addConversion([](IREE::VM::RefType type) { return type; });
  // Wrap ref types.
  addConversion([](Type type) -> Optional<Type> {
    if (RefType::isCompatible(type)) {
      return RefType::get(type);
    }
    return llvm::None;
  });

  // Pointer types remain as pointer types types are passed through unmodified.
  addConversion([this](IREE::PtrType type) -> Optional<Type> {
    // Recursively handle pointer target types (we want to convert ptr<index> to
    // ptr<i32>, for example).
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return llvm::None;
    }
    return IREE::PtrType::get(targetType);
  });

  // Convert integer types.
  addConversion([this](IntegerType integerType) -> Optional<Type> {
    if (integerType.isInteger(32)) {
      // i32 is always supported by the runtime.
      return integerType;
    } else if (integerType.getIntOrFloatBitWidth() < 32) {
      // Promote i1/i8/i16 -> i32.
      return IntegerType::get(32, integerType.getContext());
    } else if (integerType.isInteger(64)) {
      if (targetOptions_.i64Extension) {
        // i64 is supported by the VM, use directly.
        return integerType;
      } else if (targetOptions_.truncateUnsupportedIntegers) {
        // i64 is not supported and we still want to compile, so truncate to i32
        // (unsafe if all bits are actually required!).
        return IntegerType::get(32, integerType.getContext());
      }
    }
    return llvm::None;
  });

  // Convert index types to the target bit width.
  addConversion([this](IndexType indexType) -> Optional<Type> {
    return IntegerType::get(targetOptions_.indexBits, indexType.getContext());
  });

  // Convert ranked shape types (expanding all dims).
  addConversion([this](Shape::RankedShapeType rankedShape,
                       SmallVectorImpl<Type> &results) {
    auto indexType =
        IntegerType::get(targetOptions_.indexBits, rankedShape.getContext());
    for (int i = 0; i < rankedShape.getRank(); ++i) {
      if (rankedShape.isDimDynamic(i)) {
        results.push_back(indexType);
      }
    }
    return success();
  });

  // TODO(b/145876978): materialize conversion for other types
  addArgumentMaterialization([](OpBuilder &builder,
                                Shape::RankedShapeType resultType,
                                ValueRange inputs, Location loc) -> Value {
    LLVM_DEBUG(llvm::dbgs()
               << "MATERIALIZE CONVERSION: " << resultType << "\n");
    return builder.create<Shape::MakeRankedShapeOp>(loc, resultType, inputs);
  });
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
