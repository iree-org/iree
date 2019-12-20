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

#ifndef IREE_COMPILER_UTILS_TYPEUTILS_H_
#define IREE_COMPILER_UTILS_TYPEUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

static const int kBoolBitWidth = 8;
static const int kIndexBitWidth = 32;

// Converts types to MemRefs using convertTypeToMemRef.
class MemRefTypeConverter : public TypeConverter {
 public:
  explicit MemRefTypeConverter(MLIRContext *context) {}
  Type convertType(Type type) override;
};

class LLTypeConverter : public TypeConverter {
 public:
  explicit LLTypeConverter(MLIRContext *context) {}
  Type convertType(Type type) override;
};

Type legalizeType(Type type);

// Converts a type (scalar, tensor, etc) to a MemRef-based type.
MemRefType convertTypeToMemRef(Type type);
MemRefType convertTypeToMemRef(ValuePtr value);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_TYPEUTILS_H_
