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

#ifndef IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTTYPES_H_
#define IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTTYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

namespace TypeKind {
enum Kind {
  kTensorList = IREE::TypeKind::FIRST_TENSORLIST_TYPE,
};
}  // namespace TypeKind

class TensorListType
    : public Type::TypeBase<TensorListType, Type, TypeStorage> {
 public:
  using Base::Base;
  static TensorListType get(MLIRContext *context) {
    return Base::get(context, TypeKind::kTensorList);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::kTensorList; }
};

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_TENSORLIST_IR_TENSORLISTTYPES_H_
