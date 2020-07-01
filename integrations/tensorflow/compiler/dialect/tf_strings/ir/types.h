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

// IREE ops for working with buffers and buffer views.
// These are used by common transforms between the sequencer and interpreter and
// allow us to share some of the common lowering passes from other dialects.

#ifndef INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_TYPES_H_
#define INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_TYPES_H_

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace tf_strings {

namespace TFStringsTypes {
enum Kind {
  FIRST_USED_STRINGS_TYPE = Type::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE,
  String,
  LAST_USED_STRINGS_TYPE,
};
}  // namespace TFStringsTypes

class TFStringsType : public Type {
 public:
  using Type::Type;

  static bool classof(Type type) {
    return type.getKind() >= TFStringsTypes::FIRST_USED_STRINGS_TYPE &&
           type.getKind() <= TFStringsTypes::LAST_USED_STRINGS_TYPE;
  }
};

class StringType
    : public Type::TypeBase<StringType, TFStringsType, TypeStorage> {
 public:
  using Base::Base;
  static StringType get(MLIRContext* context) {
    return Base::get(context, TFStringsTypes::String);
  }

  static bool kindof(unsigned kind) { return kind == TFStringsTypes::String; }
};

}  // namespace tf_strings
}  // namespace iree_compiler
}  // namespace mlir

#endif  // INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_TYPES_H_
