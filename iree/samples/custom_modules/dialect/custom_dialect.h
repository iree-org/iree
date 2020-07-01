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

#ifndef IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_DIALECT_H_
#define IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_DIALECT_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Custom {

namespace TypeKind {
enum Kind {
  Message = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
}  // namespace TypeKind

class CustomDialect : public Dialect {
 public:
  explicit CustomDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "custom"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;
};

class MessageType : public Type::TypeBase<MessageType, Type, TypeStorage> {
 public:
  using Base::Base;
  static MessageType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Message);
  }
  static bool kindof(unsigned kind) { return kind == TypeKind::Message; }
};

#define GET_OP_CLASSES
#include "iree/samples/custom_modules/dialect/custom_ops.h.inc"

}  // namespace Custom
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CUSTOM_DIALECT_H_
