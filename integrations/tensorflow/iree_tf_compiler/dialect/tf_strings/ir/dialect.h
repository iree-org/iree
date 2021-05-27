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

#ifndef INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_DIALECT_H_
#define INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

#include "iree_tf_compiler/dialect/tf_strings/ir/op_interface.h.inc"

class TFStringsDialect : public Dialect {
 public:
  explicit TFStringsDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "tf_strings"; }

  Type parseType(DialectAsmParser& parser) const override;

  void printType(Type type, DialectAsmPrinter& os) const override;
};

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir

#endif  // INTEGRATIONS_TENSORFLOW_COMPILER_DIALECT_TFSTRINGS_IR_DIALECT_H_
