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

#ifndef IREE_COMPILER_DIALECT_SEQUENCE_IR_SEQUENCEDIALECT_H_
#define IREE_COMPILER_DIALECT_SEQUENCE_IR_SEQUENCEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Sequence {

class SequenceDialect : public Dialect {
 public:
  explicit SequenceDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "sequence"; }

  static bool isDialectOp(Operation* op) {
    return op && op->getDialect() &&
           op->getDialect()->getNamespace() == getDialectNamespace();
  }

  Type parseType(DialectAsmParser& parser) const override;
  void printType(Type type, DialectAsmPrinter& os) const override;
};

}  // namespace Sequence
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SEQUENCE_IR_SEQUENCEDIALECT_H_
