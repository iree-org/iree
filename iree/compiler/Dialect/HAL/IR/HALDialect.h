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

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALDIALECT_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class HALDialect : public Dialect {
 public:
  explicit HALDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "hal"; }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_IR_HALDIALECT_H_
