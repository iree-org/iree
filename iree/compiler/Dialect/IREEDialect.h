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

#ifndef IREE_COMPILER_DIALECT_IREEDIALECT_H_
#define IREE_COMPILER_DIALECT_IREEDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {

// TODO(b/143787186): rename to IREEDialect.
class IREEXDialect : public Dialect {
 public:
  explicit IREEXDialect(MLIRContext* context);
  // TODO(b/143787186): rename to iree.
  static StringRef getDialectNamespace() { return "ireex"; }

  /// Parses a type registered to this dialect.
  Type parseType(DialectAsmParser& parser) const override;

  /// Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter& os) const override;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREEDIALECT_H_
