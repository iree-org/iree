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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_tf_tensorlist_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_tf_tensorlist_H_

#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_types.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_integrations {
namespace tf_tensorlist {

class TFTensorListDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_tensorlist"; }
  explicit TFTensorListDialect(MLIRContext *context);
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &printer) const override;
};

}  // namespace tf_tensorlist
}  // namespace iree_integrations
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_tf_tensorlist_H_
