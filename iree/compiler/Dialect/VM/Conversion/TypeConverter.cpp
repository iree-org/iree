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
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

Type VMTypeConverter::convertType(Type t) {
  if (auto integerType = t.dyn_cast<IntegerType>()) {
    if (integerType.isInteger(32)) {
      return t;
    } else if (integerType.isInteger(1)) {
      // Promote i1 -> i32.
      return IntegerType::get(32, t.getContext());
    } else {
      // TODO(benvanik): ensure we want to do this and not something that forces
      // materialization of trunc/ext.
      return IntegerType::get(32, t.getContext());
    }
  } else if (t.isa<IREE::RefPtrType>()) {
    // All ref_ptr types are passed through unmodified.
    return t;
  }

  // Default to not supporting the type. This dialect is very limited
  // with respect to valid types and the above should be expanded as
  // needed.
  return {};
}

Operation *VMTypeConverter::materializeConversion(PatternRewriter &rewriter,
                                                  Type resultType,
                                                  ArrayRef<Value> inputs,
                                                  Location loc) {
  // TODO(b/145876978): materialize conversion when this is called.
  llvm_unreachable("unhandled materialization");
  return nullptr;
}

}  // namespace iree_compiler
}  // namespace mlir
