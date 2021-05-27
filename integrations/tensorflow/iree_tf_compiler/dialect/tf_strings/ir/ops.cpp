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

#include "iree_tf_compiler/dialect/tf_strings/ir/ops.h"

#include "iree_tf_compiler/dialect/tf_strings/ir/types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "iree_tf_compiler/dialect/tf_strings/ir/ops.cpp.inc"
#undef GET_OP_CLASSES

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

void ToStringOp::build(OpBuilder& builder, OperationState& tblgen_state,
                       Value value) {
  build(builder, tblgen_state, StringType::get(builder.getContext()), value);
}

void ToStringTensorOp::build(OpBuilder& builder, OperationState& tblgen_state,
                             Value value) {
  if (auto type = value.getType().dyn_cast<ShapedType>()) {
    auto new_type = RankedTensorType::get(
        type.getShape(), StringType::get(builder.getContext()));
    build(builder, tblgen_state, new_type, value);
    return;
  }
  llvm_unreachable("Invalid input to ToStringTensorOp");
}

void StringTensorToStringOp::build(OpBuilder& builder,
                                   OperationState& tblgen_state, Value value) {
  build(builder, tblgen_state, StringType::get(builder.getContext()), value);
}

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir
