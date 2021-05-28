// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
