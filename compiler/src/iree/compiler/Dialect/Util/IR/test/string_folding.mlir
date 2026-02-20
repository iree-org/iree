// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @itoa_fold_constant
util.func public @itoa_fold_constant() -> !util.buffer {
  %c42 = arith.constant 42 : index
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "42"
  %str = util.string.itoa %c42 : index -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %str : !util.buffer
}

// -----

// CHECK-LABEL: @itoa_fold_zero
util.func public @itoa_fold_zero() -> !util.buffer {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "0"
  %str = util.string.itoa %c0 : index -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %str : !util.buffer
}

// -----

// CHECK-LABEL: @itoa_no_fold_dynamic
util.func public @itoa_no_fold_dynamic(%v : index) -> !util.buffer {
  // CHECK-SAME: (%[[V:.+]]: index)
  // CHECK: util.string.itoa %[[V]] : index -> !util.buffer
  %str = util.string.itoa %v : index -> !util.buffer
  util.return %str : !util.buffer
}

// -----

// CHECK-LABEL: @format_fold_all_constant
util.func public @format_fold_all_constant() -> !util.buffer {
  %c5 = arith.constant 5 : index
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "blk.5.attn_q.weight"
  %key = util.string.format "blk.{}.attn_q.weight"(%c5) : (index) -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_fold_no_args
util.func public @format_fold_no_args() -> !util.buffer {
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "static_key"
  %key = util.string.format "static_key"() : () -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_partial_fold
util.func public @format_partial_fold(%dynamic : index) -> !util.buffer {
  // CHECK-SAME: (%[[DYNAMIC:.+]]: index)
  %c5 = arith.constant 5 : index
  // CHECK: util.string.format "blk.5.layer.{}.weight"(%[[DYNAMIC]]) : (index) -> !util.buffer
  %key = util.string.format "blk.{}.layer.{}.weight"(%c5, %dynamic) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_partial_fold_last_arg
util.func public @format_partial_fold_last_arg(%dynamic : index) -> !util.buffer {
  // CHECK-SAME: (%[[DYNAMIC:.+]]: index)
  %c7 = arith.constant 7 : index
  // CHECK: util.string.format "{}.layer.7"(%[[DYNAMIC]]) : (index) -> !util.buffer
  %key = util.string.format "{}.layer.{}"(%dynamic, %c7) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_fold_buffer_arg
util.func public @format_fold_buffer_arg() -> !util.buffer {
  %scope = util.buffer.constant : !util.buffer = "model"
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "model::blk.weight"
  %key = util.string.format "{}::blk.weight"(%scope) : (!util.buffer) -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_partial_fold_mixed_buffer_int
util.func public @format_partial_fold_mixed_buffer_int(%idx : index) -> !util.buffer {
  // CHECK-SAME: (%[[IDX:.+]]: index)
  %scope = util.buffer.constant : !util.buffer = "model"
  // CHECK: util.string.format "model::blk.{}.weight"(%[[IDX]]) : (index) -> !util.buffer
  %key = util.string.format "{}::blk.{}.weight"(%scope, %idx) : (!util.buffer, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_fold_escaped_braces
util.func public @format_fold_escaped_braces() -> !util.buffer {
  %c1 = arith.constant 1 : index
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "a{1}b"
  %key = util.string.format "a{{{}}}b"(%c1) : (index) -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_no_fold_all_dynamic
util.func public @format_no_fold_all_dynamic(%a : index, %b : index) -> !util.buffer {
  // CHECK-SAME: (%[[A:.+]]: index, %[[B:.+]]: index)
  // CHECK: util.string.format "blk.{}.layer.{}.weight"(%[[A]], %[[B]]) : (index, index) -> !util.buffer
  %key = util.string.format "blk.{}.layer.{}.weight"(%a, %b) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @format_identity_buffer_elimination
util.func public @format_identity_buffer_elimination(%buf : !util.buffer) -> !util.buffer {
  // CHECK-SAME: (%[[BUF:.+]]: !util.buffer)
  // CHECK-NOT: util.string.format
  // CHECK: util.return %[[BUF]]
  %result = util.string.format "{}"(%buf) : (!util.buffer) -> !util.buffer
  util.return %result : !util.buffer
}

// -----

// CHECK-LABEL: @format_single_int_to_itoa
util.func public @format_single_int_to_itoa(%num : index) -> !util.buffer {
  // CHECK-SAME: (%[[NUM:.+]]: index)
  // CHECK: %[[ITOA:.+]] = util.string.itoa %[[NUM]] : index -> !util.buffer
  // CHECK: util.return %[[ITOA]]
  %result = util.string.format "{}"(%num) : (index) -> !util.buffer
  util.return %result : !util.buffer
}

// -----

// CHECK-LABEL: @itoa_fold_through_extui
util.func public @itoa_fold_through_extui() -> !util.buffer {
  %c42 = arith.constant 42 : i8
  %wide = arith.extui %c42 : i8 to i32
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "42"
  %str = util.string.itoa %wide : i32 -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %str : !util.buffer
}

// -----

// CHECK-LABEL: @itoa_fold_through_index_cast
util.func public @itoa_fold_through_index_cast() -> !util.buffer {
  %c42 = arith.constant 42 : index
  %i32 = arith.index_cast %c42 : index to i32
  // CHECK: %[[CST:.+]] = util.buffer.constant : !util.buffer = "42"
  %str = util.string.itoa %i32 : i32 -> !util.buffer
  // CHECK: util.return %[[CST]]
  util.return %str : !util.buffer
}

// -----

// Ordinal normalization is handled by partial fold, which rebuilds format
// strings with sequential placeholders and reorders args to match segment order.
// CHECK-LABEL: @format_normalize_explicit_ordinals
util.func public @format_normalize_explicit_ordinals(%a : index, %b : index) -> !util.buffer {
  // CHECK-SAME: (%[[A:.+]]: index, %[[B:.+]]: index)
  // CHECK: util.string.format "{}.{}"(%[[A]], %[[B]]) : (index, index) -> !util.buffer
  %key = util.string.format "{0}.{1}"(%a, %b) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// Out-of-order ordinals get canonicalized by partial fold (args reordered).
// CHECK-LABEL: @format_reorder_out_of_order_ordinals
util.func public @format_reorder_out_of_order_ordinals(%a : index, %b : index) -> !util.buffer {
  // CHECK-SAME: (%[[A:.+]]: index, %[[B:.+]]: index)
  // CHECK: util.string.format "{}.{}"(%[[B]], %[[A]]) : (index, index) -> !util.buffer
  %key = util.string.format "{1}.{0}"(%a, %b) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// Duplicate ordinals get canonicalized by partial fold (args duplicated).
// CHECK-LABEL: @format_duplicate_ordinals
util.func public @format_duplicate_ordinals(%a : index, %b : index) -> !util.buffer {
  // CHECK-SAME: (%[[A:.+]]: index, %[[B:.+]]: index)
  // CHECK: util.string.format "{}.{}.{}"(%[[B]], %[[B]], %[[A]]) : (index, index, index) -> !util.buffer
  %key = util.string.format "{1}.{1}.{0}"(%a, %b) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}
