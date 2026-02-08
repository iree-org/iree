// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @string_format_single_arg
util.func public @string_format_single_arg(%idx : index) -> !util.buffer {
  // CHECK: util.string.format "blk.{}.attn_q.weight"(%arg0) : (index) -> !util.buffer
  %key = util.string.format "blk.{}.attn_q.weight"(%idx) : (index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_multi_arg
util.func public @string_format_multi_arg(%a : index, %b : index) -> !util.buffer {
  // CHECK: util.string.format "blk.{}.ffn.{}.weight"(%arg0, %arg1) : (index, index) -> !util.buffer
  %key = util.string.format "blk.{}.ffn.{}.weight"(%a, %b) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_no_args
util.func public @string_format_no_args() -> !util.buffer {
  // CHECK: util.string.format "static_key"() : () -> !util.buffer
  %key = util.string.format "static_key"() : () -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_explicit_ordinals
util.func public @string_format_explicit_ordinals(%a : index, %b : index) -> !util.buffer {
  // CHECK: util.string.format "y={1} x={0}"(%arg0, %arg1) : (index, index) -> !util.buffer
  %key = util.string.format "y={1} x={0}"(%a, %b) : (index, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_escaped_braces
util.func public @string_format_escaped_braces() -> !util.buffer {
  // CHECK: util.string.format "hello {{[{][{]}}world{{[}][}]}}"() : () -> !util.buffer
  %key = util.string.format "hello {{world}}"() : () -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_buffer_arg
util.func public @string_format_buffer_arg(%scope : !util.buffer, %idx : index) -> !util.buffer {
  // CHECK: util.string.format "{}::blk.{}.weight"(%arg0, %arg1) : (!util.buffer, index) -> !util.buffer
  %key = util.string.format "{}::blk.{}.weight"(%scope, %idx) : (!util.buffer, index) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_format_i32_arg
util.func public @string_format_i32_arg(%v : i32) -> !util.buffer {
  // CHECK: util.string.format "layer_{}"(%arg0) : (i32) -> !util.buffer
  %key = util.string.format "layer_{}"(%v) : (i32) -> !util.buffer
  util.return %key : !util.buffer
}

// -----

// CHECK-LABEL: @string_itoa_index
util.func public @string_itoa_index(%v : index) -> !util.buffer {
  // CHECK: util.string.itoa %arg0 : index -> !util.buffer
  %str = util.string.itoa %v : index -> !util.buffer
  util.return %str : !util.buffer
}

// -----

// CHECK-LABEL: @string_itoa_i32
util.func public @string_itoa_i32(%v : i32) -> !util.buffer {
  // CHECK: util.string.itoa %arg0 : i32 -> !util.buffer
  %str = util.string.itoa %v : i32 -> !util.buffer
  util.return %str : !util.buffer
}

// -----

// CHECK-LABEL: @string_itoa_i64
util.func public @string_itoa_i64(%v : i64) -> !util.buffer {
  // CHECK: util.string.itoa %arg0 : i64 -> !util.buffer
  %str = util.string.itoa %v : i64 -> !util.buffer
  util.return %str : !util.buffer
}
