// RUN: iree-opt --split-input-file --iree-consteval-jit-target-backend=vmvx --verify-diagnostics --iree-consteval-jit-debug --iree-consteval-jit-globals  %s | FileCheck %s

// CHECK-LABEL: @eval_i8_scalar
// CHECK: 42 : i8
module @eval_i8_scalar {
  util.global private @offset : i8 = -2 : i8
  util.global private @hoisted : i8
  func.func @main() -> i8 {
    %hoisted = util.global.load @hoisted : i8
    return %hoisted : i8
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant 44 : i8
    %offset = util.global.load @offset : i8
    %sum = arith.addi %cst, %offset : i8
    util.global.store %sum, @hoisted : i8
    util.initializer.return
  }
}

// -----
// CHECK-LABEL: @eval_i16_scalar
// CHECK: 42 : i16
module @eval_i16_scalar {
  util.global private @offset : i16 = -2 : i16
  util.global private @hoisted : i16
  func.func @main() -> i16 {
    %hoisted = util.global.load @hoisted : i16
    return %hoisted : i16
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant 44 : i16
    %offset = util.global.load @offset : i16
    %sum = arith.addi %cst, %offset : i16
    util.global.store %sum, @hoisted : i16
    util.initializer.return
  }
}

// -----
// CHECK-LABEL: @eval_i32_scalar
// CHECK: 42 : i32
module @eval_i32_scalar {
  util.global private @offset : i32 = -2 : i32
  util.global private @hoisted : i32
  func.func @main() -> i32 {
    %hoisted = util.global.load @hoisted : i32
    return %hoisted : i32
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant 44 : i32
    %offset = util.global.load @offset : i32
    %sum = arith.addi %cst, %offset : i32
    util.global.store %sum, @hoisted : i32
    util.initializer.return
  }
}

// -----
// CHECK-LABEL: @eval_i64_scalar
// CHECK: 42 : i64
module @eval_i64_scalar {
  util.global private @offset : i64 = -2 : i64
  util.global private @hoisted : i64
  func.func @main() -> i64 {
    %hoisted = util.global.load @hoisted : i64
    return %hoisted : i64
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant 44 : i64
    %offset = util.global.load @offset : i64
    %sum = arith.addi %cst, %offset : i64
    util.global.store %sum, @hoisted : i64
    util.initializer.return
  }
}

// -----
// CHECK-LABEL: @eval_f32_scalar
// CHECK: 4.200000e+01 : f32
module @eval_f32_scalar {
  util.global private @offset : f32 = -2.0 : f32
  util.global private @hoisted : f32
  func.func @main() -> f32 {
    %hoisted = util.global.load @hoisted : f32
    return %hoisted : f32
  }
  util.initializer attributes {iree.compiler.consteval} {
    %cst = arith.constant 44.0 : f32
    %offset = util.global.load @offset : f32
    %sum = arith.addf %cst, %offset : f32
    util.global.store %sum, @hoisted : f32
    util.initializer.return
  }
}
