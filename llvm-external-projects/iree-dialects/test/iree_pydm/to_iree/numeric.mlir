// RUN: iree-dialects-opt --split-input-file --convert-iree-pydm-to-iree %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL:   func @neg_integer(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32) -> (i32, i32) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : i32, i32
// CHECK:         }
iree_pydm.func @neg_integer(%arg0 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = neg %arg0 : !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL:   func @integer_add(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32,
// CHECK-SAME:                      %[[VAL_1:.*]]: i32) -> (i32, i32) {
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : i32, i32
// CHECK:         }
iree_pydm.func @integer_add(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = apply_binary "add", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @integer_and
// CHECK: arith.andi
iree_pydm.func @integer_and(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = apply_binary "and", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @integer_mul
// CHECK: arith.muli
iree_pydm.func @integer_mul(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = apply_binary "mul", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @integer_lshift
// CHECK: arith.shli
iree_pydm.func @integer_lshift(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = apply_binary "lshift", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @integer_or
// CHECK: arith.ori
iree_pydm.func @integer_or(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = apply_binary "or", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @integer_rshift_signed
// CHECK: arith.shrsi
iree_pydm.func @integer_rshift_signed(%arg0 : !iree_pydm.integer<32>, %arg1 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  %0 = apply_binary "rshift", %arg0, %arg1 : !iree_pydm.integer<32>, !iree_pydm.integer<32> -> !iree_pydm.integer<32>
  return %0 : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @integer_rshift_usigned
// CHECK: arith.shrui
iree_pydm.func @integer_rshift_usigned(%arg0 : !iree_pydm.integer<unsigned 32>, %arg1 : !iree_pydm.integer<unsigned 32>) -> (!iree_pydm.exception_result, !iree_pydm.integer<unsigned 32>) {
  %0 = apply_binary "rshift", %arg0, %arg1 : !iree_pydm.integer<unsigned 32>, !iree_pydm.integer<unsigned 32> -> !iree_pydm.integer<unsigned 32>
  return %0 : !iree_pydm.integer<unsigned 32>
}

// -----
// CHECK-LABEL:   func @real_add(
// CHECK-SAME:                   %[[VAL_0:.*]]: f32,
// CHECK-SAME:                   %[[VAL_1:.*]]: f32) -> (i32, f32) {
// CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : i32, f32
// CHECK:         }
iree_pydm.func @real_add(%arg0 : !iree_pydm.real<f32>, %arg1 : !iree_pydm.real<f32>) -> (!iree_pydm.exception_result, !iree_pydm.real<f32>) {
  %0 = apply_binary "add", %arg0, %arg1 : !iree_pydm.real<f32>, !iree_pydm.real<f32> -> !iree_pydm.real<f32>
  return %0 : !iree_pydm.real<f32>
}

// -----
// CHECK-LABEL: @real_mul
// CHECK: arith.mulf
iree_pydm.func @real_mul(%arg0 : !iree_pydm.real<f32>, %arg1 : !iree_pydm.real<f32>) -> (!iree_pydm.exception_result, !iree_pydm.real<f32>) {
  %0 = apply_binary "mul", %arg0, %arg1 : !iree_pydm.real<f32>, !iree_pydm.real<f32> -> !iree_pydm.real<f32>
  return %0 : !iree_pydm.real<f32>
}

// -----
// CHECK-LABEL: @real_sub
// CHECK: arith.subf
iree_pydm.func @real_sub(%arg0 : !iree_pydm.real<f32>, %arg1 : !iree_pydm.real<f32>) -> (!iree_pydm.exception_result, !iree_pydm.real<f32>) {
  %0 = apply_binary "sub", %arg0, %arg1 : !iree_pydm.real<f32>, !iree_pydm.real<f32> -> !iree_pydm.real<f32>
  return %0 : !iree_pydm.real<f32>
}
