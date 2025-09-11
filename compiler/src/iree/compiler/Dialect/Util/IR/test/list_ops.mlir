// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @list_init_ops
util.func public @list_init_ops() {
  // CHECK: %[[CAPACITY:.+]] = arith.constant 5
  %capacity = arith.constant 5 : index
  // CHECK: = util.list.create %[[CAPACITY]] : !util.list<?>
  %list_initial_capacity = util.list.create %capacity : !util.list<?>

  // CHECK: %[[LIST:.+]] = util.list.create : !util.list<?>
  %list = util.list.create : !util.list<?>

  // CHECK: %[[NEW_SIZE:.+]] = arith.constant 100
  %new_size = arith.constant 100 : index
  // CHECK: util.list.resize %[[LIST]], %[[NEW_SIZE]] : !util.list<?>
  util.list.resize %list, %new_size : !util.list<?>

  util.return
}

// -----

// CHECK-LABEL: @list_construct
util.func public @list_construct() {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : i64
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : i64

  // CHECK: %[[EMPTY:.+]] = util.list.construct [] : !util.list<?>
  %empty = util.list.construct [] : !util.list<?>

  // CHECK: = util.list.construct [%[[C0]] : i64, %[[C1]] : i64] : !util.list<i64>
  %list_i64 = util.list.construct [%c0 : i64, %c1 : i64] : !util.list<i64>

  // CHECK: = util.list.construct [%[[C0]] : i64, %[[EMPTY]] : !util.list<?>] : !util.list<?>
  %list_any = util.list.construct [%c0 : i64, %empty : !util.list<?>] : !util.list<?>

  util.return
}

// -----

util.func public @list_construct_type_mismatch() {
  %c0 = arith.constant 0 : i64
  // @expected-error@+1 {{list[0] type 'i64' cannot be be cast to list type 'i32'}}
  util.list.construct [%c0 : i64] : !util.list<i32>
  util.return
}

// -----

// CHECK-LABEL: @list_access
// CHECK-SAME: (%[[LIST:.+]]: !util.list<i32>)
util.func public @list_access(%list: !util.list<i32>) {
  %c10 = arith.constant 10 : index

  // CHECK: = util.list.get %[[LIST]][%c10] : !util.list<i32>
  %0 = util.list.get %list[%c10] : !util.list<i32>
  // CHECK: = util.list.get %[[LIST]][%c10] : !util.list<i32>
  %1 = util.list.get %list[%c10] : !util.list<i32> -> i32

  // CHECK: %[[NEW_VALUE:.+]] = arith.constant 100 : i32
  %new_value = arith.constant 100 : i32
  // CHECK: util.list.set %[[LIST]][%c10], %[[NEW_VALUE]] : !util.list<i32>
  util.list.set %list[%c10], %new_value : !util.list<i32>

  util.return
}

// -----

// CHECK-LABEL: @list_access_tensor
// CHECK-SAME: (%[[LIST:.+]]: !util.list<tensor<*xf32>>)
util.func public @list_access_tensor(%list: !util.list<tensor<*xf32>>) {
  %c10 = arith.constant 10 : index

  // CHECK: = util.list.get %[[LIST]][%c10] : !util.list<tensor<*xf32>> -> tensor<?xf32>
  %0 = util.list.get %list[%c10] : !util.list<tensor<*xf32>> -> tensor<?xf32>

  // CHECK: %[[NEW_VALUE:.+]] = arith.constant dense<1> : tensor<5xi32>
  %new_value = arith.constant dense<1> : tensor<5xi32>
  // CHECK: util.list.set %[[LIST]][%c10], %[[NEW_VALUE]] : tensor<5xi32> -> !util.list<tensor<*xf32>>
  util.list.set %list[%c10], %new_value : tensor<5xi32> -> !util.list<tensor<*xf32>>

  util.return
}

// -----

// CHECK-LABEL: @list_access_variant
// CHECK-SAME: (%[[LIST:.+]]: !util.list<?>)
util.func public @list_access_variant(%list: !util.list<?>) {
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index

  // CHECK: = util.list.get %[[LIST]][%c10] : !util.list<?> -> i32
  %0 = util.list.get %list[%c10] : !util.list<?> -> i32

  // CHECK: %[[NEW_I32_VALUE:.+]] = arith.constant 100 : i32
  %new_i32_value = arith.constant 100 : i32
  // CHECK: util.list.set %[[LIST]][%c10], %[[NEW_I32_VALUE]] : i32 -> !util.list<?>
  util.list.set %list[%c10], %new_i32_value : i32 -> !util.list<?>

  // CHECK: = util.list.get %[[LIST]][%c11] : !util.list<?> -> tensor<5xf32>
  %1 = util.list.get %list[%c11] : !util.list<?> -> tensor<5xf32>

  // CHECK: %[[NEW_TENSOR_VALUE:.+]] = arith.constant dense<1> : tensor<5xi32>
  %new_tensor_value = arith.constant dense<1> : tensor<5xi32>
  // CHECK: util.list.set %[[LIST]][%c11], %[[NEW_TENSOR_VALUE]] : tensor<5xi32> -> !util.list<?>
  util.list.set %list[%c11], %new_tensor_value : tensor<5xi32> -> !util.list<?>

  util.return
}
