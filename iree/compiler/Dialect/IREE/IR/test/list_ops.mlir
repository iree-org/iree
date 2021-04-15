// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @list_init_ops
func @list_init_ops() {
  // CHECK: %[[CAPACITY:.+]] = constant 5
  %capacity = constant 5 : index
  // CHECK: = iree.list.create %[[CAPACITY]] : !iree.list<?>
  %list_initial_capacity = iree.list.create %capacity : !iree.list<?>

  // CHECK: %[[LIST:.+]] = iree.list.create : !iree.list<?>
  %list = iree.list.create : !iree.list<?>

  // CHECK: %[[NEW_SIZE:.+]] = constant 100
  %new_size = constant 100 : index
  // CHECK: iree.list.resize %[[LIST]], %[[NEW_SIZE]] : !iree.list<?>
  iree.list.resize %list, %new_size : !iree.list<?>

  return
}

// -----

// CHECK-LABEL: @list_access
// CHECK-SAME: (%[[LIST:.+]]: !iree.list<i32>)
func @list_access(%list: !iree.list<i32>) {
  %c10 = constant 10 : index

  // CHECK: = iree.list.get %[[LIST]][%c10] : !iree.list<i32>
  %0 = iree.list.get %list[%c10] : !iree.list<i32>
  // CHECK: = iree.list.get %[[LIST]][%c10] : !iree.list<i32>
  %1 = iree.list.get %list[%c10] : !iree.list<i32> -> i32

  // CHECK: %[[NEW_VALUE:.+]] = constant 100 : i32
  %new_value = constant 100 : i32
  // CHECK: iree.list.set %[[LIST]][%c10], %[[NEW_VALUE]] : !iree.list<i32>
  iree.list.set %list[%c10], %new_value : !iree.list<i32>

  return
}

// -----

// CHECK-LABEL: @list_access_tensor
// CHECK-SAME: (%[[LIST:.+]]: !iree.list<tensor<*xf32>>)
func @list_access_tensor(%list: !iree.list<tensor<*xf32>>) {
  %c10 = constant 10 : index

  // CHECK: = iree.list.get %[[LIST]][%c10] : !iree.list<tensor<*xf32>> -> tensor<?xf32>
  %0 = iree.list.get %list[%c10] : !iree.list<tensor<*xf32>> -> tensor<?xf32>

  // CHECK: %[[NEW_VALUE:.+]] = constant dense<1> : tensor<5xi32>
  %new_value = constant dense<1> : tensor<5xi32>
  // CHECK: iree.list.set %[[LIST]][%c10], %[[NEW_VALUE]] : tensor<5xi32> -> !iree.list<tensor<*xf32>>
  iree.list.set %list[%c10], %new_value : tensor<5xi32> -> !iree.list<tensor<*xf32>>

  return
}

// -----

// CHECK-LABEL: @list_access_variant
// CHECK-SAME: (%[[LIST:.+]]: !iree.list<?>)
func @list_access_variant(%list: !iree.list<?>) {
  %c10 = constant 10 : index
  %c11 = constant 11 : index

  // CHECK: = iree.list.get %[[LIST]][%c10] : !iree.list<?> -> i32
  %0 = iree.list.get %list[%c10] : !iree.list<?> -> i32

  // CHECK: %[[NEW_I32_VALUE:.+]] = constant 100 : i32
  %new_i32_value = constant 100 : i32
  // CHECK: iree.list.set %[[LIST]][%c10], %[[NEW_I32_VALUE]] : i32 -> !iree.list<?>
  iree.list.set %list[%c10], %new_i32_value : i32 -> !iree.list<?>

  // CHECK: = iree.list.get %[[LIST]][%c11] : !iree.list<?> -> tensor<5xf32>
  %1 = iree.list.get %list[%c11] : !iree.list<?> -> tensor<5xf32>

  // CHECK: %[[NEW_TENSOR_VALUE:.+]] = constant dense<1> : tensor<5xi32>
  %new_tensor_value = constant dense<1> : tensor<5xi32>
  // CHECK: iree.list.set %[[LIST]][%c11], %[[NEW_TENSOR_VALUE]] : tensor<5xi32> -> !iree.list<?>
  iree.list.set %list[%c11], %new_tensor_value : tensor<5xi32> -> !iree.list<?>

  return
}

