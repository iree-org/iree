// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=32})' %s | FileCheck %s

// CHECK-LABEL: vm.func private @my_fn
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !vm.ref<!hal.buffer_view>)
func.func @my_fn(%buffer_view: !hal.buffer_view) {
  // CHECK: %[[CAPACITY:.+]] = vm.const.i32 5
  %capacity = arith.constant 5 : index
  // CHECK: %[[LIST:.+]] = vm.list.alloc %[[CAPACITY]] : (i32) -> !vm.list<?>
  %list = util.list.create %capacity : !util.list<?>

  // CHECK: %[[NEW_SIZE:.+]] = vm.const.i32 100
  %new_size = arith.constant 100 : index
  // CHECK: vm.list.resize %[[LIST]], %[[NEW_SIZE]] : (!vm.list<?>, i32)
  util.list.resize %list, %new_size : !util.list<?>

  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index

  // CHECK: = vm.list.get.i32 %[[LIST]], %c10 : (!vm.list<?>, i32) -> i32
  %0 = util.list.get %list[%c10] : !util.list<?> -> i32

  // CHECK: %[[NEW_I32_VALUE:.+]] = vm.const.i32 101
  %new_i32_value = arith.constant 101 : i32
  // CHECK: vm.list.set.i32 %[[LIST]], %c10, %[[NEW_I32_VALUE]] : (!vm.list<?>, i32, i32)
  util.list.set %list[%c10], %new_i32_value : i32 -> !util.list<?>

  // CHECK: = vm.list.get.ref %[[LIST]], %c11 : (!vm.list<?>, i32) -> !vm.ref<!hal.buffer_view>
  %1 = util.list.get %list[%c11] : !util.list<?> -> !hal.buffer_view

  // CHECK: vm.list.set.ref %[[LIST]], %c11, %[[BUFFER_VIEW]] : (!vm.list<?>, i32, !vm.ref<!hal.buffer_view>)
  util.list.set %list[%c11], %buffer_view : !hal.buffer_view -> !util.list<?>

  // CHECK: %[[ZERO_CAPACITY:.+]] = vm.const.i32 0
  // CHECK: %[[LIST:.+]] = vm.list.alloc %[[ZERO_CAPACITY]] : (i32) -> !vm.list<?>
  %list_no_capacity = util.list.create : !util.list<?>

  return
}

// -----

// CHECK-LABEL: @list_construct_empty
func.func @list_construct_empty() {
  // CHECK: %[[CAPACITY:.+]] = vm.const.i32 0
  // CHECK: %[[EMPTY:.+]] = vm.list.alloc %[[CAPACITY]] : (i32) -> !vm.list<?>
  %empty = util.list.construct [] : !util.list<?>
  return
}

// -----

// CHECK-LABEL: @list_construct_i64
func.func @list_construct_i64() {
  // CHECK-DAG: %[[C100:.+]] = vm.const.i64 100
  %c100 = arith.constant 100 : i64
  // CHECK-DAG: %[[C200:.+]] = vm.const.i64 200
  %c200 = arith.constant 200 : i64
  // CHECK-DAG: %[[SIZE:.+]] = vm.const.i32 2
  // CHECK: %[[LIST:.+]] = vm.list.alloc %[[SIZE]] : (i32) -> !vm.list<i64>
  // CHECK: vm.list.resize %[[LIST]], %[[SIZE]]
  // CHECK: %[[I0:.+]] = vm.const.i32 0
  // CHECK: vm.list.set.i64 %[[LIST]], %[[I0]], %[[C100]]
  // CHECK: %[[I1:.+]] = vm.const.i32 1
  // CHECK: vm.list.set.i64 %[[LIST]], %[[I1]], %[[C200]]
  %list_i64 = util.list.construct [%c100 : i64, %c200 : i64] : !util.list<i64>
  return
}

// -----

// CHECK-LABEL: @list_construct_mixed
func.func @list_construct_mixed() {
  // CHECK-DAG: %[[C1_0:.+]] = vm.const.f64 1.0
  %c1_0 = arith.constant 1.0 : f64
  // CHECK-DAG: %[[NESTED_SIZE:.+]] = vm.const.i32 0
  // CHECK: %[[NESTED:.+]] = vm.list.alloc %[[NESTED_SIZE]]
  %nested = util.list.construct [] : !util.list<?>
  // CHECK-DAG: %[[SIZE:.+]] = vm.const.i32 2
  // CHECK: %[[LIST:.+]] = vm.list.alloc %[[SIZE]] : (i32) -> !vm.list<?>
  // CHECK: vm.list.resize %[[LIST]], %[[SIZE]]
  // CHECK: %[[I0:.+]] = vm.const.i32 0
  // CHECK: vm.list.set.f64 %[[LIST]], %[[I0]], %[[C1_0]]
  // CHECK: %[[I1:.+]] = vm.const.i32 1
  // CHECK: vm.list.set.ref %[[LIST]], %[[I1]], %[[NESTED]]
  %list_any = util.list.construct [%c1_0 : f64, %nested : !util.list<?>] : !util.list<?>
  return
}
