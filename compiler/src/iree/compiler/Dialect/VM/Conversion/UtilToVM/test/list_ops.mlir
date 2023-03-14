// RUN: iree-opt --split-input-file --iree-vm-conversion --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @list_ops
module @list_ops { module {
  // CHECK: vm.func private @my_fn
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
} }
