// RUN: iree-opt -split-input-file -iree-vm-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @list_ops
module @list_ops { module {
  // CHECK: vm.func @my_fn
  // CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !vm.ref<!hal.buffer_view>)
  func @my_fn(%buffer_view: !hal.buffer_view) {
    // CHECK: %[[CAPACITY:.+]] = vm.const.i32 5
    %capacity = constant 5 : index
    // CHECK: %[[LIST:.+]] = vm.list.alloc %[[CAPACITY]] : (i32) -> !vm.list<?>
    %list = iree.list.create %capacity : !iree.list<?>

    // CHECK: %[[NEW_SIZE:.+]] = vm.const.i32 100
    %new_size = constant 100 : index
    // CHECK: vm.list.resize %[[LIST]], %[[NEW_SIZE]] : (!vm.list<?>, i32)
    iree.list.resize %list, %new_size : !iree.list<?>

    %c10 = constant 10 : index
    %c11 = constant 11 : index

    // CHECK: = vm.list.get.i32 %[[LIST]], %c10 : (!vm.list<?>, i32) -> i32
    %0 = iree.list.get %list[%c10] : !iree.list<?> -> i32

    // CHECK: %[[NEW_I32_VALUE:.+]] = vm.const.i32 101
    %new_i32_value = constant 101 : i32
    // CHECK: vm.list.set.i32 %[[LIST]], %c10, %[[NEW_I32_VALUE]] : (!vm.list<?>, i32, i32)
    iree.list.set %list[%c10], %new_i32_value : i32 -> !iree.list<?>

    // CHECK: = vm.list.get.ref %[[LIST]], %c11 : (!vm.list<?>, i32) -> !vm.ref<!hal.buffer_view>
    %1 = iree.list.get %list[%c11] : !iree.list<?> -> !hal.buffer_view

    // CHECK: vm.list.set.ref %[[LIST]], %c11, %[[BUFFER_VIEW]] : (!vm.list<?>, i32, !vm.ref<!hal.buffer_view>)
    iree.list.set %list[%c11], %buffer_view : !hal.buffer_view -> !iree.list<?>

    return
  }
} }
