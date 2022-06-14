// RUN: iree-opt --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// CHECK-LABEL: @select_i32
vm.module @my_module {
  vm.func @select_i32(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @select_ref
vm.module @my_module {
  vm.func @select_ref(%arg0 : i32,
                      %arg1 : !vm.ref<?>,
                      %arg2 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: %ref = vm.select.ref %arg0, %arg1, %arg2 : !vm.ref<?>
    %ref = vm.select.ref %arg0, %arg1, %arg2 : !vm.ref<?>
    vm.return %ref : !vm.ref<?>
  }
}

// -----

// CHECK-LABEL: @switch_i32
vm.module @my_module {
  vm.func @switch_i32(%arg0 : i32) -> i32 {
    %c5 = vm.const.i32 5
    %c100 = vm.const.i32 100
    %c200 = vm.const.i32 200
    %c300 = vm.const.i32 300
    // CHECK: vm.switch.i32 %arg0[%c100, %c200, %c300] else %c5 : i32
    %0 = vm.switch.i32 %arg0[%c100, %c200, %c300] else %c5 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @switch_ref
vm.module @my_module {
  vm.func @switch_ref(%arg0 : i32) -> !vm.buffer {
    %0 = "make_buffer"() : () -> !vm.buffer
    %1 = "make_buffer"() : () -> !vm.buffer
    %2 = "make_buffer"() : () -> !vm.buffer
    %3 = "make_buffer"() : () -> !vm.buffer
    // CHECK: vm.switch.ref %arg0[%0, %1, %2] else %3 : !vm.buffer
    %4 = vm.switch.ref %arg0[%0, %1, %2] else %3 : !vm.buffer
    vm.return %4 : !vm.buffer
  }
}
