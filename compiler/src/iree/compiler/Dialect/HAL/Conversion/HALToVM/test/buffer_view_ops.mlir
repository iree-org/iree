// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @element_type
util.func public @element_type() -> i32 {
  // CHECK: %[[RET:.+]] = vm.const.i32 553648160
  %element_type = hal.element_type<f32> : i32
  // CHECK: vm.return %[[RET]]
  util.return %element_type : i32
}

// -----

// CHECK-LABEL: @encoding_type
util.func public @encoding_type() -> i32 {
  // CHECK: %[[RET:.+]] = vm.const.i32 1
  %encoding_type = hal.encoding_type<dense_row_major> : i32
  // CHECK: vm.return %[[RET]]
  util.return %encoding_type : i32
}

// -----

// CHECK-LABEL: vm.func private @buffer_view_dims
// CHECK-SAME: %[[VIEW:.+]]: !vm.ref<!hal.buffer_view>
util.func public @buffer_view_dims(%arg0 : !hal.buffer_view) -> (index, index, index) {
  // CHECK-DAG: %[[D0_64:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %zero)
  // CHECK-DAG: %[[D1_64:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %c1)
  // CHECK-DAG: %[[D2_64:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %c2)
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
  %2 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
  // CHECK-DAG: %[[D0_32:.+]] = vm.trunc.i64.i32 %[[D0_64]]
  // CHECK-DAG: %[[D1_32:.+]] = vm.trunc.i64.i32 %[[D1_64]]
  // CHECK-DAG: %[[D2_32:.+]] = vm.trunc.i64.i32 %[[D2_64]]
  // CHECK-NEXT: vm.return %[[D0_32]], %[[D1_32]], %[[D2_32]]
  util.return %0, %1, %2 : index, index, index
}
