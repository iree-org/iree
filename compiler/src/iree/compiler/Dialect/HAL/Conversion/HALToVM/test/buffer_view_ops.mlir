// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: vm.func private @buffer_view_dims
// CHECK-SAME: %[[VIEW:.+]]: !vm.ref<!hal.buffer_view>
func.func @buffer_view_dims(%arg0 : !hal.buffer_view) -> (index, index, index) {
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
  return %0, %1, %2 : index, index, index
}
