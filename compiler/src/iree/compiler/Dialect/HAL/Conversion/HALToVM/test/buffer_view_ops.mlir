// RUN: iree-opt --split-input-file --iree-convert-hal-to-vm %s | FileCheck %s

// CHECK-LABEL: vm.func private @buffer_view_dims
// CHECK-SAME: %[[VIEW:.+]]: !vm.ref<!hal.buffer_view>
func.func @buffer_view_dims(%arg0 : !hal.buffer_view) -> (index, index, index) {
  // CHECK-DAG: %[[D0:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %zero)
  // CHECK-DAG: %[[D1:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %c1)
  // CHECK-DAG: %[[D2:.+]] = vm.call @hal.buffer_view.dim(%[[VIEW]], %c2)
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
  %2 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
  // CHECK-NEXT: vm.return %[[D0]], %[[D1]], %[[D2]]
  return %0, %1, %2 : index, index, index
}
