// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @buffer_view_dims_lt4
func @buffer_view_dims_lt4(%arg0 : !hal.buffer_view) {
  // CHECK-NEXT: %{{.+}} = vm.call @hal.buffer_view.dims.1(%arg0) : (!vm.ref<!hal.buffer_view>) -> i32
  %0 = hal.buffer_view.dims %arg0 : index
  // CHECK-NEXT: %{{.+}}:2 = vm.call @hal.buffer_view.dims.2(%arg0) : (!vm.ref<!hal.buffer_view>) -> (i32, i32)
  %1, %2 = hal.buffer_view.dims %arg0 : index, index
  // CHECK-NEXT: %{{.+}}:3 = vm.call @hal.buffer_view.dims.3(%arg0) : (!vm.ref<!hal.buffer_view>) -> (i32, i32, i32)
  %3, %4, %5 = hal.buffer_view.dims %arg0 : index, index, index
  // CHECK-NEXT: %{{.+}}:4 = vm.call @hal.buffer_view.dims.4(%arg0) : (!vm.ref<!hal.buffer_view>) -> (i32, i32, i32, i32)
  %6, %7, %8, %9 = hal.buffer_view.dims %arg0 : index, index, index, index
  return
}

// -----

// CHECK-LABEL: @buffer_view_dims_gt4
func @buffer_view_dims_gt4(%arg0 : !hal.buffer_view) -> (index, index, index, index, index, index) {
  // CHECK: %0:4 = vm.call @hal.buffer_view.dims.4(%arg0)
  // CHECK: %1 = vm.call @hal.buffer_view.dim(%arg0, %c4)
  // CHECK: %2 = vm.call @hal.buffer_view.dim(%arg0, %c5)
  %0, %1, %2, %3, %4, %5 = hal.buffer_view.dims %arg0 : index, index, index, index, index, index
  // CHECK-NEXT: vm.return %0#0, %0#1, %0#2, %0#3, %1, %2
  return %0, %1, %2, %3, %4, %5 : index, index, index, index, index, index
}
