// RUN: iree-opt <%s -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @Reserve
func @Reserve(%element_shape: !hal.buffer_view, %num_elements: !hal.buffer_view) -> !tensorlist.list {
  // CHECK: vm.call @tensorlist.reserve
  %0 = "tensorlist.Reserve"(%element_shape, %num_elements) { element_type = 50331680 : i32 } : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  return %0 : !tensorlist.list
}
// CHECK: vm.import @tensorlist.reserve

// -----

// CHECK-LABEL: @GetItem
func @GetItem(%list: !tensorlist.list, %index: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK: vm.call @tensorlist.get_item
  %0 = "tensorlist.GetItem"(%list, %index) : (!tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %0 : !hal.buffer_view
}
// CHECK: vm.import @tensorlist.get_item

// -----

// CHECK-LABEL: @SetItem
func @SetItem(%list: !tensorlist.list, %index: !hal.buffer_view, %item: !hal.buffer_view) -> !tensorlist.list{
  // CHECK: vm.call @tensorlist.set_item
  %0 = "tensorlist.SetItem"(%list, %index, %item) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  return %0 : !tensorlist.list
}
// CHECK: vm.import @tensorlist.set_item

// -----

// CHECK-LABEL: @Stack
func @Stack(%list: !tensorlist.list, %num_elements: !hal.buffer_view) -> !hal.buffer_view {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  // CHECK: vm.call @tensorlist.stack
  %0 = "tensorlist.Stack"(%allocator, %list, %num_elements) : (!hal.allocator, !tensorlist.list, !hal.buffer_view) -> !hal.buffer_view
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @Concat
func @Concat(%list: !tensorlist.list) -> !hal.buffer_view {
  %device = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  // CHECK: vm.call @tensorlist.concat
  %0 = "tensorlist.Concat"(%allocator, %list) : (!hal.allocator, !tensorlist.list) -> !hal.buffer_view
  return %0 : !hal.buffer_view
}
