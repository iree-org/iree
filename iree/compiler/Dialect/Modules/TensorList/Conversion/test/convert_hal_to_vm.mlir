// RUN: iree-opt <%s -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @Reserve
func @Reserve(%element_shape: !hal.buffer_view, %num_elements: !hal.buffer_view) -> !tensorlist.list {
  // CHECK: vm.call @tensorlist.reserve
  %0 = "tensorlist.Reserve"(%element_shape, %num_elements) : (!hal.buffer_view, !hal.buffer_view) -> !tensorlist.list
  return %0 : !tensorlist.list
}
// CHECK: vm.import @tensorlist.reserve

// -----

// CHECK-LABEL: @GetItem
func @GetItem(%list: !tensorlist.list, %index: !hal.buffer_view, %element_shape: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK: vm.call @tensorlist.get_item
  %0 = "tensorlist.GetItem"(%list, %index, %element_shape) : (!tensorlist.list, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
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
