// RUN: tensorlist-opt <%s -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @Reserve
func @Reserve(%element_shape: !iree.ref<!hal.buffer_view>, %num_elements: !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list> {
  // CHECK: vm.call @tensorlist.reserve
  %0 = "tensorlist.Reserve"(%element_shape, %num_elements) : (!iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>
  return %0 : !iree.ref<!tensorlist.list>
}
// CHECK: vm.import @tensorlist.reserve

// -----

// CHECK-LABEL: @GetItem
func @GetItem(%list: !iree.ref<!tensorlist.list>, %index: !iree.ref<!hal.buffer_view>, %element_shape: !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view> {
  // CHECK: vm.call @tensorlist.get_item
  %0 = "tensorlist.GetItem"(%list, %index, %element_shape) : (!iree.ref<!tensorlist.list>, !iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view>
  return %0 : !iree.ref<!hal.buffer_view>
}
// CHECK: vm.import @tensorlist.get_item

// -----

// CHECK-LABEL: @SetItem
func @SetItem(%list: !iree.ref<!tensorlist.list>, %index: !iree.ref<!hal.buffer_view>, %item: !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>{
  // CHECK: vm.call @tensorlist.set_item
  %0 = "tensorlist.SetItem"(%list, %index, %item) : (!iree.ref<!tensorlist.list>, !iree.ref<!hal.buffer_view>, !iree.ref<!hal.buffer_view>) -> !iree.ref<!tensorlist.list>
  return %0 : !iree.ref<!tensorlist.list>
}
// CHECK: vm.import @tensorlist.set_item
